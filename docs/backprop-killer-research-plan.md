# Research Plan — Making DFA + Ternary a Backprop Killer on the Wall-Clock-at-Scale Axis

> Status: proposal / research direction. Not part of the "complete, no further development" baseline.
> North star chosen: **beat pipelined backprop on tokens/sec/GPU to a fixed target loss, on a real decoder-only Transformer**, by exploiting DFA's parallel (backward-lock-free) update.
> Resource reality: **laptop / CPU only**. So the laptop phase does not *win* — it earns the right to spend GPU money by proving the mechanism and producing a *validated* cost model.

---

## 0. The one idea this whole plan rests on

Pipelined backprop has a structural cost DFA does not: the **backward lock**. Stage *k*'s gradient needs the backprop signal from stage *k+1*, which forces a sequential backward sweep — pipeline *bubbles*, activations *stashed* across stages, and a *global sync* barrier. DFA broadcasts the output error `e` directly to every stage (`δ_l = (B_l e) ⊙ f'(z_l)`); **every stage updates in parallel with no dependency on later stages.** In principle that removes the backward bubble, cuts stashed-activation memory, shrinks inter-stage comm, and tolerates stale/async updates.

That is the *only* axis on which this idea can ever beat backprop. Accuracy, on every benchmark FeedFlipNets has run, is strictly worse. So the program is a single bet: **can we cash the parallel-backward systems win without losing so much accuracy (or re-locking the backward) that it evaporates?**

## 1. The binding critical path — accuracy-first, systems-second

The headline metric is *tokens/sec/GPU **to a fixed target loss***. That number is **undefined if DFA never reaches the target loss.** Therefore the critical path runs through accuracy, not systems:

1. **Build a shared testbed** (does not exist): a gradient-checked NumPy/torch‑CPU decoder-only Transformer with a *pluggable backward* (exact BP / DFA / structured‑DFA / Kolen‑Pollack), a real char-level TinyStories loader, and an alignment instrument. *This is a rewrite, not reuse* — the repo's `strategies.py` is a flat-MLP interface and `np_mlp.py` is hardcoded to one hidden layer.
2. **Resolve DFA-through-attention** — how `e` enters Q/K/V/O and how LayerNorm/softmax Jacobians interact with a broadcast error. **This is the actual scientific crux and no prior DFA work has solved it cleanly** (Launay 2020 found attention is exactly where DFA breaks).
3. **Establish a tuned BP baseline** that genuinely beats bigram/unigram floors (anti–"20NG BP = 19.86%" trap).
4. **Close the accuracy gap** on the Transformer via feedback learning, reporting *attention-block alignment separately*.
5. **Prove the same method that closes the gap keeps backward-decoupling ≥ 95%** (the coupling test).
6. **Calibrate the cost model** on *measured* FLOPs/bytes/step-times (<15% error), at *real vocab* (32k, not 96), *including the feedback-learning comm cost*.
7. **GPU gate**: projected tokens/sec/GPU lower-CI ≥ threshold, *conditional on step 4 holding*.
8. **GPU scale-up** confirms.

Theory (Crux 3b) rides *alongside* this path as a credibility multiplier and falsification harness — it must **never** be on the critical path to the GPU gate.

## 2. The biggest risk — the Conway / crux-coupling trap

The three cruxes are each *individually* likely solvable but may be **jointly incompatible, and structurally so**:

- The accuracy fix that actually works on a Transformer is **feedback *learning*** (Kolen‑Pollack / weight mirrors) — because fixed random feedback's alignment floor caps quality by construction.
- KP's selling point ("the B-update is local, needs only the broadcast error") is **the unproven open question, not an established result.** In DFA the local pseudo-delta driving the KP increment is `B·e`, so `B` tracks *the matrix that produces the DFA delta*, not a faithful `Wᵀ`. Whether that closes the gap on **attention** (whose alignment target is input-conditioned and *moves every forward pass*) is unknown.
- If KP closes the gap only by becoming **non-local**, or closes it on MLP but **not attention**, then **no single configuration is simultaneously accurate *and* decoupled** → the headline metric is undefined.

Every other risk (bubble-only 1.24×, calibration error, depth-decay misattribution) merely shrinks the margin. **This one is binary and kills the north star.** It also surfaces *latest* — only at the P3 coupling test, ~10 weeks in — so the coupling test must be designed in from P0 and run on the **same** method, with **no escape hatch** (any method with backward-decoupling < 0.9 is disqualified from the systems claim regardless of accuracy).

## 3. The honest "well-tuned baseline" math

Against a *naive* GPipe baseline the numbers look great. Against a **properly tuned 1F1B + activation-checkpointing + ZeRO** baseline they shrink hard:

- Bubble fraction `β_BP = (S−1)/(m+S−1)`. At the standard operating point `m ≈ 4S`, `β ≈ 0.19` → the **bubble-only** DFA win is just **~1.24×**. Cherry-picking `m ≈ S` to claim "1.9×" is a strawman and is itself a kill condition.
- The defensible win must come from the **activation-memory** edge (DFA frees activations at forward-completion: `O(1)` vs 1F1B's `O(S)`) *enabling* a config (bigger `m`, deeper `S`, no recompute) that tuned BP cannot match at the same memory budget — **and that edge must survive checkpointing.**
- DFA does **not** remove the *forward* fill/drain bubble, and `e` is only available after forward reaches stage `S`; the unlock is *across microbatches* (≈ the 1F1B interleaving advantage), so the marginal win over 1F1B is smaller than over GPipe.
- DFA's per-stage update is an **extra matmul** (`B_l e`). Charge `t_u` honestly; `t_u ≪ t_b` is assumed, not derived.
- At real LM vocab (32k) the `e`-broadcast is **not** free; re-run the cost model at real vocab or risk a false GPU-gate pass.

**Every cost-model comparison defaults to tuned 1F1B+checkpointing+ZeRO, never vanilla GPipe.**

---

## 4. Workstreams

Each is laptop-runnable; each ends in a kill criterion so the program fails fast.

### P0 (shared, owned by one person) — Testbed + correctness gates  · 2–3 weeks
The highest-leverage *unowned* item: every workstream secretly assumes this exists. If each builds its own, FLOP counts, alignment numbers, and accuracy gaps come from incompatible codepaths and cannot be cross-compared.

- **torch-CPU** (not pure NumPy) pre-LN decoder-only Transformer; use **autograd for exact BP and for the shadow/true-gradient** the alignment instrument needs (hand-writing attention backward across 4 strategies in NumPy is weeks of debugging that buys nothing).
- Char-level TinyStories loader (contiguous id stream, frozen train/val split). *Current `datasets/tinystories.py` returns whitespace string tokens — unusable as an LM; rewrite.*
- Pluggable backward `Strategy`: BP / DFA / structured-DFA / KP, **plus a FLOP + cross-stage-byte counter** (none exists in repo today).
- Alignment instrument: a no-update shadow BP backward producing true `g_l` per block.
- **GATE-0 — GO if** every strategy's δ matches finite-difference/autograd to <1e-4 on a tiny net; alignment instrument passes controls (BP-vs-BP ≈ 0°, shuffled-`e` ≈ 90°); 200-step smoke run finite. **No science number is trusted until this is green.** **KILL if** a backward can't be verified in the 2–3 week box.

### Crux 1 — Systems: pipeline-parallel cost model + wall-clock thesis
**Objective:** turn parallel-backward into a *validated* cost model that predicts a GPU-scale win — a gate, not a victory.
- **E1** Analytical model (`feedflipnets/systems/cost_model.py`) for GPipe, **1F1B+checkpointing+ZeRO (default comparator)**, and DFA-pipeline; emits bubble fraction, peak activation memory, comm bytes, tokens/sec over an `(S,m)` grid. Must include an **`e`-availability-latency** term and the **CRUX-3 feedback-learning comm term** from day one.
- **E2** Real S-process CPU pipeline (`pipe_sim/`, stdlib `multiprocessing`) with a **1F1B** BP engine (not just GPipe-stash) and a DFA engine; instrument idle/bubble, peak activation buffers, inter-process bytes. **Model-validation gate.**
- **E3** Async/staleness probe — benchmark DFA-async vs **PipeDream-2BW-async** (the matched stale baseline), not GPipe-sync.
- **E4** Calibration transfer card: swap CPU primitives for roofline-from-spec GPU numbers (CPU-computable), propagate E2 error → projected ratio **with CI**.
- **KILL** modeled end-to-end speedup < 1.5× at the best realistic `(S,m,mem-cap)` **against the tuned baseline, with the feedback-learning comm term and honest `t_u`** · OR the only win is the ~1.24× bubble residue once checkpointing erases the memory edge · OR E4 lower-CI < 1.5× → do not spend on GPUs.

### Crux 2 — Method: depth-stable, input-conditioned feedback for Transformers
**Objective:** feedback whose per-layer alignment `ρ_l` does **not** decay with depth and carries attention's structure.
- Establish the **negative control**: fixed-random DFA's `ρ_l` vs depth (expect collapse toward the random-vector noise floor ~`1/√dim`; attention blocks collapse first). *Note: the repo's DFA projects `e` directly per layer, so the `c/√L` decay is a feedback-alignment/Refinetti property that may not even reproduce here — pre-register this control.*
- **Candidate A**: depth-normalized / orthogonal-rescaled per-stage `B` (fights the `1/√L` variance floor; structure-agnostic).
- **Candidate B**: input-conditioned feedback for attention (Launay-style per-module routing).
- **Discriminator**: rank constructions by **min `ρ_l` vs depth (slope, not single-point)** with attention split out, *and* measure each construction's backward FLOP/byte overhead.
- **KILL** no construction holds min-block AND attention-block alignment non-decaying to L=16 above fixed-DFA · OR the only one that does needs >2× backward FLOPs or explicit weight transport (forfeits the systems win) · OR alignment holds but val-bpc gap to tuned BP stays > 0.3 bpc (alignment decoupled from trainability → diagnostic not predictive).

### Crux 3a — Method: feedback learning to close the accuracy gap
**Objective:** find the variant on the **accuracy-gap-vs-added-cost** frontier that closes the gap *and* preserves backward-decoupling.
- Implement as drop-in strategies: **Kolen-Pollack / weight mirrors** (primary bet), **DRTP** (cheap lower-bound — but do *not* hardcode it as the floor; it sometimes beats DFA), **predictive coding / target prop** (upper "can recover BP" reference — guard against a *weak* PC impl with an oracle gate).
- For each: accuracy-gap to **tuned** BP, and a **falsifiable parallelism-preservation score** = `1 − (cross-stage sequential FLOPs / forward FLOPs)` **plus a hard binary "reads any downstream-layer tensor in backward?"** (YES → score 0, disqualified regardless of accuracy).
- Give the feedback methods their **own** hyperparameter grid (`η_B`, decay, mirror period) — "compute-matched to BP" must not silently mean "untuned KP."
- **KILL** even oracle-gated PC can't close the gap (gap isn't a feedback-quality problem → wrong lever) · OR the only gap-closer reintroduces a sequential backward dependency / full transport · OR KP's attention-block alignment merely *ties* fixed-DFA at the depth where `1/√L` bites (feedback learning bought nothing for the hardest sublayer).

### Crux 3b — Theory (off the critical path)
**Objective:** remove the circularity in the existing theorem, which *assumes* `p>½` and `ρ_l` instead of deriving them. Every assumption maps to a quantity the repo already logs, so each bound is laptop-falsifiable.
- **T1** Derive sign-advantage from alignment via the Gaussian arcsin identity `p = ½ + (1/π)·arcsin(ρ_eff)`. *Watch the observable mismatch:* `trainer.py`'s `phat` is vector-level on the quantized update, not the per-coordinate Gaussian pair the identity models — validate against both a synthetic-pair control and the repo `phat`.
- **T2** Derive the alignment floor `γ(L, B, Σ)` for deep-linear DFA: `Θ(1/√L)` for Gaussian `B`, `Ω(1)` for structured `B`.
- **T3** Non-vacuous rate for 1-hidden nonlinear DFA + an **STE smoothness fix** (soft surrogate's bias `B_τ → 0` as temperature → hard quantizer, *or* a Kushner-Yin drift bound) — and state which object the headline rate uses.
- **T4** Separation / solution-quality: when is DFA's fixed point strictly worse than BP, and does KP provably send `γ→1` **using only stage-local tensors**? (This locality proof is the only on-thesis deliverable here — make it a hard gate.)
- **KILL-northstar (the missing gate that ties theory to the goal):** derive `1/c(γ)` for structured `B` and compare to Crux 1's bubble-speedup factor. If DFA's per-token deficit exceeds the parallel-backward speedup, DFA loses on wall-clock *even with perfect parallelism* — theory is irrelevant no matter how clean.

### Integrating first-signal experiment (all three cruxes, one run)
A 6-block char-level TinyStories Transformer with the best Crux-2 feedback + best Crux-3a feedback-learning, instrumented for alignment-vs-depth and the cost-model proxy. Pre-registered numeric gates: BP ≤ 1.8 bpc · best-method alignment ≤ 60° and non-decaying (slope ≥ −2°/block) at L=8 · val-bpc gap ≤ 0.15 (95% CI excludes > 0.25), **n ≥ 5 seeds** · ≥ 95% backward-decoupling on that *same* method · calibrated model predicts ≥ 1.3× tokens/sec/GPU at ≥ 16 stages. **Integrating kill:** no single config satisfies alignment *and* accuracy *and* decoupling at once.

---

## 5. Phasing and go/no-go gates

| Phase | Weeks | Work | Exit gate |
|---|---|---|---|
| **P0** | 2–3 | Shared testbed + correctness/alignment instrument | **GATE-0**: all backward strategies grad-checked <1e-4; instrument passes controls |
| **P1** | 2–3 | Tuned BP baseline + DFA-through-attention construction + fixed-DFA depth control (pre-registered) | **GATE-1**: BP ≤ 1.8 bpc (beats bigram); attention feedback specified & grad-checked; a real gap exists |
| **P2** | 4–6 | Parallel: depth-stable feedback (C2) ∥ feedback-learning (C3a) ∥ theory T1/T2 (C3b) | **GATE-2**: ≥1 construction holds attention-block alignment non-decaying at <2× FLOPs AND a tuned gap-closer beats fixed-DFA attention alignment |
| **P3** | 3–4 | Integration + **coupling test** + cost-model calibration at real vocab | **GATE-3 (THE GPU GO/NO-GO)** |
| **P4** | few-hundred GPU-h | GPU scale-up (350M–1.3B, S∈{4,8,16}) vs tuned 1F1B BP | **GATE-4**: measured ≥1.5× at S≥8, loss within margin, **win grows with S** |

**GATE-3 — GO if** one seed-averaged method (n≥5) hits the bpc gap within margin **AND** backward-decoupling ≥ 0.95 on that *same* method **AND** steps-to-target ≤ ~2× BP **AND** the calibrated model (validated <15%, re-run at vocab 32k, with feedback-learning comm + honest `t_u`) predicts tokens/sec/GPU lower-CI ≥ 1.5× vs the **tuned** baseline, win growing in S. **KILL if** the accuracy fix re-locks the backward (Conway trap fires) · OR lower-CI < 1.5× at real vocab · OR the modeled win is purely bubble-removal once 1F1B+checkpointing erases the memory edge. **Do not rationalize past this tripwire into GPU spend.**

## 6. Cross-workstream gaps to assign an owner *now*

1. **The testbed itself is unowned** — one named owner delivers ONE testbed + ONE backward interface + ONE alignment instrument in P0 before any science starts.
2. **DFA-through-attention construction** — load-bearing for C2, C3, and theory; currently specified by no one.
3. **FLOP/comm instrumentation** — assumed by C1 and C3, does not exist; analytic FLOPs won't reconcile with wall-clock samples/sec (different units) — define the reconciliation.
4. **One frozen baseline contract** — which BP schedule, which knobs tuned, which floors to beat — owned centrally or every gate is corrupted by a silent strawman.
5. **The coupling test** — one hard owned gate; **no "KP may have lower decoupling if it buys accuracy" escape hatch.**
6. **Real-vocab comm honesty** — re-run the cost model at vocab 32k (config-S's ~96 makes the `e`-broadcast artificially cheap).
7. **Statistical power** — pre-register seed count vs margin (MNIST-MLP seed std alone is ~0.2–0.5 pp; 3 seeds can't resolve a 1 pp gate).
8. **Async laptop proxy** — a cheap staleness proxy should gate GPU spend; the matched async baseline is PipeDream-2BW, not GPipe-sync.

## 7. Honest odds

Fund through **GATE-3** as a ~12–15 week bounded bet whose worst case is still a strong paper and a reusable testbed.

- **~55–65%** — a strong, publishable **negative/characterization result**: "the first honest, gradient-checked, compute-matched pipeline-parallel DFA testbed for decoder Transformers; here is exactly where and why the parallel-backward win fails to cash out (attention-alignment collapse and/or the accuracy-fix-relocks-the-backward coupling)." Genuinely valuable and very likely true regardless of which way the cruxes break.
- **~20–30%** — a **niche/conditional win**: DFA-pipeline beats tuned pipelined BP in a specific regime (very deep pipelines, memory-bound stages, slightly loosened margin, non-attention-dominated workload). Real, but regime-bound with an honest asterisk.
- **~5–10%** — a clean ≥1.5× wall-clock win at scale, accuracy within margin, with the depth-scaling signature — the "actual backprop killer on one axis" outcome.
- **<2%** — broadly displaces backprop (not even the stated goal).

The single fact that compresses the upside: against a properly tuned 1F1B+checkpointing+ZeRO baseline, the bubble win alone is ~1.24× at `m=4S`, so the entire headline rests on a **four-way conjunction** — the activation-memory edge being real AND large AND surviving the feedback-learning comm cost AND the accuracy gap being closeable *on attention*.

## 8. What to build first (week 1)

1. `datasets/tinystories.py` → real char-level loader (contiguous ids, frozen split).
2. torch-CPU pre-LN decoder-only Transformer with a pluggable backward `Strategy` + autograd-based exact BP and shadow-gradient.
3. The finite-difference grad-check harness (GATE-0) — **green before any alignment or accuracy number is recorded.**
4. The FLOP + cross-stage-byte counter.
5. Pre-register GATE-1/2/3 thresholds and seed/power plan in this file.
