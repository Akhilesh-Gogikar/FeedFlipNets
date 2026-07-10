# FeedFlip: feedback-driven bit-flips for ternary nets — and the per-weight sign barrier

The real thesis of FeedFlipNets: train ternary networks by using a cheap feedback signal to **flip
bits directly**, with **no float shadow weights** — store only ternary `W ∈ {-1,0,+1}` (≈1.58 bits)
plus a small signed integer flip-accumulator `c` per weight. Each step: `c += -sign(grad_est)`; when
`|c| ≥ K`, flip the bit (`-1↔0↔+1`) and reset. A flip needs only the **sign**, so the natural
feedback is DFA (cheap, transport-free, shadow-free). Prototypes: `experiments/feedflip_bitflip.py`
(MLP), `experiments/ternary_cpu_headtohead.py`, and the transformer flip study (scratchpad).

## 1. Is DFA a good CPU ternary trainer? (gradient-descent framing) — No.
Fair head-to-head with **absmean-scaled** ternary (BitNet/TWN recipe; raw ±1 blows up BP+STE),
deep MLP, n=3:

| arm | final acc | ms/step | peak KB | wall→target |
| --- | --------- | ------- | ------- | ----------- |
| bp_float    | 0.645 | 0.84 | 1629 | 0.13s |
| bp_ternary  | 0.624 | 2.16 | 1948 | **0.45s** |
| dfa_float   | 0.568 | 0.75 | 1607 | — |
| dfa_ternary | 0.584 | 1.99 | 1926 | **1.11s** |

BP+ternary wins on accuracy and is ~2.5× faster to a target (DFA needs ~2.75× more steps); memory is
identical; ternary gives no NumPy speedup. As a *gradient method*, DFA adds nothing.

## 2. The FeedFlip mechanism (bit-flips, no shadow) — works with a good sign, and is efficient.
Deep MLP, n=3, sign-driven flips vs a float-shadow BP anchor:

| arm | final acc | ms/step | state bits/w |
| --- | --------- | ------- | ------------ |
| bp_shadow (float32)     | 0.619 | 2.06 | 32.0 |
| **feedflip_bp (K=8)**   | **0.629** | 1.03 | **5.67** |
| feedflip_dfa (K=8)      | 0.313 | 0.81 | 5.67 |
| feedflip_dfa (K=32)     | 0.496 | 0.81 | 7.60 |
| feedflip_dfa+ortho (K=8)  | 0.303 | 0.80 | 5.67 |
| **feedflip_dfa+ortho (K=32)** | **0.548** | 0.81 | 7.60 |

Sign-driven flips with a **good (BP) sign** match float-shadow BP at **5.6× less state and ~2× faster**.
Transport-free (DFA) flips are viable *only partially* on the MLP: vote-averaging (K) + orthogonal
feedback compound to 0.548 (within ~0.08 of BP), transport-free, at ~4× less state.

## 3. Loop-closing test on a TRANSFORMER — transport-free flipping fails; AR-DFA does not rescue it.
Stacked ternary char-LM (CFG corpus, floor 2.88 bpc, no float shadow), n=3:

| arm | val bpc | vs floor | bits/w | per-weight sign-match `p` |
| --- | ------- | -------- | ------ | ------------------------- |
| shadow_BP (float32)  | **1.12** | −1.76 | 32.0 | — |
| BP_flip (exact sign) | **1.72** | −1.16 | 5.67 | — |
| FROZEN (blocks never flip) | 2.01 | −0.87 | 5.67 | — |
| AR-DFA_flip K8   | 3.28 | +0.40 | 5.67 | 0.53–0.56 |
| AR-DFA_flip K32  | 2.80 | −0.08 | 7.60 | — |
| AR-DFA+② K8      | 3.34 | +0.46 | 5.67 | — |
| fixed-DFA_flip K8  | 3.21 | +0.33 | 5.67 | 0.52–0.54 |
| fixed-DFA_flip K32 | 2.95 | +0.07 | 7.60 | — |

- **The flip mechanism works with a good sign:** `BP_flip` (1.72) beats the floor and the FROZEN control.
- **Transport-free flipping FAILS on the transformer:** every DFA/AR-DFA arm is **worse than FROZEN
  (2.01)** — flipping bits from the DFA sign is worse than not flipping. K32 only claws back to ~floor.
- **AR-DFA does not rescue it:** it ties fixed-DFA, and per-weight sign-match `p ≈ 0.53` for **both** —
  barely above chance.

## 4. Pre-registered attack on the sign barrier (MLP rebuild) — NO-GO; the barrier held.
The §2 prototype was never committed. `experiments/feedflip_bitflip.py` is a committed,
reproducible rebuild: task difficulty calibrated on **baseline arms only**, then frozen
(D_IN=32, C=10, MLP [32,64,64,64,64,10], 8000 steps, batch 16, n_train 16384, n=3);
attack arms and the GO/NO-GO gate (any transport-free arm ≥ 0.60 mean acc at ≤ 8 bits/w)
were pre-registered in the file docstring before any attack was run.

Frozen baselines (`baselines.json`): bp_shadow **0.604** ± .013 (32 bits/w) ·
dfa_k32 0.536 · dfa_ortho_k32 0.534 (7.6 bits/w) · feedflip_bp_k8 0.489 ·
dfa_ortho_k8 0.483 · dfa_k8 0.473 (5.67 bits/w). Rebuild caveat: unlike the §2 table,
BP-sign flips here **lag** the float-shadow anchor (0.489 vs 0.604) — the uncommitted
0.629 config could not be recovered, so §2's feedflip_bp≈bp_shadow claim should be
treated as unreproduced until shown otherwise.

Attack arms (`attacks.json`, all 7.6 bits/w, n=3):

| arm | mean acc | sign-match `p` |
| --- | -------- | -------------- |
| combo (gate+anneal+ortho) | **0.541** ± .010 | **0.747** |
| conf_gate   | 0.539 ± .009 | 0.717 |
| k_anneal    | 0.534 ± .001 | 0.717 |
| taught_B ②  | 0.507 ± .012 | **0.572** |
| k_layerwise | 0.490 ± .042 | 0.705 |

**Gate: NO-GO** (best 0.541 < 0.60). Two diagnostics survive the negative result:
- **Sign-match is not the binding constraint at this margin:** combo raised `p`
  0.71→0.75 yet gained only +0.005 over dfa_k32 — within seed noise.
- **taught_B ② learns a *different* direction, not a better one:** `p` drops to 0.57
  (below fixed-B) while accuracy stays at baseline — perturbation-taught feedback
  stops imitating BP without finding a superior transport-free signal.

## 5. Round 2 — the feedback-information budget (pre-registered): sign fidelity is NOT the bottleneck.
`experiments/feedflip_feedback_budget.py`; pre-registration + gates frozen at commit
`7075d13` (`PREREG_round2_feedback_budget.md`) **before** any full run. Same frozen
benchmark as §4, K=32, n=3. New axis: bits of information flowing from W into the
feedback path (`fb_comm` bits/w/step; one ternary W^T refresh = log2(3) ≈ 1.585,
refreshed every k steps). Results (`feedflip_round2.json`):

| arm | fb_comm (bits/w/step) | mean acc | sign-match `p` (late) |
| --- | --------------------- | -------- | --------------------- |
| bp_shadow (§4 anchor, 32 bits/w state) | — | 0.604 ± .013 | — |
| sign_transport k=1 (bit-exact BP votes) | 1.585 | 0.551 ± .002 | **1.000** |
| sign_transport k=8    | 0.198 | 0.555 ± .008 | 0.999 |
| sign_transport k=64   | 0.025 | 0.542 ± .007 | 0.992 |
| sign_transport k=512  | 0.003 | 0.557 ± .006 | 0.974 |
| sign_transport k=4096 | 0.0004 | 0.557 ± .005 | 0.870 |
| fa_random (control)   | 0 | 0.530 ± .004 | 0.717 |
| kp_fa (Kolen–Pollack mirror) | 0 | 0.304 ± .037 | 0.328 |

**Both gates NO-GO** (G-A1 kp_fa ≥ 0.60: no; G-A3 any k ≥ 8 ≥ 0.60: no) — and the
pre-registered interpretation rule **P5 fired**: even k=1, which feeds the flip
mechanism the *bit-exact BP backward pass* (`p = 1.0` by construction), reaches only
0.551.

- **The §4 "sign barrier" framing is overturned.** Going from p ≈ 0.70 (random B) to
  p = 1.0 (perfect transport) buys only +0.015 of the 0.07 gap to bp_shadow. The
  binding constraint is the magnitude-blind vote/flip accumulator, not feedback sign
  fidelity: what the float shadow buys is *magnitude accumulation*, not better signs.
- **Sign fidelity is nearly free anyway:** acc(k) is flat out to k=4096
  (0.0004 bits/w/step, late p 0.87, acc 0.557) — stale ternary W^T is as good as
  fresh transport. P1 (monotone decay in k) is not observed in this regime.
- **kp_fa diverged** (P3 falsified): gradient co-mirroring under flip dynamics
  bootstraps its own feedback target (late p 0.33, acc 0.30); the KP theorem's
  shared-update premise genuinely matters — W moving by flips breaks it.
- **P4 confirmed:** fa_random (0.530) ≈ dfa_k32 (0.536) — chaining topology alone
  changes nothing.

Consequences: registered arms A2 (prealigned init) and A4 (forward-gradient control
variate) target sign fidelity and are now predicted capped near 0.55; A5 (wall-clock
throughput) is unaffected. A round-3 attack should target the *mechanism*:
magnitude-carrying votes (e.g. `c += -sign(g)·q(|g|)` with a 2–3-bit quantizer,
or stochastic-rounding flips with P(flip) ∝ |g|), staying ≤ 8 state bits/w.

## The unifying insight: cosine alignment ≠ per-weight sign
AR-DFA's large *cosine*-alignment gain on attention (worst-case angle 90°→46°, `data/report/m2/`)
does **not** translate into per-weight *sign* correctness (`p ≈ 0.53`, barely above chance). Bit-flips
use only the sign, so the alignment gain is invisible to them. **The binding constraint for
transport-free discrete/ternary learning is per-weight sign accuracy, and "alignment" (cosine — the
standard DFA metric) is the wrong proxy for it.** The FeedFlip *mechanism* is genuinely efficient
(≈5.6× less state, ~2× faster, shadow-free); the open problem is a transport-free rule whose
per-weight sign — not just its aggregate direction — is reliably correct at depth and through attention.

**Round-2 revision (§5):** on the MLP benchmark, sign accuracy is *necessary but not sufficient* —
even p = 1.0 (bit-exact BP votes) recovers only 0.551 vs bp_shadow's 0.604. Beyond sign
correctness, the deeper constraint is the flip accumulator's magnitude-blindness: the
K-threshold vote counter discards |g|, which is exactly what the float shadow retains.
