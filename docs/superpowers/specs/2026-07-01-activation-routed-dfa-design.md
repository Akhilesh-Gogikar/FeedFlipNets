# Design Spec — Activation-Routed DFA (①) + Perturbation-Taught Feedback (②)

**Date:** 2026-07-01
**Status:** Approved design, pre-implementation
**Topic:** The next evolution of Direct Feedback Alignment — a credit-assignment mechanism that keeps per-layer alignment high **at depth** and **through attention**, while preserving DFA's lock-free backward.

---

## 1. Goal

Replace DFA's fixed random feedback `B` with a mechanism that fixes the two failure modes that make plain DFA lose to backprop as models scale:

- **Depth decay** — fixed random feedback's alignment floor scales like `c/√L`; the angle between the DFA pseudo-gradient and the true gradient worsens monotonically with depth.
- **Attention's moving target** — self-attention's effective Jacobian is input-conditioned (`softmax(QKᵀ/√d)V` re-routes credit every forward pass), so one fixed `B` per layer cannot stay aligned across the input distribution.

We attack them with two composable mechanisms — **① Activation-Routed feedback** (structure, ≈ free) and **② Perturbation-Taught feedback** (transport-free learning) — validated on a laptop via alignment-vs-depth curves and accuracy gap to a *tuned* backprop baseline.

Non-goal for this spec: the pipeline-parallel systems/wall-clock win (separate track, see `docs/backprop-killer-research-plan.md`), GPU scale-up, ternary-LLM-beyond-STE.

## 2. Hard invariant (enforced, not aspirational)

**Inside `backward`, the only tensors that may cross a layer boundary are the two dimension-independent broadcast channels: (i) the shared output error `e`, and (ii) a scalar loss/delta `ΔL` (②'s perturbation signal).** No strategy may read any *other* downstream-layer tensor — specifically **no `W_{l+1}ᵀ` and no per-layer `δ_{l+1}`**. Reusing a layer's *own* (or upstream) forward-cached tensors is allowed and is the basis of ① (plain DFA already reuses `f'(z_l)`, a cached activation). This is the only property that makes DFA worth evolving over backprop.

**Why the invariant is subtle (and why the naive probe is wrong).** `e` is the network's *output* error, so it is a global function of *all* weights, including downstream ones. Therefore perturbing a downstream weight and re-running the forward changes `e`, which changes `grad_l` even for correct lock-free DFA — a naive "perturb downstream weights, assert `grad_l` unchanged" probe would falsely flag DFA as a violation. The invariant is about the *computational dependency inside `backward`*, with `e` and `ΔL` whitelisted as sources, **not** about end-to-end sensitivity. It is enforced by the two-part lock-free probe in §5.4.

## 3. Mechanism ① — Activation-Routed DFA ("reuse the free half of the Jacobian")

For a pre-LN decoder block, the block Jacobian factors into two kinds of maps:

- **Weight maps** (`W_Q, W_K, W_V, W_O, W_1, W_2`) — using their transpose is *weight transport*; **forbidden**. Replaced by a fixed-random (②-adapted) **surrogate** of the transposed shape.
- **Data maps** — `A = softmax(QKᵀ/√d)`, the row-wise softmax Jacobian `J_A = diag(a) − aaᵀ`, LayerNorm scale `γ/σ`, the MLP nonlinearity mask `φ'`, and (optional) the ternary dead-zone mask. **All are cached in the forward pass, so reusing them is not transport.** Reused **exactly**.

**Mechanism.** The error entering a block is the direct broadcast of top error `e`, projected to `d_model` by a DFA matrix `B_block` (keeps blocks independent given `e` → cross-layer lock-free). *Within* the block, that error is routed to each projection's gradient through the cached data maps exactly (`A`, `J_A`, LN scale, `φ'`), using the random/adapted surrogates only where a weight-transpose would otherwise appear. Net effect: **the input-conditioned *mixing* (`A`, `J_A`, LN scale, `φ'`) is tracked exactly; the input-conditioned *projection* through `W_{Q,K,V,O,1,2}` is approximated via surrogates**, since it factors as (fixed weight) × (input) and the weight half is forbidden. So ① recovers only the softmax/LN/nonlinearity Jacobian factors, not the full attention Jacobian — but those factors are exactly where the moving target lives.

**Scope of ①'s win is asymmetric — measure it.** The exactness benefit is strong on the **value / `A`-contraction path** (`W_V`, `W_O` gradients see the exact `A`), but weak on the **score path** (`W_Q`, `W_K` gradients reach the softmax only *through* the random `W_V`/`W_O` surrogates in series, so they may collapse to plain-DFA quality). The alignment probe (§5.4) therefore reports `ρ` broken out per path (`{W_V,W_O}` vs `{W_Q,W_K}`), and we **pre-register that ①'s win may be value-path-only** until ② adapts the `W_V`/`W_O` surrogates.

**Which mechanism owns which failure mode (do not conflate).** ① lifts the *level* of per-block alignment (intra-block structure). It does **nothing** to the `c/√L` **depth-slope**, which lives in the composition of independent random `B_block`s across depth and is owned by ② (adapting `B_block`). See §6/M2 for the split gate.

**Contrast with baselines.** Plain DFA: `δ_l = (B_l e) ⊙ f'(z_l)`, one fixed random `B_l`, no attention structure. ①: `e` broadcast per block, then intra-block routing via cached `A`/`J_A`/LN — the softmax mixing is carried, not averaged away.

## 4. Mechanism ② — Perturbation-Taught Feedback ("learn B with zero transport")

`B_block` (and intra-block surrogates) start random and are **adapted transport-free** via node perturbation:

1. Perturb a block's pre-activation (dimension `d_pert`, = `d_model` unless perturbing a bottleneck) by `ξ ~ N(0, ρ²I)`; run the (partial) forward from that block onward to get `ΔL = L(perturbed) − L(base)`.
2. Form the directional estimate `ĝ = (ΔL / ρ²)·ξ` of the true gradient w.r.t. that activation. It is unbiased to leading order with an `O(ρ²)` curvature bias; **antithetic `±ξ` cancels that bias** (it is a *bias* control, not a variance control).
3. Adapt `B_block` toward `ĝ` with a **scale-invariant, direction-only** step (both operands unit-normalized, since alignment `ρ` is a cosine and we must not couple `B` to the loss scale): `dB ∝ (ĝ/‖ĝ‖ − B·e/‖B·e‖) eᵀ`. A unit test asserts the learned `B*` is invariant to `ρ`.

**Variance model (this sets M1's budget).** `ĝ` is a node-perturbation estimate of a `d_pert`-dimensional gradient, so `Var(ĝ_total) ≈ ‖g‖²·d_pert` and a single sample's `cos(ĝ,g) ~ 1/√d_pert`. The variance is set by `d_pert`, **not** by `B`'s rank — where `ĝ` lands downstream does not shrink it. Feasibility comes purely from **averaging over samples**: reaching cosine `ρ_target` between the *averaged* estimate and `g` needs `N ∝ d_pert` samples per block (order-of-magnitude; the exact constant is model-dependent — derive and record it in M1, do not quote a single authoritative number). Real variance reduction is therefore **mandatory**: (i) accumulate an **EMA of `ĝ` per block** so samples compound across round-robin visits, and/or `K` perturbations/block/step; (ii) a common-random-number / baseline **control variate** for `ΔL`; (iii) couple `B`'s step size to a running SNR estimate. Cost: one extra *partial* forward (perturbed block → output) per perturbed block; amortized round-robin (one block/step).

**Fixed point and its ceiling (scope ② honestly).** The update's expected fixed point is the linear regression `B* = E[g eᵀ] E[e eᵀ]⁻¹` — a *fixed linear* map from `e` to `g`. A fixed `B` therefore **cannot** track an input-conditioned (per-forward-pass-moving) credit direction; that job belongs to ①'s cached data maps (`A`, `J_A`). So ②'s role is scoped to adapting the **input-independent weight-surrogate rotation** and killing the cross-block `c/√L` depth decay (the way weight-mirrors/Kolen-Pollack do), **without ever touching `Wᵀ`**. Making `B` itself input-conditioned (a small hypernetwork of `e` or cached features) is a larger design change — flagged, not assumed.

Composition: ① supplies the structured intra-block update and the input-conditioned mixing; ② slowly improves the input-independent `B_block`/surrogate rotation and bends the depth-slope. Either runs standalone.

## 5. Architecture & interfaces

### 5.1 Widened activation contract
Extend `ActivationState` (`feedflipnets/core/types.py`) with an **optional** `block_cache` carrying per-block `{A, softmax-Jac handle, LN scale (γ/σ), φ' mask, dead-zone mask, layer inputs}`. Additive and optional — existing MLP strategies (`Backprop`, `DFA`, `TernaryDFA`, `StructuredFeedback`) ignore it and keep working unchanged.

### 5.2 New strategies (`feedflipnets/core/strategies.py`)
- `ActivationRoutedDFA` — implements ①; consumes `block_cache`.
- `PerturbationTaughtFeedback` — implements ②; wraps/composes with any feedback strategy and mutates `state.feedback` from perturbation estimates.
Both follow the existing `FeedbackStrategy` Protocol (`init(model)→state`, `backward(activations, error, state)→(grads, state)`).

### 5.3 Testbed (`feedflipnets/core/transformer.py`, new)
- A small **pre-LN decoder-only Transformer block** and a **configurable-depth MLP**, both exposing `block_cache` and a pluggable backward.
- **Framework: torch-CPU.** Rationale: autograd provides exact backprop *for free* as (a) the tuned baseline and (b) the ground-truth alignment target `g_l`; hand-writing attention backward in NumPy across strategies is the debugging tar-pit flagged in the research plan. Our ①/② strategies remain **explicit** — they read cached forward tensors (`A`, `J_A`, LN scale, masks) but **do not call autograd** to compute their own gradients; autograd is used *only* for the exact-BP reference/target. (No contradiction: reusing a cached tensor ≠ differentiating through it.)
- **Determinism (the repo prizes it — make it real).** `torch.use_deterministic_algorithms(True)`; pin `torch.set_num_threads` / `OMP_NUM_THREADS`; seed torch + numpy + python RNGs **including the ② perturbation stream** as a separate named generator. The autograd exact-BP **reference/target runs in float64**; strategies may run float32. GATE-0 rel-err is measured against the float64 reference, with a **separate, looser tolerance for attention** (softmax/LN amplify float error) stated numerically in §6.

### 5.4 Instruments (`feedflipnets/eval/alignment.py`, new)
- **Finite-difference / autograd grad-check** — GATE-0 build gate (float64 reference, §5.3).
- **Alignment probe** — computes, on the *same* `(activations, error e)` pair evaluated at the same weights and batch (pre-update), a **per-weight-matrix** cosine `ρ = cos(Δ_strategy^W, g_BP^W)` for each of `{W_Q,W_K,W_V,W_O,W_1,W_2}`. Reported statistics: **attention-block `ρ` = min over `{W_Q,W_K,W_V,W_O}`** (MLP maps reported separately), the per-path breakdown `{W_V,W_O}` vs `{W_Q,W_K}` (§3), and the **depth-slope** `θ_min(L)` where `θ = arccos(clip(ρ,−1,1))` in degrees (see §7 for the slope fit). Never flatten different-shaped matrices into one vector.
- **Lock-free probe — two parts, both must pass** (enforces §2). *(a) Primary — dependency/taint check:* run `backward` under a torch tensor-taint tracer with `e` and the scalar `ΔL` whitelisted as cut sources; tag every downstream-layer **weight and activation** tensor and assert none is an ancestor of any `grad_l` except through the whitelisted `e`/`ΔL`. Perturb the *actual* downstream parameter objects the strategy could reference (same tensor id / storage the forward uses, not a detached copy) and assert no surrogate shares storage or graph-ancestry with a downstream weight. *(b) Secondary — `e`-fixed perturbation smoke test:* (1) forward once, cache all activations and `e`; (2) `backward(activations, e, state) → grad_l`; (3) perturb **only** downstream weight/activation tensors visible to `backward` **without** recomputing `e` or layer-`l`'s own cache; (4) assert `grad_l` is unchanged. *Frozen set:* `e`, `ΔL`, and activations of layer `l` and earlier. *Varied set:* downstream weights and downstream activation caches. A lock-free strategy is invariant; backprop's `δ_l = (W_{l+1}ᵀ δ_{l+1}) ⊙ f'(z_l)` changes (probe correctly rejects BP). During ② training, additionally assert `dB ∝ (…)eᵀ` never explicitly reads `W_{l+1}` (an *emergent* correlation `B → W_{l+1}ᵀ` is allowed; explicit reads are not).

## 6. Staged milestones & pre-registered gates

### GATE-0 (build, blocks all science)
Every strategy's grads match the **float64 autograd reference** (§5.3): rel-err **< 1e-4** for MLP maps, **< 1e-3** for attention maps (softmax/LN amplify float error). The **two-part lock-free probe (§5.4) passes** on every non-BP strategy and **correctly rejects BP**; 200-step smoke run finite. No alignment/accuracy number is trusted until green.

### M1 — deep MLP (L ∈ {2,4,8,16}) + ②
Cheapest falsification of the transport-free-feedback-learning claim; builds all reusable plumbing.
- **Negative control:** fixed-random DFA's `min_l ρ_l` decays with depth (reproduce the `c/√L`-style collapse), `θ_min(L)` slope significantly negative — pre-registered.
- **Sample-budget precondition (guards against a false kill).** Before judging ②, derive `N` (samples/block to reach the target averaged-cosine at `d_pert`, per §4's variance model) and **spend ≥ `N`** (via EMA accumulation / `K`-per-step). The kill criterion is only valid once the budget is met; an under-provisioned run is a *no-decision*, not a kill.
- **Success:** with ≥`N` samples spent, ②'s adapted feedback bends the slope — `θ_min(L)` depth-slope **≥ −2°/layer** (flat within the pre-registered noise band = ± seed-std of the slope), and at L=16 beats fixed-DFA by an **absolute effect ≥ 5° in `θ` with non-overlapping 95% CIs** (not merely "> 1 seed-std"); accuracy gap to tuned BP shrinks vs fixed-DFA. **n ≥ 5 seeds**.
- **Kill:** with the budget met, ② still cannot bend the slope or clear the effect-size floor at L=16 → ② is dead as specified; revisit the estimator (control variate, hypernetwork `B`) or drop ②.

### M2 — Transformer block: ① (level), then ①+② (slope) — with an ablation table
The headline: alignment survives **through attention**. **Two orthogonal claims on two axes — do not cross-wire them (§3):**
- **Baseline guard (numeric, anti "20NG BP = 19.86%" trap):** tuned BP on the char-level TinyStories decoder must reach **≤ 1.8 bpc** (clearly beating bigram ~3.3 / unigram ~4.5).
- **Claim ① (LEVEL, fixed depth):** at the deepest block, ①'s **attention-block** `ρ` **exceeds fixed-random-DFA's by ≥ 5° in `θ` with non-overlapping 95% CIs** (slope *not* asserted for ① alone). Report the per-path breakdown; ①'s win may be value-path (`W_V,W_O`) only.
- **Claim ②/①+② (SLOPE, across depth):** the **depth-slope** non-decay (`θ_min` slope ≥ −2°/block, attention-block min-`ρ` ≤ 60°) is attributed to `B_block` adaptation, i.e. **② or ①+②**, never ① alone.
- **Isolated ②-on-transformer sub-gate:** ①+② attention-block min-`ρ` must exceed ①-alone by a registered margin (non-overlapping CIs), else **kill ② on transformers** independently of the bpc number (② is otherwise only ever falsified on MLPs).
- **Accuracy gate:** ①+② val-bits-per-char gap to tuned BP **≤ 0.15 bpc, with the 95% CI (n ≥ 5) excluding a gap > 0.25**.
- **Ablation table (mandatory):** `fixed-DFA / ①-only / ②-only / ①+②` × {attention-block min-`ρ`, `θ` depth-slope, per-path `ρ`, val bpc} — so the level win is credited to ① and the slope win to ②, unambiguously.
- **Kill:** ①'s attention-block `ρ` merely *ties* fixed-DFA at fixed depth (reusing `A` bought nothing → ① central hypothesis fails); OR neither ② nor ①+② bends the depth-slope; OR the lock-free probe (§5.4) fails on the accurate configuration (invariant §2 violated).

## 7. Metrics (definitions)
- **Alignment `ρ^W`** = `cos(Δ_strategy^W, g_BP^W)` **per weight matrix** `W` (not a flattened concat of different-shaped matrices), with `g_BP` from the float64 autograd reference, evaluated at the *same* weights and batch **before** the update step. Attention-block `ρ` = **min over `{W_Q,W_K,W_V,W_O}`**; MLP maps reported separately; per-path breakdown `{W_V,W_O}` vs `{W_Q,W_K}`.
- **Depth-slope** — define `θ = arccos(clip(ρ,−1,1))` in degrees; the slope is an **OLS fit of `θ_min(L)` vs `L`** (state the axis; `log2 L` is the alternative — we use `L`) over the depth sweep, reported with 95% CI. Gates are on **min-`ρ`** (worst layer). A "non-decaying" slope must have its 95% CI exclude the pre-registered decay noise band (± seed-std of the fixed-DFA slope). Units: **per-layer** for M1 (MLP), **per-block** for M2 (Transformer) — registered separately, not interchangeable.
- **Accuracy gap** = tuned-BP metric − strategy metric (val bits-per-char for M2, accuracy for M1), mean ± 95% t-CI, n ≥ 5.
- **Parallelism/lock-free** = the two-part probe of §5.4 (dependency-taint + `e`-fixed perturbation) passes; `e` and scalar `ΔL` are the only whitelisted cross-layer channels.

## 8. Scope / YAGNI
- **In:** deep MLP + single/few-head decoder block, CPU, alignment + accuracy on small data.
- **Out (this spec):** GPU, pipeline/systems wall-clock, multi-head at scale, large vocab, ternary-LLM-beyond-STE. Ternary forward is an **optional flag** reusing the existing quantizer — not required for the alignment result.
- No refactor of the existing NumPy MLP strategies beyond the additive `block_cache` field.

## 9. Risks & mitigations
- **Perturbation variance (the dominant ② risk, `Var ∝ d_pert`):** the real controls are **sample averaging** — EMA of `ĝ` per block + `K` perturbations/step to reach the §4 budget `N` — plus a **control variate** for `ΔL`. Antithetic `±ξ` is a **bias** control (cancels `O(ρ²)` curvature), *not* a variance control; do not rely on it for feasibility. M1's kill is gated on spending ≥ `N` samples.
- **② can only learn a fixed linear `e→g` map** (`B* = E[g eᵀ]E[e eᵀ]⁻¹`): it cannot track attention's moving target — that is ①'s job. Scope ② to the input-independent rotation + depth-slope; if the residual demands input-conditioning, escalate to a hypernetwork `B` (larger change).
- **Alignment not predictive of loss:** always pair `ρ` curves with an actual accuracy/bpc gap; treat `ρ` as necessary-not-sufficient.
- **①'s score-path (`W_Q,W_K`) may collapse to plain-DFA quality** (it sits behind random `W_V/W_O` surrogates): the per-path probe makes this observable; pre-register that ①'s win may be value-path-only until ② adapts `W_V/W_O`.
- **torch-in-core departure:** CPU-only, determinism knobs per §5.3; keep torch confined to the testbed + autograd reference, strategies stay explicit.

## 10. File plan
- `feedflipnets/core/types.py` — add optional `block_cache` to `ActivationState`.
- `feedflipnets/core/strategies.py` — add `ActivationRoutedDFA`, `PerturbationTaughtFeedback`.
- `feedflipnets/core/transformer.py` (new) — torch-CPU pre-LN decoder block + deep MLP, `block_cache` + autograd exact-BP reference.
- `feedflipnets/eval/alignment.py` (new) — grad-check, alignment probe, lock-free probe.
- `datasets/tinystories.py` — upgrade to char-level contiguous loader + frozen train/val split (M2 only).
- `tests/` — grad-check, lock-free invariant, alignment negative-control (fixed-DFA must decay).
- `experiments/` — M1 depth sweep, M2 attention-alignment runners (deterministic seeds).

## 11. Resolved decisions
- Core = ① with ② bolt-on; both preserve lock-free backward (invariant §2). ✔
- Testbed framework = torch-CPU (autograd for reference/target only). ✔
- Staging = M1 (MLP + ②) → M2 (Transformer + ①). ✔
