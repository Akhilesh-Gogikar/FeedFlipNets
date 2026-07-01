# M1 result — Perturbation-Taught Feedback vs fixed DFA

Source of numbers: `m1_depth_sweep.json` (produced by `python -m experiments.m1_depth_sweep`).
Config: width 32, D_in 16, C 4, N 128, depths {2, 4, 8, 16}, 5 seeds, 300 steps, lr 0.1,
`perturb` uses samples_per_step 8 / lr_B 0.2. Alignment = min-over-layers θ (worst layer)
of `cos(Δ_strategy, Δ_backprop)`, averaged over seeds.

## Measured mean worst-layer θ (deg) vs depth

| depth L | dfa    | perturb |
| ------- | ------ | ------- |
| 2       | 55.57  | 56.72   |
| 4       | 74.08  | 65.95   |
| 8       | 90.00  | 90.00   |
| 16      | null   | null    |

`theta_slope_deg_per_layer`: dfa = null, perturb = null (a finite OLS slope requires a
finite θ at every depth; L=16 is non-finite — see below).

## Pre-registered gate (spec §6/M1)

- **Negative control:** fixed-DFA `theta_slope_deg_per_layer` > 0 (alignment decays with
  depth). **PASS.** Measured on the pre-registered control config (fixed random DFA at
  `steps=0`, depths {2, 4, 8}, 5 seeds): worst-layer θ = 93.00 → 97.14 → 101.39,
  slope **+1.35°/layer** (95% CI ±0.49, excludes 0). The `c/√L` decay reproduces robustly.
  `tests/test_m1_smoke.py::test_fixed_dfa_negative_control` passes.

- **Success (perturb slope ≥ −2°/layer AND perturb beats dfa by ≥5° at L=16 with
  non-overlapping 95% CIs):** **NOT MET — uncomputable on this testbed.**
  Under the pre-registered lr=0.1, the deep bias-free ReLU MLP **diverges during training
  at depth**: at L=16 the weights explode (|W| ~ 1e145 by ~step 67), the forward pass
  overflows to inf, and softmax-CE yields NaN, so the L=16 alignment is non-finite (`null`)
  for **both** strategies. The slope, the L=16 comparison, and their CIs therefore cannot
  be evaluated. Even at L=8 both strategies collapse to θ = 90.0° (fully decorrelated),
  and |W| already reaches ~1e24 — the trained-network alignment signal is destroyed by
  instability before the strategies can be distinguished. At the only depths where both
  stay finite (L=2, 4) `perturb` is modestly better than `dfa` (56.72 vs 55.57 at L=2 —
  effectively tied; 65.95 vs 74.08 at L=4 — perturb +8.1° better), which is suggestive but
  well short of the pre-registered L=16 criterion.

- **Budget precondition (samples_per_step × steps ≥ N(width)):** width = 32, spend =
  8 × 300 = 2400 ≥ 32. **MET.** (Budget is not the limiter here; training stability is.)

## Decision: revise the testbed, then re-gate (NO-GO to M2 as-is)

The pre-registered success gate is **NOT MET**, but the cause is a **testbed defect, not a
falsification of ②**: the fixed-random-DFA depth-decay hypothesis (the negative control) is
confirmed, yet the *trained-weight* alignment measurement is confounded by training
divergence at depth ≥ 8 under lr=0.1 on a bias-free, un-normalized ReLU MLP. We cannot
conclude ② fails, nor that it succeeds — the L=16 measurement does not exist.

Recommended revision before re-running the gate (do NOT proceed to M2 until it passes):
- Stabilize depth training so L=16 stays finite: lower/anneal lr (empirically lr ≈ 0.01
  keeps L=16 finite here), and/or add per-layer scale control (LayerNorm-style
  normalization or gradient/weight-norm clipping). Keep the alignment metric and the
  negative control unchanged.
- Re-run `python -m experiments.m1_depth_sweep` and re-evaluate the success gate against a
  finite L=16 for both strategies.

Honest go/no-go: **NO-GO / DEFER** — negative control PASS, success gate uncomputable due
to depth-training divergence. This is a real, recorded finding, not a pass.
