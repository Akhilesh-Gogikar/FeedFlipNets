# M1 result — Perturbation-Taught Feedback (②) vs fixed DFA

Source: `m1_depth_sweep.json` (`python -m experiments.m1_depth_sweep`).
Config: width 32, D_in 16, C 4, N 128, depths {2, 4, 8, 16}, 5 seeds, 300 steps, lr 0.1,
**global grad-norm clip at 1.0** (substrate amendment — see below), `perturb` uses
samples_per_step 8 / lr_B 0.2. Alignment = min-over-layers θ (worst layer) of
`cos(Δ_strategy, Δ_backprop)`, averaged over seeds. Lower θ = better aligned.

## Substrate amendment (pre-registered, 2026-07-01)

The initial run at lr=0.1 with **no clipping diverged** at depth (bias-free deep MLP: |W| ~ 1e145
by ~step 67 at L=16 → NaN), making the gate uncomputable. Fix: a **global grad-norm clip at 1.0**
— the value the FeedFlipNets paper itself recommends — applied **identically to every strategy**.
It stabilizes the weight trajectory (L=16 now finite; L=8 un-saturates from 90.0° → 63–78°).
The alignment **metric and gate thresholds are unchanged**; clipping touches only the update
trajectory, never the raw gradients alignment is measured on. lr kept at 0.1 (lowering it
*worsened* worst-layer alignment in probes, so clipping alone is the minimal fix).

## Measured mean worst-layer θ (deg) vs depth

| depth L | dfa    | perturb | ② advantage |
| ------- | ------ | ------- | ----------- |
| 2       | 55.57  | 56.72   | −1.15       |
| 4       | 73.87  | 65.72   | **+8.15**   |
| 8       | 78.12  | 63.43   | **+14.69**  |
| 16      | 105.84 | 106.11  | −0.27       |

`theta_slope_deg_per_layer`: dfa = **+3.26** (CI ±1.12), perturb = **+3.41** (CI ±1.69).

## Pre-registered gate (spec §6/M1)

- **Negative control — PASS.** Fixed-DFA worst-layer θ increases with depth: slope **+3.26°/layer**
  (trained; 95% CI ±1.12 excludes 0), corroborated by the init-time control (+1.35°/layer). The
  `c/√L` alignment decay reproduces robustly. `test_fixed_dfa_negative_control` passes.

- **Success (② slope ≥ −2°/layer [flat, non-decaying] AND ② beats dfa by ≥5° at L=16 with
  non-overlapping 95% CIs) — NOT MET.**
  - ②'s slope is **+3.41°/layer** — it decays with depth *at the same rate* as fixed DFA, not flat.
    ② does **not bend the depth-slope**.
  - At L=16 the two are **tied** (−0.27°, overlapping CIs); ② does not help the deepest case.

- **Budget precondition:** 8 × 300 = 2400 ≥ N(width=32). **MET.**

## Honest interpretation

② produces a **real, growing mid-depth improvement** in worst-layer alignment (**+8.1° at L=4,
+14.7° at L=8**) — transport-free feedback learning *does* lift the alignment *level* where training
is stable. But it **fails the pre-registered gate**: it does not change the depth-*slope*, and it
collapses back to fixed-DFA at L=16 (both go anti-aligned, θ ≈ 106°).

This is exactly the ceiling the spec (§4) predicted: node-perturbation adaptation converges to a
**fixed linear map** `B* = E[g eᵀ]·E[e eᵀ]⁻¹`, which can lower the level but **cannot bend the
composition-driven `c/√L` decay** and cannot track a worst layer that has gone anti-aligned. Bending
the slope / handling the moving target is the job of **① Activation-Routed DFA** (M2), not ②.

## Decision: **NO-GO on the M1 gate — but a genuine partial positive for ②.**

② is **not falsified** (clear mid-depth win, theory-consistent) but **does not clear the depth-slope
bar on its own**. Options: (a) proceed to **M2 (① Activation-Routed DFA)**, where the mechanism that
targets the slope/moving-target lives, and test ①+② together; (b) first try to strengthen ② beyond a
fixed linear map (input-conditioned `B` hypernetwork — a larger change flagged in spec §4). Recommend
(a): the M1 signal says ② helps but is not the slope-fixer; ① is. Recorded honestly; not a pass.
