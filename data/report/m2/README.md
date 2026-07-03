# M2 result — Activation-Routed DFA through attention (alignment ablation)

Produced by `experiments/m2_attention_alignment.py` (torch 2.8 CPU, float64, d=32, d_ff=64,
T=16, n=5 seeds, budget 2000 steps x K=16). Numbers below are the ACTUAL run recorded in
`m2_ablation.json`, not the prototype estimates.

Pre-registered gate (spec §6/M2, LEVEL claim; bpc accuracy DEFERRED to M2b):

- **Value-path exact (①):** dW_O, dW_2 vs autograd < 1e-5. **PASS** — measured
  `value_path_exact_err = 7.1e-15` (one_only, all seeds).
- **A-reuse ceiling (①):** exact-surrogate (`R_O = W_Oᵀ`) value-path theta = 0.00 deg. **PASS**
  (asserted in `tests/test_activation_routed_dfa.py::test_a_reuse_ceiling_exact_surrogate_gives_zero_angle`,
  cos > 0.999).
- **LEVEL win (①+②, HONEST transport-free ②):** attention-block theta beats fixed-DFA by >= 5 deg,
  non-overlapping 95% CIs (n=5). **PASS** — one_two `46.35 ± 5.60 deg` vs fixed_dfa `90.56 ± 0.42 deg`,
  **Δ = 44.2 deg**, non-overlapping CIs (fixed_dfa lower 90.13, one_two upper 51.95).
  NOTE: ② is PARTIAL — `R_O_rel_err = 1.057` (NOT exact; ‖R_O−W_Oᵀ‖/‖W_Oᵀ‖ ≈ 1.06, the M1 1/√d
  variance wall). The win comes from ①'s A-reuse + ②'s partial directional recovery, NOT from R_O
  reaching W_Oᵀ. Per-path: value {Wv,Wo} = `42.86 deg`, score {Wq,Wk} = `44.17 deg` — both clear >= 5 deg.
- **②-only (A-free adapted Bv, no ①):** isolating control — `91.82 ± 1.38 deg`, essentially the ~90 deg
  noise floor (barely off fixed_dfa's 90.56 deg). Confirms the win is credited to ①'s A-reuse, not ②
  alone. (The smoke test drives this below fixed_dfa at K=32; the aggregate table uses K=16.)
- **①-alone (fixed random R_O):** `90.78 ± 0.87 deg` — marginal (~0.2 deg over fixed_dfa), does NOT
  clear >= 5 deg. A random surrogate in series with the exact A collapses ①-alone to the noise floor.
- **Per-path:** {value: Wv,Wo} vs {score: Wq,Wk} theta reported per condition in `m2_ablation.json`.
- **Lock-free:** structural check passes ① / flags the `bp_block_grads` positive control; e-fixed
  transpose-perturb change = 0 for ①, > 0 for the positive control. **PASS**
  (`tests/test_attention_lockfree.py`).
- **SLOPE non-decay** is a ②/①+② claim — asserted only for adapted conditions (deferred to the M2b
  depth sweep with bpc).

## Recorded numbers (from `m2_ablation.json`)

| condition | attn_block_theta (deg) | value_theta | score_theta | R_O_rel_err |
|-----------|------------------------|-------------|-------------|-------------|
| fixed_dfa | 90.56 ± 0.42           | 90.44       | 90.26       | —           |
| one_only  | 90.78 ± 0.87           | 90.35       | 90.11       | —           |
| two_only  | 91.82 ± 1.38           | 89.03       | 90.09       | —           |
| one_two   | 46.35 ± 5.60           | 42.86       | 44.17       | 1.057       |

## Decision: GO to M2b (bpc)

The LEVEL gate is MET with a wide margin (Δ = 44.2 deg, non-overlapping 95% CIs). The honest read:
① is value-exact (`7.1e-15`) and its A-reuse is real; ②'s transport-free adaptation only PARTIALLY
recovers the surrogate (`R_O_rel_err = 1.06`, the 1/√d variance wall) but is enough — WITH ① — to
clear the level gate by a wide margin. Exactness of R_O is not required and not achieved. Proceed to
the M2b plan (TinyStories char-level bits-per-char accuracy gate + depth-slope non-decay).
