# PRE-REGISTRATION — FeedFlip round 4: gentle-step int8 SR shadow ("ishad" revisited)

Status: FROZEN at commit time of this file. Written and committed BEFORE any
full run. Only the calibration probes disclosed below were executed first
(smoke/crash checks + dead-oracle diagnostic; 300 steps; never gate-eligible).

## Motivation

Round 3 (freeze `06b13da`, results `81a6e35`) killed its ishadE arms
(E in {2, 8}) via a zero-gradient absorbing state — diagnosed as a step-scale
dynamics failure, not evidence about 8-bit SR-SGD capacity. Round 4 re-probes
the identical int8 stochastic-rounding shadow mechanism at gentler step sizes
and adds a bit-exact-BP oracle arm per step size as a ceiling control, to
separate accumulation-precision capacity from feedback quality for the
residual sign-only -> bp-shadow gap (~0.574 -> ~0.604, per round-3 README;
constants imported as REF_SIGN_ONLY / REF_BP_SHADOW from
`experiments/feedflip_magnitude_votes.py`).

## Harness

`experiments/feedflip_ishad_gentle.py` (committed in this freeze).

- Benchmark identical to rounds 1–3 (`make_task`, `TernaryNet`, STEPS, BATCH,
  SEEDS all imported unchanged from `experiments/feedflip_bitflip.py`).
- Batch stream: `default_rng(seed + 900)` — identical to rounds 1–3.
- SR noise stream: `default_rng(seed + 1300)` — round-3 offset, unchanged.
- Shadow: int8, clip ±ISHAD_CLIP, SR update `x = -E * g / max(EMA_mag, 1e-12)`,
  ternarize via round-3 `ternarize_from_shadow` (unchanged).
- State cost: 8.0 bits/weight (`state_bits_per_w`).

## Arms (9 = 3 step sizes x 3 feedback levels)

- E in {0.25, 0.5, 1.0} (`e025`, `e05`, `e10`) — vs round-3's fatal {2, 8}.
- Feedback: `st512` (sign-transposed ternary B, refresh every ST_K steps),
  `farand` (fixed random FA), `bp` (exact backprop deltas — ORACLE CEILING
  CONTROL, never gate-eligible).
- Seeds: SEEDS (unchanged). All rows written; no exclusions.

## Metrics (per arm-seed, all written to JSON)

acc, gap_closure, sign_match_p / sign_match_p_late (last 10% of steps),
dead_step (first step with all-layer zero gradient), churn_per_w_per_kstep,
final_sat_frac (shadow at clip), final_nonzero_w, ms_per_step,
fb_comm_bits_per_w_step.

## Frozen decision rules

- **G_R4 (go/no-go):** GO iff any gate-eligible arm (st512/farand only) has
  mean acc >= GATE_ACC (0.60) across seeds. Strict variant `G_R4_strict`:
  mean acc >= REF_BP_SHADOW.
- **P4 (precision ceiling):** Let C = max mean acc over VALID bp oracle arms.
  - C < 0.59  -> "precision binding, frontier confirmed"
  - C >= 0.604 -> "precision not binding, frontier overturned"
  - 0.59 <= C < 0.604 -> "boundary unresolved"
  - **Validity guard:** a bp arm is VALID only if `n_dead == 0` across all
    seeds. If NO bp arm is valid, P4 returns
    `oracle_invalid_dynamics_failure_no_precision_conclusion` — a dead oracle
    is a dynamics failure and MUST NOT be read as "precision binding".

## Calibration probes disclosed (300 steps; NOT gate evidence)

Smoke (seed 0): `ishad_e05_bp` acc 0.330 DEAD@124; `ishad_e05_st512` 0.472;
`ishad_e025_farand` 0.449. Dead-oracle diagnostic (seeds 0,1): e025_bp alive
both seeds (acc 0.363/0.331); e05_bp dead 1/2 (dead@124); e10_bp dead 2/2
(dead@41/33). Gate-eligible arms never died. Interpretation risk this guard
addresses: BP gradients flow through downstream ternary weights and vanish
when layers ternarize toward zero (absorbing state, churn drops); B-matrix
feedback bypasses downstream weights and keeps paths alive. The validity
guard was added to the harness BEFORE this freeze in direct response.

## Procedure

1. Smoke test (done, disclosed above).
2. Commit this doc + harness (freeze commit) BEFORE any full run.
3. Full suite: 9 arms x SEEDS x STEPS in background;
   output `data/report/feedflip/feedflip_round4.json`.
4. Results written to round-4 results md; gates evaluated exactly as frozen
   here; any post-hoc analysis labelled EXPLORATORY.
