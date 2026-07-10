# PRE-REGISTRATION — FeedFlip Round 3: magnitude-carrying votes

Committed BEFORE any full arm run. This file and
`experiments/feedflip_magnitude_votes.py` land in the same commit; full runs
start only after that commit exists (verifiable from git history).

## Motivation (from the round-2 result, commit 2588463)

Round 2 overturned the sign-barrier framing: bit-exact BP vote signs
(sign_match_p = 1.0) reach only .551 vs bp_shadow .604, and acc(k) is flat out
to k=4096. The binding constraint is the MECHANISM — the K-threshold vote
counter discards gradient magnitude |g|, which is exactly what the float
shadow retains. Round 3 puts magnitude back into the accumulator while keeping
training state <= 8 bits/weight and feedback communication <= 0.01 bits/w/step.

## Frozen benchmark (unchanged from rounds 1-2)

Task seed 7 teacher MLP; DIMS [32, 64, 64, 64, 64, 10]; 8000 steps; batch 16;
n_train 16384; test 2048; seeds {0, 1, 2}; net init seed+100; batch stream
seed+900 (byte-identical to rounds 1-2: mechanism noise uses a SEPARATE stream
seed+1300 so batch sequences stay comparable). Vote threshold K=32 where a
vote accumulator exists. Per-layer alpha fixed at init for ALL arms.

## Mechanisms (hyperparameters frozen here)

Per layer, per step: g = h^T d / batch; s_l = EMA(0.99) of mean|g| per layer
(initialised to the first batch's mean|g|; one float per layer, not per-weight
state). s_safe = max(s_l, 1e-12).

- **magq2** — 2-bit magnitude-weighted votes:
  `c += -sign(g) * clip(rint(|g|/s_safe), 0, 3)`; flip at |c| >= 32, reset to 0.
  State: 1.58 + log2(2*(31+3)+1) = 7.69 bits/w.
- **stochT** (T in {1, 4}) — unbiased stochastic vote rate:
  `c += -sign(g) * Bernoulli(min(1, |g|/(T*s_safe)))`; flip at |c| >= 32.
  E[increment] = -g/(T*s): the accumulator integrates the true (scaled)
  gradient in expectation. State: 1.58 + log2(65) = 7.60 bits/w.
- **ishadE** (E in {2, 8}) — int8 stochastic-rounded shadow (SR-SGD analog):
  `a += SR(-E * g/s_safe)`, a clipped to [-127, 127] int8;
  `W = ternarize(a)` with threshold 0.7*mean|a| (recomputed per step, matching
  the TernaryNet/bp_shadow ternarizer); a_0 = 8 * W_0 (so ternarize(a_0) = W_0).
  SR(x) = floor(x) + Bernoulli(frac(x)). No vote accumulator.
  State: 8.0 bits/w (W is derived from a, not stored separately).

## Feedback (second-order per round 2; two levels per mechanism)

- **st512** — stale ternary eff^T transport refreshed every 512 steps
  (round-2 D1 form): fb_comm = log2(3)/512 = 0.0031 bits/w/step.
- **farand** — fixed random chained FA, seed-regenerable: 0 bits.

Arms = {magq2, stoch1, stoch4, ishad2, ishad8} x {st512, farand} = 10 arms,
3 seeds each. Also logged per row: sign_match_p vs BP votes (cumulative and
final-10%), ms/step, state bits, fb_comm, and gap closure
GC = (acc - .557) / (.604 - .557) using the round-2 sign-only reference
(sign_transport_k512 = .557) and the round-1 bp_shadow anchor (.604).

## Gates (GO/NO-GO, frozen)

- **G-R3 (primary)**: any arm reaches mean acc >= 0.60 at <= 8 state bits/w
  and <= 0.01 fb bits/w/step (all 10 arms satisfy the budgets by construction,
  so the gate reduces to the accuracy bar). GO = magnitude-carrying votes
  close >= ~91% of the sign-only -> bp_shadow gap.
- **G-R3-strict (secondary, reported not gating)**: any arm mean acc >= 0.604
  (matches the bp_shadow point estimate — "backprop matched at 1/4 the state").

## Pre-registered predictions

- P1: ordering by magnitude information: ishad8 >= stoch1 >= magq2 >= 0.557
  (the sign-only reference), per feedback level.
- P2: stoch1_st512 beats 0.557 by > 2 sigma (unbiased magnitude integration
  is the mechanism round 2 said was missing).
- P3: feedback stays second-order: |acc(st512) - acc(farand)| <= 0.02 within
  each mechanism (round-2 flatness transfers to magnitude votes).
- P4 (interpretation rule, frozen now): if BOTH ishad arms < 0.60, then <= 8
  bits of per-weight magnitude state is insufficient on this benchmark; the
  outcome is a state-bits frontier (0-bit sign votes .55 -> 8-bit SR shadow
  -> 32-bit float .604), not a GO. If any ishad arm >= 0.60 but no vote arm
  (magq2/stoch) does, the honest headline is "low-bit SR-SGD works", NOT
  "FeedFlip voting works" — the vote/flip mechanism itself remains capped.

## Procedure

- Smoke runs (<= 300 steps, seed 0 only, one arm per mechanism family) for
  crash/shape checks only; accuracies discarded, no tuning.
- Hyperparameters above are frozen. Any post-hoc sweep is EXPLORATORY and
  cannot flip a gate. The {T} and {E} grids are part of the registered arms;
  with sigma ~= .01 and a .05 bar-to-gate margin, grid multiplicity (10 arms)
  cannot manufacture a false GO.
- Analysis: mean/std over seeds {0,1,2}; no exclusions; all rows written to
  `data/report/feedflip/feedflip_round3.json`.

## Deviations

None post-freeze. The harness was not modified after the freeze commit
(`06b13da`); the ishad collapse diagnostic below is exploratory analysis only
(no re-runs of registered arms, no gate impact). Pre-freeze, one smoke-exposed
crash fix (None-safe summary aggregation when a row has no valid sign-metric
pairs) landed inside the freeze commit itself.

---

## RESULTS (appended after the frozen run; `feedflip_round3.json`, no exclusions)

| arm | mean acc | std | GC | late sign-match p | bits/w | fb bits/w/step |
| --- | -------- | --- | -- | ----------------- | ------ | -------------- |
| stoch1_st512 | **0.574** | .009 | 0.37 | **0.975** | 7.60 | 0.0031 |
| stoch4_st512 | 0.569 | .008 | 0.27 | 0.969 | 7.60 | 0.0031 |
| stoch4_farand | 0.568 | .009 | 0.22 | 0.774 | 7.60 | 0 |
| magq2_st512 | 0.560 | .029 | 0.07 | 0.918 | 7.69 | 0.0031 |
| magq2_farand | 0.555 | .019 | −0.05 | 0.776 | 7.69 | 0 |
| stoch1_farand | 0.544 | .006 | −0.27 | 0.728 | 7.60 | 0 |
| ishad{2,8}_{st512,farand} | 0.330 | .000 | −4.83 | — (g ≡ 0) | 8.00 | — |

### Gates: **NO-GO on all three**

- **G-R3 FAIL** — best arm stoch1_st512 0.574 ± .009 < 0.60.
- **G-R3-strict FAIL** — no arm ≥ 0.604.
- **P4 vote-mechanism GO FAIL** — no vote arm ≥ 0.60.

### Predictions scored

- **P1 REFUTED** — ordering inverted at the top: ishad arms finish dead last
  (0.330, see collapse note), stoch1 ≥ stoch4 ≥ magq2 among survivors.
- **P2 MARGINAL MISS** — stoch1_st512 beats 0.557 by +0.017 = 1.9σ
  (seed std .009); the frozen bar was > 2σ, so not confirmed.
- **P3 VIOLATED for stoch1** — st512 − farand = +0.030 > 0.02 (late p 0.975
  vs 0.728). Holds for magq2 (+0.006) and stoch4 (+0.002). Feedback quality
  re-emerges as first-order exactly for the unbiased high-rate vote mechanism.
- **P4 rule FIRES** (both ishad < 0.60) → the registered interpretation is a
  **state-bits frontier**: sign-only votes 0.557 (≈7.6 bits) → magnitude votes
  0.574 (7.6 bits) → float32 shadow 0.604 (32 bits). CAVEAT: the ishad arms do
  not support this reading independently — they collapsed for a dynamics
  reason (below) and are uninformative about SR-SGD at gentler step sizes; the
  frontier reading rests on the vote arms.

### Exploratory diagnostic: ishad collapse mode (post-hoc, non-gating)

Not int8 range saturation (sat_frac ≤ 0.19 at all layers). The E-scaled SR
steps (per-step shadow increment ~E vs initial shadow scale 8) churn weights
through the 0.7·mean|a| ternarizer; deeper layers lose nonzero weights
(0.56 → 0.33 by step ~60), active paths die, and the network enters a
zero-gradient absorbing state: mean|g| = 0 exactly ⇒ SR(0) = 0 ⇒ frozen
forever at majority-class accuracy 0.330 — identically across all seeds and
both feedback types (death precedes any feedback influence).
