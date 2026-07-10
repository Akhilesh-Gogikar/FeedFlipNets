# PRE-REGISTRATION — FeedFlip Round 2: the feedback-information budget

Committed BEFORE any full arm run. This file and
`experiments/feedflip_feedback_budget.py` land in the same commit; full runs
start only after that commit exists (verifiable from git history).

## Motivation (from the round-1 NO-GO, commit 98a717e)

Round 1 froze the deep-MLP FeedFlip benchmark and showed: pure transport-free
voting (random/orthogonal/taught B, confidence gating, K-annealing) caps at
mean acc ~.54 vs bp_shadow .604, with per-weight vote sign-match p ~= 0.70-0.71
vs the BP vote. Interpretation: with ZERO information flowing from W into the
feedback path, ~30% of votes have the wrong sign and the flip mechanism cannot
close the gap.

Round 2 attacks the barrier on a new axis: HOW MUCH information (bits) must
flow from the forward weights into the feedback path — and by WHAT route
(local learning vs explicit communication) — to recover BP-level accuracy?

## Frozen benchmark (unchanged from round 1)

Task seed 7 teacher MLP; DIMS [32, 64, 64, 64, 64, 10]; 8000 steps; batch 16;
n_train 16384; test 2048; seeds {0, 1, 2}; vote threshold frozen at K=32 (best
round-1 value). Same RNG seed offsets as round 1 (net seed+100, batch stream
seed+900), so runs are batch-for-batch comparable with round-1 arms.

Forward-path training state: ternary W + vote accumulator
= 1.58 + log2(65) = 7.60 bits/w (<= 8, unchanged for every round-2 arm).

## Accounting (new axes, defined before running)

- `fwd_state_bits/w` — ternary W + vote accumulator (round-1 definition): 7.60.
- `fb_state_bits/w` — storage on the feedback path. Matrices procedurally
  regenerable from a shared seed count 0 (the convention round 1 implicitly
  used for random B).
- `fb_comm_bits/w/step` — information transported from W into the feedback
  path during training. One ternary refresh = log2(3) = 1.585 bits/w.

## Arms

| id | arm | status | fb_comm | fb_state |
|----|-----|--------|---------|----------|
| C0 | `fa_random` | wired | 0 | 0 (seed) |
| A1 | `kp_fa` | wired | 0 | 32 (float B) |
| A3 | `sign_transport_k{1,8,64,512,4096}` | wired | 1.585/k | 1.58 |
| A2 | `prealigned_init` | registered, not wired | 0 | 0 (seed) |
| A4 | `fwd_cv` | registered, not wired | 0 | 0 (seed) |
| A5 | `wallclock_stale` | registered, not wired | n/a | n/a |

**C0 `fa_random`** — layerwise-chained feedback (FA): deltas chain backwards
through fixed random B_j (same shapes as W_j^T) instead of DFA's direct
error->layer maps. Control arm: separates chaining *topology* from alignment,
so A1/A3 gains can be attributed.

**A1 `kp_fa` (Kolen-Pollack mirror)** — float B_j updated with the SAME
locally available signals used for layer j's own weight update; W is never
read: `B_j <- (1 - 1e-3) * B_j - 0.05 * (h_j^T d_j)^T / batch`.
Frozen hyperparameters: lr_B = 0.05, decay = 1e-3, B init random Gaussian
(fan-scaled), update applied after the vote/flip step each iteration.
Registered caveat: W moves by discrete flips, so the KP convergence theorem
(B - W^T -> 0 under shared updates + shared decay) does NOT apply; whether
alignment can still be *learned* through the flip dynamics is the question.

**A3 `sign_transport_k`** — B_j = stale copy of eff(j)^T = (alpha_j W_j)^T,
refreshed every k steps (at steps ≡ 0 mod k, before the forward pass);
k in {1, 8, 64, 512, 4096}. B_0 = eff_0^T via shared init seed (0 marginal
bits). Only the ternary pattern is ever communicated: alpha_j is fixed at init
and derivable from the shared init seed at both ends (vote signs are invariant
to per-layer positive scaling in exact arithmetic; eff^T rather than bare W^T
is used for bit-exactness — see Deviations D1). fb_comm = 1.585/k bits/w/step.
k=1 is the family's envelope: it produces exactly the BP vote signs
(structurally, sign_match_p must equal 1.0 — used as a code sanity check),
and doubles as the missing `feedflip_bp_k32` baseline from round 1.
Equivalence note (registered): k=1 vote behaviour is ALSO reachable at
fb_comm = 0 by KP update-sharing (flip events are computed locally and can be
applied to B^T directly); the k-axis therefore measures the price of NOT
sharing updates, and the curve acc(k) is the bits-of-feedback scaling law.

**A2 `prealigned_init`** (spec frozen now, wired later) — FA with fixed
random seed-shared B; W_0 initialised as ternarize(B^T) per hidden layer
(0.7-absmean sparsification, matching TernaryNet init), layer 0 unchanged.
Tests whether alignment created at init survives flip training.

**A4 `fwd_cv`** (spec frozen now, wired later) — DFA + forward-gradient
control variate: g_est = g_dfa + gamma * (g_fwd - E[g_fwd | dfa direction]),
g_fwd from M=4 JVP tangent probes per step (forward-only), gamma frozen at 1.
Vote = -sign(g_est). fb_comm 0; cost +M forward passes/step.

**A5 `wallclock_stale`** (spec frozen now, wired later) — throughput contest,
not accuracy: bp_shadow (depth-sequential backward) vs DFA with one-step-stale
error fully overlapped with the next forward (simulated pipeline). Metric:
wall-clock time-to-0.55 test acc on this machine, single numpy process.

## Gates (GO/NO-GO, frozen)

- **G-A1**: `kp_fa` mean acc >= 0.60 -> GO (learned alignment closes the
  barrier at zero communication).
- **G-A3**: any k >= 8 with mean acc >= 0.60 -> GO (<= 0.2 bits/w/step buys
  back the gap).
- G-A2: mean acc >= 0.60. G-A4: mean acc >= 0.60 at fb_comm 0.
- G-A5: stale-DFA wall-clock time-to-0.55 strictly less than bp_shadow's.

## Pre-registered predictions

- P1: acc(k) monotone non-increasing in k (within seed noise).
- P2: `sign_transport_k1` sign_match_p == 1.0 exactly (structural; a violation
  means the code is wrong -> fix and log a deviation, do not reinterpret).
- P3: `kp_fa` late-training sign_match_p (final 10% of steps) > 0.75 if
  learned alignment works (round-1 barrier: 0.70-0.71).
- P4: `fa_random` ~= dfa_k32 (.536): chaining topology alone does not help.
- P5 (interpretation rule, frozen now): if even `sign_transport_k1` mean acc
  < 0.60, the residual gap is MECHANISM-limited (vote/flip dynamics), not
  feedback-fidelity-limited, and A1/A3 NO-GOs must not be read as evidence
  about the sign barrier itself.

## Procedure

- Smoke runs (<= 300 steps, seed 0 only) for crash/shape checks and the P2
  structural assert are executed before freezing; their accuracies are
  discarded and not used for tuning anything.
- Hyperparameters above are frozen. Any post-hoc sweep must be labeled
  EXPLORATORY and cannot flip a gate.
- Analysis: mean/std over seeds {0,1,2}; no exclusions; all rows written to
  `data/report/feedflip/feedflip_round2.json`.
- Metrics per row: acc, ms_per_step, fwd/fb accounting, cumulative
  sign_match_p, and sign_match_p over the final 10% of steps.

## Deviations

- **D1** (pre-freeze, during the declared smoke runs): P2 initially failed at
  p = 0.9997 with scale-free B = W^T. Cause: floating-point rounding — deltas
  through B = W^T vs eff^T = (alpha W)^T differ by per-layer positive scales;
  mathematically sign-identical, but near-cancellation dot products can round
  to opposite signs. Fix (code, per the P2 rule): B_j now copies eff(j)^T,
  which is bit-identical to the BP backward matrix. alpha is fixed at init and
  seed-derivable at both ends, so communicated bits are unchanged (1.585/k for
  the ternary pattern). Smoke re-run must show sign_match_p == 1.0 exactly.
  No accuracies from either smoke run were used for tuning.
