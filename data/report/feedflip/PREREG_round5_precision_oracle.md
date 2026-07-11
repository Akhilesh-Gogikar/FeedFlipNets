# PRE-REGISTRATION — FeedFlip round 5: precision oracle

Frozen before any full run. Harness: `experiments/feedflip_precision_oracle.py`.
Benchmark, task, batch stream (seed+900), mechanism stream (seed+1300), net init
(seed+100), STEPS=8000, BATCH=16, SEEDS=[0,1,2] — all identical to rounds 1–4.

## Question P5

Round 4's best 8-bit arm plateaued at **0.572** (e025_st512) vs the float32
anchor **0.604** (bp_shadow). Round 4's bp-transport ceiling arm entered a
zero-gradient absorbing state (validity guard fired), so precision vs
transport/feedback attribution stayed open. **Is the residual 0.572 → 0.604 gap
attributable to shadow-state precision (float32 vs 8 bits)?**

## Key mechanical fact motivating the design

`run_bp_shadow` (the 0.604 anchor) is NOT backward-through-float: it backprops
**through the ternarized weights** (STE-style), re-deriving alpha and the
ternary pattern from the float shadow's absmean **every step**
(thr = 0.7·mean|Wf|), with plain SGD `Wf -= 0.1·g`. The only thing an 8-bit arm
cannot copy is the float32 storage. So the cleanest precision isolation is an
int8 clone of that exact loop — not a backward-through-shadow variant.

## Arms (6 arms × 3 seeds × 8000 steps; all run, no post-hoc selection)

| arm | role | design | state | comm (bits/w/step) |
|---|---|---|---|---|
| i8clone_g8  | oracle | bp_shadow loop, shadow on int8 grid, q_l = mean\|Wf_l(init)\|/8  | 8 b/w | — (full BP) |
| i8clone_g16 | oracle | same, G=16 | 8 b/w | — |
| i8clone_g32 | oracle | same, G=32 | 8 b/w | — |
| bps_e025 | oracle | round-4 ishadE update (E=0.25), backward through **shadow-derived** eff (liveness-protected) | 8 b/w | — |
| bps_e05  | oracle | same, E=0.5 | 8 b/w | — |
| st512s_e025 | **gate** | round-4 e025 update, stale chained B from shadow-derived eff^T, refresh every 512 steps | 8 b/w | 8/512 = 0.015625 |

i8clone details (only axis moved vs anchor = precision): update
`n_l += SR(−0.1·g/q_l)`, clip ±127; per-layer grid q_l frozen at init; alpha and
ternary W re-derived from `n_l·q_l` each step exactly as the anchor. Backward
through ternarized W (same as anchor) — inherits the round-4 death risk, hence
the guard below and the liveness-protected bps fallback whose backward path
(`alpha0_j·shadow_j/8`, bit-exact to eff at init) cannot be zeroed by ternary
cells. All three G values run regardless of smoke outcome (no grid selection).

## Validity guard (unchanged from round 4)

Dead step := all layers have mean|g| = 0 at that step. An oracle arm is VALID
iff n_dead = 0 on **all** seeds. Invalid arms are excluded from C5. If no
oracle arm is valid, the guard fires and P5 stays unresolved (report as such).

## Decision rule P5 (boundaries frozen, carried from round 4)

C5 := max over VALID oracle arms of mean test acc across seeds.

- **C5 ≥ 0.604** → precision NOT binding at 8 b/w: an 8-bit state matches the
  float anchor given exact gradients + matched dynamics; the 0.572 plateau is
  attributable to transport/feedback (or update-rule interaction), and the
  frontier claim "8-bit state caps at ~0.57" is overturned.
- **C5 < 0.59** → precision binding CONFIRMED: 8-bit accumulation caps below
  the anchor even with exact gradients and matched dynamics; the residual gap
  is a genuine precision cost.
- **0.59 ≤ C5 < 0.604** → unresolved band; report partial attribution, no
  frontier update.

## Gate G_R5 (gate-eligible arm only: st512s_e025)

GO iff mean acc ≥ 0.60 (= GATE_ACC, unchanged since round 1); strict success
iff mean acc ≥ 0.604. State ≤ 8 bits/w; comm 0.015625 bits/w/step.

## Metrics (all arms)

acc, gap_closure = (acc − 0.572)/(0.604 − 0.572), n_dead, dead_step, sat_frac
(fraction of shadow cells at ±127 at end), ms/step; i8clone adds alpha_ratio
(final/init mean); st512s adds sign_match_p and sign_match_p_late vs BP
(hidden layers, nonzero-pair convention as rounds 3–4).

## Smoke policy (disclosed)

One smoke run (300 steps, seed 0 only, all arms) checks liveness and runtime
ONLY. No accuracy-based tuning; no arm, grid value, or constant may change
after the freeze commit except to fix a crash, and any such fix is recorded as
a Deviation in the results doc.

## Frozen constants

LR_SHADOW=0.1 · ISHAD_CLIP=127 · EMA=0.99 · ST_K=512 · G∈{8,16,32} ·
E∈{0.25,0.5} · REF_BP_SHADOW=0.604 · REF_BEST_8BIT=0.572 · CEIL_HIGH=0.604 ·
CEIL_LOW=0.59 · thr=0.7·absmean · SR = floor + Bernoulli(frac), stream seed+1300.
