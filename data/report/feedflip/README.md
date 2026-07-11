# FeedFlip: feedback-driven bit-flips for ternary nets — and the per-weight sign barrier

The real thesis of FeedFlipNets: train ternary networks by using a cheap feedback signal to **flip
bits directly**, with **no float shadow weights**: store only ternary `W ∈ {-1,0,+1}` (≈1.58 bits)
plus a small signed integer flip-accumulator `c` per weight. Each step: `c += -sign(grad_est)`; when
`|c| ≥ K`, flip the bit (`-1↔0↔+1`) and reset. A flip needs only the **sign**, so the natural
feedback is DFA (cheap, transport-free, shadow-free). Prototypes: `experiments/feedflip_bitflip.py`
(MLP), `experiments/ternary_cpu_headtohead.py`, and the transformer flip study (scratchpad).

## Bottom line (five pre-registered rounds)

On the frozen MLP benchmark, a float32 shadow trained through the ternary forward pass reaches
**0.604** accuracy. No transport-free method at ≤ 8 bits of optimizer state per weight beat **0.574**
across five rounds, each with a gate frozen before the run. The residual gap is **not** what the first
four rounds successively guessed: not per-weight sign fidelity (round 2 gave the mechanism bit-exact
gradient signs and gained almost nothing), and not accumulator precision (round 5's int8 shadow
*matched* the anchor, 0.609, once it borrowed the anchor's update rule). What remains is a single,
sharply located obstacle: a transport-free feedback signal good enough to drive a **magnitude-scaled,
fixed-grid weight step**, the update rule that separates the 8-bit plateau from the float anchor.

| round (§) | question it froze | gate | what it established |
| --- | --- | --- | --- |
| 1 (§4) | can a better transport-free *sign* close the gap? | NO-GO · 0.541 | sign quality alone does not; the barrier holds |
| 2 (§5) | is sign *fidelity* the bottleneck? | NO-GO · 0.551 at `p=1.0` | no; even perfect signs stall (overturns round 1) |
| 3 (§6) | do magnitude-carrying votes help? | NO-GO · 0.574 | partly; looked like a state-*precision* frontier |
| 4 (§7) | does a gentler int8 shadow break the plateau? | NO-GO · 0.572 | same plateau; the precision question stays open |
| 5 (§8) | can 8-bit state reach the anchor with exact grads? | gate NO-GO · 0.554; oracle **0.609** | precision is *not* binding (overturns rounds 3–4); the gap is the update rule |

Sections 1–3 set up the mechanism and the barrier; sections 4–8 are the five rounds above, in order.
Each round's later revision (bottom of this page) records where a subsequent round corrected it, so the
reasoning trail is honest rather than retrofitted.

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
  (2.01)**: flipping bits from the DFA sign is worse than not flipping. K32 only claws back to ~floor.
- **AR-DFA does not rescue it:** it ties fixed-DFA, and per-weight sign-match `p ≈ 0.53` for **both**,
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
BP-sign flips here **lag** the float-shadow anchor (0.489 vs 0.604); the uncommitted
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
  0.71→0.75 yet gained only +0.005 over dfa_k32, within seed noise.
- **taught_B ② learns a *different* direction, not a better one:** `p` drops to 0.57
  (below fixed-B) while accuracy stays at baseline; perturbation-taught feedback
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

**Both gates NO-GO** (G-A1 kp_fa ≥ 0.60: no; G-A3 any k ≥ 8 ≥ 0.60: no). The
pre-registered interpretation rule **P5 fired**: even k=1, which feeds the flip
mechanism the *bit-exact BP backward pass* (`p = 1.0` by construction), reaches only
0.551.

- **The §4 "sign barrier" framing is overturned.** Going from p ≈ 0.70 (random B) to
  p = 1.0 (perfect transport) buys only +0.015 of the 0.07 gap to bp_shadow. The
  binding constraint is the magnitude-blind vote/flip accumulator, not feedback sign
  fidelity: what the float shadow buys is *magnitude accumulation*, not better signs.
- **Sign fidelity is nearly free anyway:** acc(k) is flat out to k=4096
  (0.0004 bits/w/step, late p 0.87, acc 0.557): stale ternary W^T is as good as
  fresh transport. P1 (monotone decay in k) is not observed in this regime.
- **kp_fa diverged** (P3 falsified): gradient co-mirroring under flip dynamics
  bootstraps its own feedback target (late p 0.33, acc 0.30); the KP theorem's
  shared-update premise genuinely matters; W moving by flips breaks it.
- **P4 confirmed:** fa_random (0.530) ≈ dfa_k32 (0.536): chaining topology alone
  changes nothing.

Consequences: registered arms A2 (prealigned init) and A4 (forward-gradient control
variate) target sign fidelity and are now predicted capped near 0.55; A5 (wall-clock
throughput) is unaffected. A round-3 attack should target the *mechanism*:
magnitude-carrying votes (e.g. `c += -sign(g)·q(|g|)` with a 2–3-bit quantizer,
or stochastic-rounding flips with P(flip) ∝ |g|), staying ≤ 8 state bits/w.

## 6. Round 3 — magnitude-carrying votes (pre-registered): the residual gap is state *precision*, not information.
`experiments/feedflip_magnitude_votes.py`; pre-registration + gates frozen at commit
`06b13da` (`PREREG_round3_magnitude_votes.md`) **before** any full run. Same frozen
benchmark; new axis: gradient *magnitude* inside the ≤ 8 bits/w accumulator.
Mechanisms: magq2 (2-bit weighted votes), stochT (Bernoulli vote rate ∝ |g|;
unbiased integrator), ishadE (int8 SR shadow), each × {st512 stale ternary
transport (0.0031 bits/w/step), farand (0 bits)}. Results (`feedflip_round3.json`):

| arm | mean acc | late sign-match `p` | state bits/w |
| --- | -------- | ------------------- | ------------ |
| bp_shadow (§4 anchor) | 0.604 ± .013 | — | 32.0 |
| **stoch1_st512** | **0.574** ± .009 | **0.975** | 7.60 |
| stoch4_st512 | 0.569 ± .008 | 0.969 | 7.60 |
| stoch4_farand | 0.568 ± .009 | 0.774 | 7.60 |
| magq2_st512 | 0.560 ± .029 | 0.918 | 7.69 |
| magq2_farand | 0.555 ± .019 | 0.776 | 7.69 |
| sign-only reference (§5, k=512) | 0.557 ± .006 | 0.974 | 7.60 |
| stoch1_farand | 0.544 ± .006 | 0.728 | 7.60 |
| ishad{2,8}_* (int8 SR shadow) | 0.330 ± .000 | — (`g ≡ 0`) | 8.00 |

**All gates NO-GO** (best 0.574 < 0.60; strict 0.604 unreached; no vote arm ≥ 0.60).
Three findings survive:

- **A state-precision frontier, not an information gap.** With late sign-match
  at 0.975 *and* quantized magnitude in the votes, accuracy gains only +0.017
  over sign-only, ~37% of the gap to the float shadow. The information
  channel is saturated; what float32 still buys is *precision of
  accumulation*: 0.557 (sign-only, 7.6 b) → 0.574 (magnitude votes, 7.6 b) →
  0.604 (float shadow, 32 b). P2 missed its frozen bar narrowly (+1.9σ vs > 2σ).
- **Feedback quality re-emerges once votes are unbiased (P3 violated for
  stoch1):** st512 − farand = +0.030 (late `p` 0.975 vs 0.728): the first
  arm family in this series where transport quality moves accuracy beyond
  noise. Round 2's "sign fidelity is nearly free" holds only for
  magnitude-blind votes.
- **ishad caveat (exploratory diagnostic):** both int8 SR-shadow arms entered a
  zero-gradient absorbing state by ~step 60 (E-scaled SR steps churn weights
  through the ternarizer; units die; `g ≡ 0` ⇒ SR(0) = 0; 0.330 identically
  across seeds/feedback). Not int8 saturation, but a step-size dynamics failure.
  The frontier reading therefore rests on the vote arms, and P1's predicted
  ordering (ishad ≥ stoch ≥ magq2) is refuted.

## 7. Round 4 — gentle-step int8 SR shadow (pre-registered): ishad rescued, but it lands on the same plateau; the bp ceiling control is dynamically invalid.
`experiments/feedflip_ishad_gentle.py`; pre-registration + gates frozen at commit
`5dbe4fc` (`PREREG_round4_ishad_gentle.md`) **before** any full run. Round 3's ishad
arms (E ∈ {2, 8}) died in an absorbing state; round 4 re-probes the identical int8
SR-shadow mechanism at E ∈ {0.25, 0.5, 1.0}, adding a bit-exact-BP oracle per E as a
*ceiling control* (never gate-eligible) to separate accumulation-precision capacity
from feedback quality. Results (`feedflip_round4.json`, n=3):

| arm | mean acc | late sign-match `p` | dead seeds |
| --- | -------- | ------------------- | ---------- |
| **ishad_e025_st512** | **0.572** ± .014 | **0.970** | 0 |
| ishad_e05_st512 | 0.556 ± .011 | 0.964 | 0 |
| ishad_e025_farand | 0.548 ± .013 | 0.782 | 0 |
| ishad_e05_farand | 0.517 ± .023 | 0.766 | 0 |
| ishad_e10_farand | 0.441 ± .079 | 0.753 | 1 |
| ishad_e10_st512 | 0.390 ± .085 | 0.857 | 2 |
| ishad_e025_bp (oracle) | 0.468 ± .098 | — | 1 |
| ishad_e05_bp / e10_bp (oracle) | 0.330 ± .000 | — | 3 / 3 |

**Gate: NO-GO** (best eligible 0.572 < 0.60; strict 0.604 unreached). **P4 returned
no precision conclusion**. The frozen validity guard fired: no bp oracle arm
survived dead-free. Findings:

- **Round 3's ishad caveat is confirmed as a step-scale dynamics failure, not
  mechanism incapacity.** At E = 0.25 the same int8 SR shadow trains cleanly to
  0.572, statistically indistinguishable from round-3's best vote arm (0.574).
  Accuracy is monotone in gentleness (0.572 → 0.556 → dead-ridden as E rises;
  the collapse boundary sits between E = 0.5 and E = 1.0).
- **The 8-bit-state plateau is mechanism-independent.** Vote counters and SR
  shadows (two different ≤ 8 b/w accumulators) both cap at ≈ 0.57, reinforcing
  the state-bits-frontier reading of §6 rather than revising it.
- **Transport quality is again first-order** (replicating §6's stoch1 reversal in
  a second mechanism): st512 − farand = +0.023 (E=.25) / +0.039 (E=.5), late `p`
  0.97 vs 0.78.
- **The bp oracle is dynamically invalid, not weak (exploratory).** BP-fed shadows
  died at every E (even E=0.25, 1/3 seeds; alive seeds reached 0.530/0.545):
  exact gradients flow *through* downstream ternary weights, so when layers
  ternarize toward zero the gradient vanishes and SR(0) = 0 is absorbing. B-matrix
  feedback bypasses downstream W and never died at E ≤ 0.5, so feedback-driven arms
  are *more* stable than exact-gradient arms here. The question the oracle was
  built to answer (would bit-exact BP through an 8-bit accumulator reach 0.604?)
  remains open and needs a liveness-protected oracle design.

## 8. Round 5 — precision oracle (pre-registered): the frontier's far end falls; 8-bit state DOES reach the anchor.

PREREG frozen at `08daadb` before any full run
(`data/report/feedflip/PREREG_round5_precision_oracle.md`, harness
`experiments/feedflip_precision_oracle.py`, results `feedflip_round5.json`).
The design exploits a mechanical fact about the 0.604 anchor itself:
`bp_shadow` already backprops *through the ternarized weights* (STE-style),
re-deriving alpha and the ternary pattern from the shadow's absmean every
step; the only thing an 8-bit arm cannot copy is the float32 storage. So
the primary oracle is an **int8 SR clone of the anchor's exact loop**
(per-layer grid `q_l = mean|Wf_l(init)|/G`, update `n += SR(−0.1·g/q_l)`,
clip ±127), moving *only* the precision axis. G ∈ {8, 16, 32} all ran (no
selection); a backward-through-shadow arm (`bps`) served as
liveness-protected fallback, and `st512s` (stale 8-bit shadow feedback,
0.0156 b/w/step) was the gate-eligible arm.

| arm | mean acc (n=3) | valid (n_dead=0) |
|---|---|---|
| i8clone_g8 | 0.563 | yes |
| i8clone_g16 | 0.592 | yes |
| **i8clone_g32** | **0.609** (0.623/0.603/0.601) | **yes** |
| bps_e025 / bps_e05 | 0.330 (collapse) | no — guard fired |
| st512s_e025 (gate) | 0.554 | yes |

**P5 resolved: C5 = 0.609 ≥ 0.604, so precision is NOT binding at 8 bits/w.**
The frozen decision rule lands in the overturn zone: an 8-bit stochastically
rounded shadow, given the anchor's own update rule and transport, matches the
float32 anchor dead-free on all seeds (sat ≤ 1.2%, alpha drift 1.8×).
**Gate: NO-GO** (0.554 < 0.60). Findings:

- **The ≈0.57 plateau was never a precision cost.** What separates the plateau
  arms from the anchor is the *update rule*: rounds 3–4 arms take
  s_ema-normalized fixed-size steps (magnitude re-blinded per step), while the
  anchor and i8clone take magnitude-scaled steps `−0.1·g` on a fixed grid.
  Carrying |g| into the *step size*, not merely into vote rates, is worth the
  final +0.03 even under 8-bit accumulation.
- **Grid resolution is monotone and saturates within int8**: 0.563 → 0.592 →
  0.609 for G = 8/16/32 (typical init cell ±G, clip 127 leaves 4× headroom at
  G=32 with ~1% saturation).
- **"Liveness-protected" backward-through-shadow is not protected.** Both bps
  arms collapsed to the majority class (bps_e05 near-absorbing: 6.5–8.0k dead
  steps of 8k). The round-4 death mode is *activity* death (all hidden ReLUs
  off ⇒ zero inputs ⇒ zero g everywhere), not backward-path death; rerouting
  the backward matrices around the ternary zeros does not prevent it.
- **Higher-fidelity stale transport did not help the gate arm**: shipping 8-bit
  shadow values instead of ternary signs (5× the refresh bits of round-4's
  st512) *lost* 0.018 vs round 4's 0.572 despite late sign-match p ≈ 0.91–0.94.

## The unifying insight: cosine alignment ≠ per-weight sign
AR-DFA's large *cosine*-alignment gain on attention (worst-case angle 90°→46°, `data/report/m2/`)
does **not** translate into per-weight *sign* correctness: `p ≈ 0.53`, barely above chance. Bit-flips
read only the sign, so the alignment gain is invisible to them. **The binding constraint for
transport-free discrete/ternary learning is per-weight sign accuracy; "alignment" (the standard
cosine DFA metric) is the wrong proxy for it.** The FeedFlip *mechanism* is genuinely efficient:
≈5.6× less state, ~2× faster, shadow-free. What stays open is a transport-free rule whose per-weight
sign, not merely its aggregate direction, is reliably correct at depth and through attention.

**Round-2 revision (§5):** on the MLP benchmark, sign accuracy proved *necessary but not sufficient*.
Even p = 1.0 (bit-exact BP votes) recovers only 0.551 against bp_shadow's 0.604. Beyond sign
correctness, the deeper constraint is the flip accumulator's magnitude-blindness: the K-threshold
vote counter discards |g|, exactly what the float shadow retains.

**Round-3 revision (§6):** magnitude-blindness was also not the last wall. With sign
fidelity saturated (late p 0.975) *and* 2-bit/stochastic magnitude in the votes,
transport-free ternary training still caps at 0.574 vs 0.604. Across three pre-registered
NO-GO rounds the residual gap traced a diminishing-returns **state-bits frontier**:
0.557 (sign-only, ~7.6 b/w) → 0.574 (magnitude votes, 7.6 b/w) → 0.604 (float32 shadow,
32 b/w). Read that way, what backprop's shadow ultimately buys on this benchmark is
*accumulation precision*. One second-order reversal complicated the picture: transport
quality becomes first-order again (+0.030) once votes are unbiased and high-rate (stoch1),
so "stale/random feedback is free" is a property of magnitude-blind voting, not of the
mechanism family.

**Round-4 revision (§7):** the state-bits frontier survived a fourth pre-registered
attack, now with the mechanism axis controlled. An int8 SR shadow at the correct step
scale (E=0.25) lands on the same ≈0.57 plateau as the vote counters, so the frontier
looked like a property of ≤ 8 b/w accumulation *state* rather than of any particular
accumulator. The transport-quality reversal replicated (+0.02–0.04 across two
mechanisms). One caveat attached to the frontier's far end: the 0.604 float-shadow
anchor had not been shown reachable by exact gradients under 8-bit accumulation,
because BP-fed shadows are dynamically unstable in ternary nets. (Gradient paths die
through ternarized layers, an absorbing state that transport-free feedback happens to
escape.) So "precision is what the shadow buys" rested on the vote/shadow plateau,
pending a liveness-protected oracle.

**Round-5 revision (§8), correcting §6–§7.** The precision oracle resolves the open
question and *overturns* the "state-bits frontier" reading: an int8 (8 b/w) stochastic
shadow reaches **0.609 ≥ 0.604** when it uses the anchor's own update rule, dead-free
on all seeds. So the residual 0.57→0.60 gap was **never a bit-precision cost**. It is
the **update rule**. Rounds 1–4's ≤8-bit arms all take *step-normalized* moves: each
step re-normalizes by the per-layer gradient scale `s_ema`, re-discarding the gradient's
across-step dynamic range. The anchor and the i8clone instead take *magnitude-scaled* SGD
steps `−0.1·g` on a fixed grid, preserving the relative size of large versus small
gradients from one step to the next. Carrying magnitude into the **step size**, not
merely the vote rate, closes the gap at 8 bits. The genuinely hard problem is therefore
narrower and sharper than the four-round arc suggested: **a transport-free feedback
signal accurate enough to drive a magnitude-scaled fixed-grid step**. The gate arm
(`st512s`, stale 8-bit transport) still stalls at 0.554, so the wall is feedback
fidelity for magnitude-scaled updates, not accumulator precision and not sign
correctness. The headline number to beat is unchanged (0.604 at 32 b/w): now matched
at 8 b/w *with full transport*, and unmatched by anything transport-free.
