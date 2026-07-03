# Speed & binding-constraint analysis — is DFA a backprop killer on the GPU-dollar axis?

Prompted by the right question: *a slow DFA is pointless — it must be cheaper than backprop to save
GPU dollars.* This note records the two analyses that answer it, and the honest conclusion.
Prototypes: `scratchpad/pipeline_cost_model.py`, `scratchpad/accuracy_transport_frontier.py`
(analytical + controlled deep-MLP; not productionized — first-cut evidence for the strategic call).

## 1. Per-step speed (measured, M2b): DFA is SLOWER on one device

`data/report/m2b/`: ①+② is **~30× slower per step** than BP on CPU. Driver: ②'s node-perturbation
adds extra forward passes; ① adds a routing matmul. On a single device doing dense matmuls, DFA is
equal-or-slower per step **by construction**. So the per-step axis is a loss.

## 2. Pipeline throughput (analytical cost model): a win, but CONDITIONAL

DFA's only real efficiency edge is backward **parallelism** — no weight transport, no sequential
backward, no global sync — which cashes out only as multi-GPU pipeline throughput. Model
(`tf=1`, `tb=2`; DFA-pipeline vs tuned 1F1B+checkpointing):

| regime | DFA/BP throughput |
| ------ | ----------------- |
| m=4S, memory just fits, `tu≈tf` (cheap update) | **1.66×** |
| m=4S, memory just fits, `tu≈tb` (update ≈ backward cost) | **1.15×** |
| tight memory (BP forced to checkpoint/recompute, DFA not) | **2.2–2.85×** |

So the pipeline thesis is **not dead** — it clears 1.5× **iff** DFA's per-stage update `tu` is cheaper
than a BP backward, or BP is memory-bound. But it **hinges on `tu`** (unmeasured), and the model
**ignores the `e`-broadcast comm** (`|e|·S`, large at real vocab) which would erode it.

## 3. The coupling that dominates everything: throughput ≠ GPU dollars

GPU dollars = **wall-clock to a *target loss*** = (steps-to-target) × (time-per-step). DFA can win the
second factor, but **M2b showed it loses the first decisively**: ①+② plateaus at ~2.7 bpc while BP
reaches 0.77. DFA never reaches BP's loss, so **wall-clock-to-BP-quality is infinite** — a 2×
throughput edge is irrelevant. **A faster path to a worse model saves no GPU dollars.**

## 4. Is the accuracy gap cheaply closeable? (frontier probe): NO free lunch

Sweeping `λ`: update = `(1-λ)·DFA + λ·BP` on a controlled deep-MLP teacher task (gap 6pp):
recovery **tracks λ roughly proportionally** — 25% of the gap at λ=0.10, 52% at λ=0.25, 82% at λ=0.50.
There is **no cheap-closure regime** where a little transport recovers most of the accuracy. To get
BP-competitive accuracy you need most of BP's transport — which erases the parallelism advantage.
(Caveat: 6pp-gap task; suggestive, not the large-gap LM regime — but consistent with the DFA
literature that closing the gap costs transport ~proportionally.)

## Conclusion: the binding constraint is ACCURACY, not pipeline mechanics

- **① Activation-Routed DFA is a genuine, novel, transport-free advance in feedback-alignment
  *quality*** (value-exact grads; worst-case attention alignment 90°→46°; beats plain DFA everywhere).
- **But it is not a backprop killer.** The chain fails at every economic axis:
  per-step slower (§1); pipeline-throughput win only conditional (§2); **irrelevant because DFA
  converges to a worse loss so time-to-target is unbounded** (§3); and the accuracy gap does **not**
  close cheaply with partial transport (§4).
- **The leverage — if any exists — is entirely in closing the accuracy gap** (a stronger,
  input-conditioned or gradient-recovering credit-assignment mechanism), not in the pipeline model.
  The evidence now maps this as a hard, well-studied wall: methods that fully close the gap (predictive
  coding, target prop) reintroduce sequential transport — the Conway trap — killing the speed premise.

Honest bottom line: **as a training algorithm, this family is worse than backprop on every axis that
saves GPU dollars.** Its real contribution is the alignment result, recorded as a characterization.
