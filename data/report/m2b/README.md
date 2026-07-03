# M2b result — does ①+②'s attention-alignment win buy competitive accuracy? (NO-GO)

Source: `m2b_lm_ablation.json` (`.venv/bin/python -m experiments.m2b_lm_ablation --corpus cfg --seeds 5 --steps 1500`).
Char-level CFG-grammar corpus (vocab 38, bigram floor **2.881 bpc**), float only, thread-pinned.

## Result (val bits-per-char, n=5)

| condition | val bpc | vs BP | per-step |
| --------- | ------- | ----- | -------- |
| tuned BP (backprop) | **0.774 ± 0.007** | — | ~5 ms |
| ①+② (Activation-Routed DFA + node-perturbation ②) | **2.702 ± 0.088** | **+1.93** | ~150 ms |
| fixed-DFA | 3.804 ± 0.055 (fails to beat bigram) | +3.03 | ~60 ms |

- **Gate (spec §6/M2): ①+② − BP ≤ 0.15 bpc, 95% CI upper < 0.25 → FAIL.** Gap = 1.929 ± 0.091 bpc, ~13× the threshold. `decision: NO-GO`.
- ①+② **beats fixed-DFA** by 1.10 bpc and edges below the bigram floor — a genuine learner, just far from BP.
- Guards PASS: ①'s value path is exact in the multi-block LM (`dW_O`,`dW_2` ~1e-15 vs a causal autograd ref); ②'s adaptation is transport-free (no `autograd.grad`, no `W_Oᵀ`; positive control flagged).

## The two ways this loses to backprop (both measured)

1. **Accuracy:** ~1.93 bpc worse than BP, and still descending but decelerating — closing a 2-bpc gap to 0.15 is implausible. Alignment is **necessary but not sufficient**: M2 showed ①+② lifts worst-case attention alignment from ~90° (random) to ~46°, but that does **not** translate into competitive next-char loss.

2. **Per-step speed:** ①+② is **~30× slower per step than BP** on this CPU testbed. This is dominated by **②'s node-perturbation adaptation**, which estimates gradients transport-free by running *extra forward passes* per adapted block. ①'s own routing also adds a matmul BP doesn't do.

## Honest framing of the speed number (important)

The ~30× per-step slowdown is **not** DFA's intrinsic cost and **not** the thesis. On a single device doing dense matmuls, DFA is equal-or-slower per step by construction (it does an extra broadcast/routing matmul, and node-perturbation ② adds forwards). **DFA's only real efficiency advantage is backward-pass *parallelism* — no weight transport, no sequential backward, no global sync — which only cashes out as wall-clock/energy on multi-GPU pipeline-parallel training** (bubble removal + reduced stashed-activation memory + async tolerance). A pure-NumPy CPU testbed can never demonstrate that axis; see `docs/backprop-killer-research-plan.md` (Crux 1).

Corollary: **node-perturbation ② is a speed *liability*.** It is the accuracy fix, and it reintroduces exactly the compute cost DFA is meant to save. A "save-GPU-dollars" DFA must use a *cheap local* feedback-learning rule (e.g. Kolen–Pollack: one extra outer-product, no extra forwards), not node perturbation, and be measured on pipeline-parallel hardware.

## Arc conclusion (M1 → M2 → M2b)

- **① Activation-Routed DFA is a real, novel, transport-free advance in feedback-alignment *quality*** — value-exact gradients + worst-case attention alignment 90°→46°, beating plain DFA everywhere.
- **But it is not backprop-competitive as a trainer.** ② alone can't bend the depth-slope (M1, NO-GO); ①+②'s alignment gains don't close the accuracy gap (M2b, NO-GO); and per-step it is far slower on a single device.
- **As built, it is worse than backprop on both accuracy and per-step speed.** The value proposition ("cheaper than BP") lives on a *different, unbuilt* axis — multi-GPU pipeline wall-clock with a cheap local feedback rule — not in this algorithm-on-a-laptop.

Recorded as an honest NO-GO. The gate was not met; nothing was tuned to force a pass.
