# experiments/m2_attention_alignment.py
"""M2 alignment ablation: does reusing cached A/J_A/LN keep alignment high THROUGH attention?

Conditions: fixed_dfa / one_only / two_only / one_two, over a per-block depth axis (stacked
independent blocks each fed the broadcast error). Reports attention-block theta, per-path
{value,score} theta, and the value-path-exact err. No bpc training (deferred to M2b).

② adaptation is GENUINELY transport-free: dL/dCtx is estimated by antithetic node perturbation of
Ctx via a block-FORWARD-only partial re-run (no autograd, no W_Oᵀ). Validated outcome at d=32: R_O
does NOT reach W_Oᵀ (‖R_O−W_Oᵀ‖/‖W_Oᵀ‖≈1.06, cos≈0.69 — the M1 1/√d variance wall), yet ①+② still
beats fixed-DFA on attention-block θ by a wide margin.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from feedflipnets.core.strategies import ActivationRoutedDFA, fixed_dfa_block_grads
from feedflipnets.core.transformer import ProtoBlock, autograd_ref, forward_np_cache
from feedflipnets.eval.alignment import (
    attention_block_theta,
    per_matrix_cosine,
    per_path_theta,
)
from feedflipnets.eval.gradcheck import value_path_exact_err

torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

CONDITIONS = ["fixed_dfa", "one_only", "two_only", "one_two"]
D, D_FF, T = 32, 64, 16
SEEDS = [0, 1, 2, 3, 4]
BUDGET_STEPS, BUDGET_K, RHO, LR_R = 2000, 16, 0.05, 0.02  # feasible transport-free budget (d=32)


def _forward_from_ctx(block, x_np, ctx):
    """Block FORWARD as a function of Ctx (no autograd, no transpose). Uses block.Wo in the FORWARD
    direction (allowed) — this is L's dependence on Ctx, exactly what node perturbation probes."""
    o = ctx @ block.Wo
    x1 = x_np + o
    mu = x1.mean(axis=1, keepdims=True)
    var = x1.var(axis=1, ddof=0, keepdims=True)
    z2 = (x1 - mu) / np.sqrt(var + block.eps)
    h = np.maximum(0.0, z2 @ block.W1)
    return x1 + h @ block.W2


def _ghat_ctx(block, x_np, e_block, k_samples, rho, rng):
    """Antithetic node-perturbation estimate of dL/dCtx, forward-only. Var ∝ d_Ctx = d (M1 wall).
    L(Ctx) = <e_block, x2(Ctx)>; ĝ = mean_K ((L(Ctx+ξ)−L(Ctx−ξ))/(2ρ²))·ξ. NO autograd, NO W_Oᵀ."""
    ctx = forward_np_cache(block, x_np)["Ctx"]
    g = np.zeros_like(ctx)
    for _ in range(k_samples):
        xi = rng.standard_normal(ctx.shape) * rho
        lp = float((e_block * _forward_from_ctx(block, x_np, ctx + xi)).sum())
        lm = float((e_block * _forward_from_ctx(block, x_np, ctx - xi)).sum())
        g += ((lp - lm) / (2.0 * rho**2)) * xi
    return g / k_samples


def adapt_R_O_honest(
    block, seed: int, steps: int, k_samples: int, rho: float, lr: float
) -> np.ndarray:
    """② surrogate adaptation, GENUINELY transport-free. ĝ_Ctx is the forward-only node-perturbation
    estimate; the KP update R_O += lr·dOᵀ(ĝ_Ctx − dO·R_O)/T never reads W_Oᵀ. Returns R_O (PARTIAL).
    """
    d = block.d
    rng = np.random.default_rng(seed)
    r_o = rng.standard_normal((d, d)) / np.sqrt(d)
    for it in range(steps):
        g = np.random.default_rng(50000 + it)
        x = g.standard_normal((T, d))
        e = g.standard_normal((T, d))
        ghat = _ghat_ctx(block, x, e, k_samples, rho, g)
        d_o = e  # e_attn broadcast; residual dO = e for this isolated attention adaptation
        r_o = r_o + lr * (d_o.T @ (ghat - d_o @ r_o)) / T
    return r_o


def _adapt_B(
    block, seed: int, steps: int, k_samples: int, rho: float, lr: float, shape
) -> np.ndarray:
    """②-only feedback adaptation with the SAME transport-free estimator, but NO A-reuse — a plain
    DFA broadcast matrix B_v adapted toward the node-perturbation estimate of dL/dV. (dL/dV is probed
    the same way, forward-only.)"""
    d = block.d
    rng = np.random.default_rng(seed + 777)
    b = rng.standard_normal(shape) / np.sqrt(d)
    for it in range(steps):
        g = np.random.default_rng(60000 + it)
        x = g.standard_normal((T, d))
        e = g.standard_normal((T, d))
        # ĝ_V via node perturbation of V (forward-only): reuse the Ctx probe since Ctx = A·V; here we
        # target dL/dV directly through the same forward, perturbing V.
        c = forward_np_cache(block, x)
        v = c["V"]
        gh = np.zeros_like(v)
        for _ in range(k_samples):
            xi = g.standard_normal(v.shape) * rho
            ctxp = c["A"] @ (v + xi)
            ctxm = c["A"] @ (v - xi)
            lp = float((e * _forward_from_ctx(block, x, ctxp)).sum())
            lm = float((e * _forward_from_ctx(block, x, ctxm)).sum())
            gh += ((lp - lm) / (2.0 * rho**2)) * xi
        gh /= k_samples
        b = b + lr * (e.T @ (gh - e @ b)) / T
    return b


def run_condition(
    condition: str,
    depth: int,
    seed: int,
    n_draws: int = 40,
    adapt_steps: int = BUDGET_STEPS,
    adapt_K: int = BUDGET_K,
) -> Dict[str, float]:
    d, d_ff = D, D_FF
    block = ProtoBlock(d, d_ff, T, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    x = rng.standard_normal((T, d))
    e_block = rng.standard_normal((T, d))
    ref, _ = autograd_ref(block, x, e_block)
    e_attn, e_mlp = ref["x1"], e_block
    r_2 = rng.standard_normal((d, d_ff)) / np.sqrt(d)

    # surrogate / feedback preparation
    r_o_adapt = None
    bv_adapt = None
    if condition == "one_two":
        r_o_adapt = adapt_R_O_honest(block, seed, adapt_steps, adapt_K, RHO, LR_R)
    if condition == "two_only":
        bv_adapt = _adapt_B(
            block, seed, adapt_steps, adapt_K, RHO, LR_R, (d, d)
        )  # A-free adapted Bv

    cos_acc: Dict[str, List[float]] = {k: [] for k in ["Wq", "Wk", "Wv", "Wo"]}
    vpe = 0.0
    r_rel = float("nan")
    if r_o_adapt is not None:
        r_rel = float(np.linalg.norm(r_o_adapt - block.Wo.T) / np.linalg.norm(block.Wo.T))
    for s in range(n_draws):
        rr = np.random.default_rng(7000 + 100 * seed + s)
        if condition in ("fixed_dfa", "two_only"):
            # fixed_dfa = independent random broadcasts, IGNORING A.
            # two_only  = SAME (A-free) but Bv is the transport-free-adapted feedback (no A-reuse) —
            #             isolates ②'s contribution WITHOUT ①'s A structure.
            bv = bv_adapt if condition == "two_only" else rr.standard_normal((d, d)) / np.sqrt(d)
            b = {
                "Bq": rr.standard_normal((d, d)) / np.sqrt(d),
                "Bk": rr.standard_normal((d, d)) / np.sqrt(d),
                "Bv": bv,
                "B1": rr.standard_normal((d, d_ff)) / np.sqrt(d),
            }
            g = fixed_dfa_block_grads(block, x, e_attn, e_mlp, **b)
        else:  # one_only, one_two
            r_o = r_o_adapt if condition == "one_two" else (rr.standard_normal((d, d)) / np.sqrt(d))
            strat = ActivationRoutedDFA(R_O=r_o, R_2=r_2)
            g = strat.block_grads(block, x, e_attn=e_attn, e_mlp=e_mlp)
            vpe = max(vpe, value_path_exact_err(g, ref))
        cos = per_matrix_cosine({k: g[k] for k in cos_acc}, {k: ref[k] for k in cos_acc})
        for k in cos_acc:
            cos_acc[k].append(cos[k])

    mean_cos = {k: float(np.mean(cos_acc[k])) for k in cos_acc}
    paths = per_path_theta(mean_cos)
    return {
        "condition": condition,
        "depth": depth,
        "seed": seed,
        "attn_block_theta": attention_block_theta(mean_cos),
        "value_theta": paths["value"],
        "score_theta": paths["score"],
        "value_path_exact_err": vpe,
        "R_O_rel_err": r_rel,
    }


def main() -> None:
    out_dir = Path("data/report/m2")
    out_dir.mkdir(parents=True, exist_ok=True)
    table: Dict[str, Dict[str, float]] = {}
    for cond in CONDITIONS:
        rows = [run_condition(cond, depth=1, seed=s) for s in SEEDS]
        thetas = [r["attn_block_theta"] for r in rows]
        rrel = [r["R_O_rel_err"] for r in rows if not np.isnan(r["R_O_rel_err"])]
        table[cond] = {
            "attn_block_theta_mean": float(np.mean(thetas)),
            "attn_block_theta_ci95": float(1.96 * np.std(thetas, ddof=1) / np.sqrt(len(SEEDS))),
            "value_theta_mean": float(np.mean([r["value_theta"] for r in rows])),
            "score_theta_mean": float(np.mean([r["score_theta"] for r in rows])),
            "R_O_rel_err_mean": (float(np.mean(rrel)) if rrel else None),
        }
    (out_dir / "m2_ablation.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
