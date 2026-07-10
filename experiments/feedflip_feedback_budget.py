"""FeedFlip round-2: feedback-information-budget arms (pre-registered).

Wired arms (spec + gates frozen in data/report/feedflip/PREREG_round2_feedback_budget.md,
committed BEFORE any full run):
  C0 fa_random          -- layerwise-chained feedback (FA), fixed random B. Control:
                           separates chaining topology from alignment.
  A1 kp_fa              -- Kolen-Pollack mirror: float B_j learned from the SAME locally
                           available signals as layer j's own update; W is never read.
  A3 sign_transport_k*  -- B_j = stale ternary W_j^T refreshed every k steps;
                           fb_comm = log2(3)/k bits/weight/step. k=1 is the envelope
                           (exact BP vote signs; doubles as the missing feedflip_bp_k32).

Benchmark, RNG offsets, and vote/flip mechanism are identical to round 1
(experiments/feedflip_bitflip.py). Vote threshold frozen at K=32.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.feedflip_bitflip import (
    BATCH,
    DIMS,
    GATE_ACC,
    N_TRAIN,
    SEEDS,
    STEPS,
    TernaryNet,
    bp_deltas,
    make_task,
)
from feedflipnets.core.deep_mlp import output_error

Array = np.ndarray

K_VOTE = 32  # frozen: best round-1 vote threshold
FWD_BITS = round(1.58 + float(np.log2(2 * K_VOTE + 1)), 2)  # 7.6 bits/w, as round 1
TERNARY_BITS = float(np.log2(3.0))  # 1.585 bits/w per W^T refresh
KP_LR_B = 0.05  # frozen (PREREG)
KP_DECAY = 1e-3  # frozen (PREREG)

SIGN_TRANSPORT_KS = [1, 8, 64, 512, 4096]
ARMS = ["fa_random", "kp_fa"] + [f"sign_transport_k{k}" for k in SIGN_TRANSPORT_KS]


def fa_deltas(derivs, err: Array, B: List[Optional[Array]]) -> List[Array]:
    """Chained feedback: exact analog of bp_deltas with B[j] replacing eff(j).T."""
    L = len(B)
    deltas: List[Optional[Array]] = [None] * L
    deltas[L - 1] = err
    d = err
    for i in reversed(range(L - 1)):
        d = (d @ B[i + 1]) * derivs[i]
        deltas[i] = d
    return deltas  # type: ignore[return-value]


def make_B_fa(seed: int) -> List[Optional[Array]]:
    """Fixed random chained feedback, shapes matching eff(j).T; seed-regenerable."""
    rng = np.random.default_rng(seed + 700)
    B: List[Optional[Array]] = [None]
    for j in range(1, len(DIMS) - 1):
        B.append(rng.standard_normal((DIMS[j + 1], DIMS[j])) / np.sqrt(DIMS[j + 1]))
    return B


def transposed_ternary_B(net: TernaryNet) -> List[Optional[Array]]:
    """Stale copy of eff(j)^T = (alpha_j W_j)^T. alpha is fixed at init and
    seed-derivable at both ends, so only the ternary pattern (log2(3) bits/w per
    refresh) is ever communicated. eff^T (not bare W^T) makes the k=1 envelope
    bit-exact to the BP backward pass (PREREG Deviations D1)."""
    return [None] + [net.eff(j).T.copy() for j in range(1, len(net.W))]


def run_arm2(arm: str, seed: int, steps: int = STEPS) -> Dict:
    Xtr, ytr, Xte, yte = make_task()
    rng = np.random.default_rng(seed + 900)  # identical batch stream to round 1
    net = TernaryNet(seed + 100)
    L = len(net.W)

    k_refresh: Optional[int] = None
    if arm.startswith("sign_transport_k"):
        k_refresh = int(arm[len("sign_transport_k") :])
        B = transposed_ternary_B(net)
        fb_comm, fb_state = TERNARY_BITS / k_refresh, TERNARY_BITS
    elif arm == "kp_fa":
        B = make_B_fa(seed)
        fb_comm, fb_state = 0.0, 32.0  # float B: tests learnability, not deployability
    elif arm == "fa_random":
        B = make_B_fa(seed)
        fb_comm, fb_state = 0.0, 0.0  # regenerable from shared seed
    else:
        raise ValueError(f"unknown round-2 arm: {arm}")

    acc_c = [np.zeros_like(w, dtype=np.int32) for w in net.W]
    p_num = p_den = 0.0
    p_num_late = p_den_late = 0.0
    late_from = int(0.9 * steps)
    t0 = time.perf_counter()

    for step in range(steps):
        if k_refresh is not None and step % k_refresh == 0:
            B = transposed_ternary_B(net)  # PREREG: refresh before the forward pass
        idx = rng.integers(0, N_TRAIN, size=BATCH)
        Xb, yb = Xtr[idx], ytr[idx]
        inputs, derivs, logits = net.forward(Xb)
        err = output_error(logits, yb)
        deltas_bp = bp_deltas(net, inputs, derivs, err)
        deltas = fa_deltas(derivs, err, B)

        for i in range(L):
            g = inputs[i].T @ deltas[i] / BATCH
            if i < L - 1:  # barrier metric vs BP vote, hidden layers only
                g_bp = inputs[i].T @ deltas_bp[i] / BATCH
                m = (g != 0) & (g_bp != 0)
                hits = float((np.sign(g)[m] == np.sign(g_bp)[m]).sum())
                p_num += hits
                p_den += float(m.sum())
                if step >= late_from:
                    p_num_late += hits
                    p_den_late += float(m.sum())
            vote = -np.sign(g).astype(np.int32)
            acc_c[i] += vote
            flip = np.abs(acc_c[i]) >= K_VOTE
            if flip.any():
                net.W[i] = np.clip(
                    net.W[i] + np.sign(acc_c[i]).astype(np.int8) * flip, -1, 1
                ).astype(np.int8)
                acc_c[i][flip] = 0

        if arm == "kp_fa":  # PREREG: mirror update AFTER the vote/flip step
            for j in range(1, L):
                g_j = inputs[j].T @ deltas[j] / BATCH  # same local signals as layer j
                B[j] = (1.0 - KP_DECAY) * B[j] - KP_LR_B * g_j.T

    ms = (time.perf_counter() - t0) / steps * 1000
    return {
        "arm": arm,
        "seed": seed,
        "acc": net.acc(Xte, yte),
        "ms_per_step": ms,
        "fwd_bits_per_w": FWD_BITS,
        "fb_state_bits_per_w": round(fb_state, 3),
        "fb_comm_bits_per_w_step": round(fb_comm, 5),
        "sign_match_p": (p_num / p_den) if p_den else None,
        "sign_match_p_late": (p_num_late / p_den_late) if p_den_late else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default="data/report/feedflip/feedflip_round2.json")
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            r = run_arm2(arm, seed, steps=args.steps)
            rows.append(r)
            print(
                f"{arm:20s} seed={seed} acc={r['acc']:.3f} "
                f"comm={r['fb_comm_bits_per_w_step']} "
                f"p={None if r['sign_match_p'] is None else round(r['sign_match_p'], 4)} "
                f"p_late={None if r['sign_match_p_late'] is None else round(r['sign_match_p_late'], 4)} "
                f"ms={r['ms_per_step']:.2f}",
                flush=True,
            )

    summary = {}
    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm]
        ps = [r["sign_match_p"] for r in sub if r["sign_match_p"] is not None]
        pl = [r["sign_match_p_late"] for r in sub if r["sign_match_p_late"] is not None]
        summary[arm] = {
            "mean_acc": float(np.mean([r["acc"] for r in sub])),
            "std_acc": float(np.std([r["acc"] for r in sub])),
            "fb_comm_bits_per_w_step": sub[0]["fb_comm_bits_per_w_step"],
            "fb_state_bits_per_w": sub[0]["fb_state_bits_per_w"],
            "mean_sign_match_p": float(np.mean(ps)) if ps else None,
            "mean_sign_match_p_late": float(np.mean(pl)) if pl else None,
        }

    g_a1 = summary["kp_fa"]["mean_acc"] >= GATE_ACC if "kp_fa" in summary else None
    a3 = [
        a
        for a in summary
        if a.startswith("sign_transport_k") and int(a[len("sign_transport_k") :]) >= 8
    ]
    g_a3 = any(summary[a]["mean_acc"] >= GATE_ACC for a in a3) if a3 else None
    envelope = summary.get("sign_transport_k1", {}).get("mean_acc")
    gate = {
        "gate_acc": GATE_ACC,
        "G_A1_kp_fa": g_a1,
        "G_A3_comm_le_0.2_bits": g_a3,
        "P5_mechanism_limited": (envelope < GATE_ACC) if envelope is not None else None,
        "go": bool(g_a1) or bool(g_a3),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary, "gate": gate}, indent=2) + "\n")
    print(json.dumps({"summary": summary, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
