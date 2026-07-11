"""FeedFlip round-4: gentle-step int8 SR shadow (ishad revisited), pre-registered.

Round 3 (freeze 06b13da, results 81a6e35) killed its ishadE arms (E in {2,8})
via a zero-gradient absorbing state — a step-scale dynamics failure, not
evidence about 8-bit SR-SGD capacity. Round 4 re-probes the identical int8 SR
shadow mechanism at E in {0.25, 0.5, 1.0} and adds a bit-exact-BP oracle arm
per E as a CEILING CONTROL (never gate-eligible): it separates
accumulation-precision capacity from feedback quality for the residual
0.574 -> 0.604 gap.

Spec + gates frozen in data/report/feedflip/PREREG_round4_ishad_gentle.md,
committed BEFORE any full run. Benchmark identical to rounds 1-3; batch stream
seed+900, SR noise stream seed+1300 (both unchanged).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

from experiments.feedflip_bitflip import (
    BATCH,
    GATE_ACC,
    N_TRAIN,
    SEEDS,
    STEPS,
    TernaryNet,
    bp_deltas,
    make_task,
)
from experiments.feedflip_feedback_budget import (
    TERNARY_BITS,
    fa_deltas,
    make_B_fa,
    transposed_ternary_B,
)
from experiments.feedflip_magnitude_votes import (
    EMA,
    ISHAD_CLIP,
    REF_BP_SHADOW,
    REF_SIGN_ONLY,
    ST_K,
    ternarize_from_shadow,
)
from feedflipnets.core.deep_mlp import output_error

ES = {"e025": 0.25, "e05": 0.5, "e10": 1.0}
FBS = ["bp", "st512", "farand"]
ARMS = [f"ishad_{e}_{fb}" for e in ES for fb in FBS]
GATE_ELIGIBLE_FBS = ("st512", "farand")  # bp is an oracle ceiling control
CEIL_LOW, CEIL_HIGH = 0.59, 0.604  # frozen P4 ceiling-rule boundaries
STATE_BITS = 8.0


def run_arm4(arm: str, seed: int, steps: int = STEPS) -> Dict:
    _, e_key, fb = arm.split("_")
    E = ES[e_key]
    Xtr, ytr, Xte, yte = make_task()
    rng = np.random.default_rng(seed + 900)  # batch stream: identical rounds 1-4
    rng_mech = np.random.default_rng(seed + 1300)  # SR noise stream (round-3 offset)
    net = TernaryNet(seed + 100)
    L = len(net.W)
    n_weights = float(sum(w.size for w in net.W))

    if fb == "st512":
        B = transposed_ternary_B(net)
        fb_comm = TERNARY_BITS / ST_K
    elif fb == "farand":
        B = make_B_fa(seed)
        fb_comm = 0.0
    else:  # bp oracle: no feedback matrix; deltas are exact BP
        B = None
        fb_comm = None

    shadow = [(8 * w).astype(np.int8) for w in net.W]
    s_ema = [0.0] * L
    p_num = p_den = p_num_late = p_den_late = 0.0
    late_from = int(0.9 * steps)
    dead_step = None
    churn_flips = 0.0
    t0 = time.perf_counter()

    for step in range(steps):
        if fb == "st512" and step % ST_K == 0:
            B = transposed_ternary_B(net)
        idx = rng.integers(0, N_TRAIN, size=BATCH)
        Xb, yb = Xtr[idx], ytr[idx]
        inputs, derivs, logits = net.forward(Xb)
        err = output_error(logits, yb)
        deltas_bp = bp_deltas(net, inputs, derivs, err)
        deltas = deltas_bp if fb == "bp" else fa_deltas(derivs, err, B)

        g_max = 0.0
        for i in range(L):
            g = inputs[i].T @ deltas[i] / BATCH
            mag = np.abs(g)
            m_mean = float(mag.mean())
            g_max = max(g_max, m_mean)
            s_ema[i] = m_mean if step == 0 else EMA * s_ema[i] + (1 - EMA) * m_mean
            s_safe = max(s_ema[i], 1e-12)
            if fb != "bp" and i < L - 1:  # sign-match metric (trivially 1 for bp)
                g_bp = inputs[i].T @ deltas_bp[i] / BATCH
                m = (g != 0) & (g_bp != 0)
                hits = float((np.sign(g)[m] == np.sign(g_bp)[m]).sum())
                p_num += hits
                p_den += float(m.sum())
                if step >= late_from:
                    p_num_late += hits
                    p_den_late += float(m.sum())

            x = -E * g / s_safe
            f = np.floor(x)
            sr = f + (rng_mech.random(g.shape) < (x - f))
            shadow[i] = np.clip(
                shadow[i].astype(np.int32) + sr.astype(np.int32),
                -ISHAD_CLIP,
                ISHAD_CLIP,
            ).astype(np.int8)
            w_new = ternarize_from_shadow(shadow[i])
            churn_flips += float((w_new != net.W[i]).sum())
            net.W[i] = w_new

        if dead_step is None and g_max == 0.0:
            dead_step = step

    ms = (time.perf_counter() - t0) / steps * 1000
    acc = net.acc(Xte, yte)
    return {
        "arm": arm,
        "E": E,
        "fb": fb,
        "gate_eligible": fb in GATE_ELIGIBLE_FBS,
        "seed": seed,
        "acc": acc,
        "gap_closure": round((acc - REF_SIGN_ONLY) / (REF_BP_SHADOW - REF_SIGN_ONLY), 3),
        "ms_per_step": ms,
        "state_bits_per_w": STATE_BITS,
        "fb_comm_bits_per_w_step": None if fb_comm is None else round(fb_comm, 5),
        "sign_match_p": (p_num / p_den) if p_den else None,
        "sign_match_p_late": (p_num_late / p_den_late) if p_den_late else None,
        "dead_step": dead_step,
        "churn_per_w_per_kstep": round(churn_flips / n_weights / steps * 1000, 4),
        "final_sat_frac": round(
            float(np.mean([(np.abs(s) == ISHAD_CLIP).mean() for s in shadow])), 4
        ),
        "final_nonzero_w": round(float(np.mean([(w != 0).mean() for w in net.W])), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default="data/report/feedflip/feedflip_round4.json")
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            r = run_arm4(arm, seed, steps=args.steps)
            rows.append(r)
            print(
                f"{arm:16s} seed={seed} acc={r['acc']:.3f} GC={r['gap_closure']} "
                f"dead={r['dead_step']} churn/kstep={r['churn_per_w_per_kstep']} "
                f"sat={r['final_sat_frac']} nz={r['final_nonzero_w']} "
                f"p_late={None if r['sign_match_p_late'] is None else round(r['sign_match_p_late'], 3)} "
                f"ms={r['ms_per_step']:.2f}",
                flush=True,
            )

    summary = {}
    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm]
        p_late = [r["sign_match_p_late"] for r in sub if r["sign_match_p_late"] is not None]
        summary[arm] = {
            "gate_eligible": sub[0]["gate_eligible"],
            "mean_acc": float(np.mean([r["acc"] for r in sub])),
            "std_acc": float(np.std([r["acc"] for r in sub])),
            "mean_gap_closure": float(np.mean([r["gap_closure"] for r in sub])),
            "mean_sign_match_p_late": float(np.mean(p_late)) if p_late else None,
            "n_dead": sum(1 for r in sub if r["dead_step"] is not None),
            "mean_churn_per_w_per_kstep": float(np.mean([r["churn_per_w_per_kstep"] for r in sub])),
        }

    eligible = {a: v for a, v in summary.items() if v["gate_eligible"]}
    bp_arms = {a: v for a, v in summary.items() if not v["gate_eligible"]}
    # P4 validity guard (frozen pre-run): a bp oracle arm is VALID only if no
    # seed trips the dead-detector. Calibration probes showed bp-fed shadows
    # can die via ternary path-collapse (dead_step 33-124 at E>=0.5), which is
    # a dynamics failure, not evidence about accumulation precision.
    valid_bp = {a: v for a, v in bp_arms.items() if v["n_dead"] == 0}
    ceiling = max(v["mean_acc"] for v in valid_bp.values()) if valid_bp else None
    if not bp_arms:
        ceiling_zone = "not_run"
    elif ceiling is None:
        ceiling_zone = "oracle_invalid_dynamics_failure_no_precision_conclusion"
    elif ceiling < CEIL_LOW:
        ceiling_zone = "precision_binding_frontier_confirmed"
    elif ceiling >= CEIL_HIGH:
        ceiling_zone = "precision_not_binding_frontier_overturned"
    else:
        ceiling_zone = "boundary_unresolved"

    gate = {
        "gate_acc": GATE_ACC,
        "G_R4": any(v["mean_acc"] >= GATE_ACC for v in eligible.values()),
        "G_R4_strict": any(v["mean_acc"] >= REF_BP_SHADOW for v in eligible.values()),
        "bp_ceiling_C": ceiling,
        "P4_ceiling_zone": ceiling_zone,
        "go": any(v["mean_acc"] >= GATE_ACC for v in eligible.values()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary, "gate": gate}, indent=2) + "\n")
    print(json.dumps({"summary": summary, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
