"""FeedFlip round-3: magnitude-carrying votes (pre-registered).

Round 2 (commit 2588463) showed the residual FeedFlip gap is MECHANISM-limited:
bit-exact BP vote signs reach only .551 vs bp_shadow .604 — the K-threshold
vote counter discards |g|. Round 3 puts magnitude back at <= 8 state bits/w:

  magq2  -- c += -sign(g) * clip(rint(|g|/s_l), 0, 3)      (2-bit weighted votes)
  stochT -- c += -sign(g) * Bern(min(1, |g|/(T*s_l)))      (unbiased vote rate)
  ishadE -- int8 shadow a += SR(-E*g/s_l); W = ternarize(a) (8-bit SR-SGD)

Feedback is second-order (round-2 finding); each mechanism runs with
  st512  -- stale ternary eff^T transport, k=512 (0.0031 bits/w/step)
  farand -- fixed random chained FA (0 bits, seed-regenerable)

Spec + gates frozen in data/report/feedflip/PREREG_round3_magnitude_votes.md,
committed BEFORE any full run. Benchmark identical to rounds 1-2; batch RNG
stream unchanged (mechanism noise uses a separate stream, seed+1300).
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
from feedflipnets.core.deep_mlp import output_error

Array = np.ndarray

K_VOTE = 32
EMA = 0.99
Q_MAX = 3
ISHAD_CLIP = 127
ST_K = 512  # frozen feedback refresh for the st512 level

MECHS = {"magq2": None, "stoch1": 1.0, "stoch4": 4.0, "ishad2": 2.0, "ishad8": 8.0}
FBS = ["st512", "farand"]
ARMS = [f"{m}_{fb}" for m in MECHS for fb in FBS]

# Frozen references for gap closure (PREREG): round-2 sign-only .557, bp_shadow .604
REF_SIGN_ONLY = 0.557
REF_BP_SHADOW = 0.604

STATE_BITS = {
    "magq2": round(1.58 + float(np.log2(2 * (K_VOTE - 1 + Q_MAX) + 1)), 2),  # 7.69
    "stoch1": round(1.58 + float(np.log2(2 * K_VOTE + 1)), 2),  # 7.6
    "stoch4": round(1.58 + float(np.log2(2 * K_VOTE + 1)), 2),
    "ishad2": 8.0,
    "ishad8": 8.0,
}


def ternarize_from_shadow(a: Array) -> Array:
    thr = 0.7 * float(np.abs(a).mean())
    return ((a > thr).astype(np.int8) - (a < -thr).astype(np.int8)).astype(np.int8)


def run_arm3(arm: str, seed: int, steps: int = STEPS) -> Dict:
    mech, fb = arm.rsplit("_", 1)
    param = MECHS[mech]
    Xtr, ytr, Xte, yte = make_task()
    rng = np.random.default_rng(seed + 900)  # batch stream: identical to rounds 1-2
    rng_mech = np.random.default_rng(seed + 1300)  # separate mechanism-noise stream
    net = TernaryNet(seed + 100)
    L = len(net.W)

    if fb == "st512":
        B = transposed_ternary_B(net)
        fb_comm = TERNARY_BITS / ST_K
    else:
        B = make_B_fa(seed)
        fb_comm = 0.0

    acc_c = [np.zeros_like(w, dtype=np.int32) for w in net.W]
    shadow = [(8 * w).astype(np.int8) for w in net.W] if mech.startswith("ishad") else None
    s_ema = [0.0] * L
    p_num = p_den = p_num_late = p_den_late = 0.0
    late_from = int(0.9 * steps)
    t0 = time.perf_counter()

    for step in range(steps):
        if fb == "st512" and step % ST_K == 0:
            B = transposed_ternary_B(net)
        idx = rng.integers(0, N_TRAIN, size=BATCH)
        Xb, yb = Xtr[idx], ytr[idx]
        inputs, derivs, logits = net.forward(Xb)
        err = output_error(logits, yb)
        deltas_bp = bp_deltas(net, inputs, derivs, err)
        deltas = fa_deltas(derivs, err, B)

        for i in range(L):
            g = inputs[i].T @ deltas[i] / BATCH
            mag = np.abs(g)
            m_mean = float(mag.mean())
            s_ema[i] = m_mean if step == 0 else EMA * s_ema[i] + (1 - EMA) * m_mean
            s_safe = max(s_ema[i], 1e-12)
            if i < L - 1:  # barrier metric vs BP vote, hidden layers only
                g_bp = inputs[i].T @ deltas_bp[i] / BATCH
                m = (g != 0) & (g_bp != 0)
                hits = float((np.sign(g)[m] == np.sign(g_bp)[m]).sum())
                p_num += hits
                p_den += float(m.sum())
                if step >= late_from:
                    p_num_late += hits
                    p_den_late += float(m.sum())

            if mech == "magq2":
                v = -np.sign(g).astype(np.int32) * np.clip(
                    np.rint(mag / s_safe).astype(np.int32), 0, Q_MAX
                )
            elif mech.startswith("stoch"):
                p_vote = np.minimum(mag / (param * s_safe), 1.0)
                v = -np.sign(g).astype(np.int32) * (rng_mech.random(g.shape) < p_vote).astype(
                    np.int32
                )
            else:  # ishadE: int8 stochastic-rounded shadow, no vote accumulator
                x = -param * g / s_safe
                f = np.floor(x)
                sr = f + (rng_mech.random(g.shape) < (x - f))
                shadow[i] = np.clip(
                    shadow[i].astype(np.int32) + sr.astype(np.int32),
                    -ISHAD_CLIP,
                    ISHAD_CLIP,
                ).astype(np.int8)
                net.W[i] = ternarize_from_shadow(shadow[i])
                continue

            acc_c[i] += v
            flip = np.abs(acc_c[i]) >= K_VOTE
            if flip.any():
                net.W[i] = np.clip(
                    net.W[i] + np.sign(acc_c[i]).astype(np.int8) * flip, -1, 1
                ).astype(np.int8)
                acc_c[i][flip] = 0

    ms = (time.perf_counter() - t0) / steps * 1000
    acc = net.acc(Xte, yte)
    return {
        "arm": arm,
        "seed": seed,
        "acc": acc,
        "gap_closure": round((acc - REF_SIGN_ONLY) / (REF_BP_SHADOW - REF_SIGN_ONLY), 3),
        "ms_per_step": ms,
        "state_bits_per_w": STATE_BITS[mech],
        "fb_comm_bits_per_w_step": round(fb_comm, 5),
        "sign_match_p": (p_num / p_den) if p_den else None,
        "sign_match_p_late": (p_num_late / p_den_late) if p_den_late else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default="data/report/feedflip/feedflip_round3.json")
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            r = run_arm3(arm, seed, steps=args.steps)
            rows.append(r)
            print(
                f"{arm:16s} seed={seed} acc={r['acc']:.3f} GC={r['gap_closure']} "
                f"bits={r['state_bits_per_w']} comm={r['fb_comm_bits_per_w_step']} "
                f"p_late={None if r['sign_match_p_late'] is None else round(r['sign_match_p_late'], 3)} "
                f"ms={r['ms_per_step']:.2f}",
                flush=True,
            )

    summary = {}
    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm]
        p_late = [r["sign_match_p_late"] for r in sub if r["sign_match_p_late"] is not None]
        summary[arm] = {
            "mean_acc": float(np.mean([r["acc"] for r in sub])),
            "std_acc": float(np.std([r["acc"] for r in sub])),
            "mean_gap_closure": float(np.mean([r["gap_closure"] for r in sub])),
            "state_bits_per_w": sub[0]["state_bits_per_w"],
            "fb_comm_bits_per_w_step": sub[0]["fb_comm_bits_per_w_step"],
            "mean_sign_match_p_late": float(np.mean(p_late)) if p_late else None,
            "n_seeds_with_sign_metric": len(p_late),
        }

    go = any(v["mean_acc"] >= GATE_ACC for v in summary.values())
    vote_arms = [a for a in summary if a.startswith(("magq2", "stoch"))]
    gate = {
        "gate_acc": GATE_ACC,
        "G_R3": go,
        "G_R3_strict_matched_bp_shadow": any(
            v["mean_acc"] >= REF_BP_SHADOW for v in summary.values()
        ),
        "P4_vote_mechanism_go": any(summary[a]["mean_acc"] >= GATE_ACC for a in vote_arms),
        "go": go,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary, "gate": gate}, indent=2) + "\n")
    print(json.dumps({"summary": summary, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
