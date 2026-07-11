"""FeedFlip round-5: precision oracle (pre-registered).

Round 4 left P5 open: the bp-transport ceiling arm entered a zero-gradient
absorbing state (validity guard fired), so "is the residual 0.572 -> 0.604 gap
a *precision* effect?" could not be answered. Round 5 isolates the precision
axis with oracles matched to the anchor's exact dynamics:

  i8clone_gG -- int8 SR clone of run_bp_shadow: identical loop, identical
                backward-through-ternary path, identical per-step adaptive
                alpha; ONLY the shadow storage moves float32 -> int8 grid
                (per-layer step q_l = mean|Wf_l(init)| / G, G in {8,16,32}).
  bps_eE     -- liveness-protected fallback oracle: round-4 ishadE update
                (E in {0.25, 0.5}) with backward through SHADOW-derived
                effective weights (zeroed ternary cells cannot kill the
                gradient path).
  st512s_e025 -- gate-eligible: round-4 e025 update, stale chained feedback
                refreshed from shadow-derived eff^T every 512 steps
                (8 bits/w per refresh = 0.015625 bits/w/step).

Spec + gates frozen in data/report/feedflip/PREREG_round5_precision_oracle.md,
committed BEFORE any full run. Benchmark identical to rounds 1-4; batch RNG
stream unchanged (seed+900), mechanism noise stream unchanged (seed+1300).
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
    LR_SHADOW,
    N_TRAIN,
    SEEDS,
    STEPS,
    TernaryNet,
    bp_deltas,
    make_task,
)
from experiments.feedflip_feedback_budget import fa_deltas
from experiments.feedflip_magnitude_votes import EMA, ISHAD_CLIP, ternarize_from_shadow
from feedflipnets.core.deep_mlp import output_error

Array = np.ndarray

ST_K = 512  # frozen feedback refresh (rounds 2-4)
SHADOW_BITS_PER_REFRESH = 8.0  # st512s ships int8 shadow values, not ternary

G_GRID = [8, 16, 32]  # all run; no post-hoc selection
E_GRID = [0.25, 0.5]

ARMS = [f"i8clone_g{g}" for g in G_GRID] + ["bps_e025", "bps_e05", "st512s_e025"]
ORACLE_ARMS = [a for a in ARMS if not a.startswith("st512s")]

# Frozen references (PREREG round 5)
REF_BP_SHADOW = 0.604  # float32 anchor, rounds 1-4
REF_BEST_8BIT = 0.572  # round-4 e025_st512 plateau
CEIL_HIGH = 0.604  # C5 >= HIGH: precision NOT binding at 8 bits/w
CEIL_LOW = 0.59  # C5 <  LOW : precision binding CONFIRMED


def _sr(x: Array, rng: np.random.Generator) -> Array:
    f = np.floor(x)
    return f + (rng.random(x.shape) < (x - f))


def run_i8clone(G: int, seed: int, steps: int = STEPS) -> Dict:
    """Exact int8 SR-SGD clone of run_bp_shadow; only shadow precision moves."""
    Xtr, ytr, Xte, yte = make_task()
    rng = np.random.default_rng(seed + 900)  # batch stream: identical to rounds 1-4
    rng_mech = np.random.default_rng(seed + 1300)  # SR stream
    rng_w = np.random.default_rng(seed + 100)  # identical shadow init to bp_shadow
    Wf = [
        rng_w.standard_normal((DIMS[i], DIMS[i + 1])) / np.sqrt(DIMS[i])
        for i in range(len(DIMS) - 1)
    ]
    q = [float(np.abs(wf).mean()) / G for wf in Wf]  # per-layer grid, frozen at init
    n = [
        np.clip(np.rint(wf / qi), -ISHAD_CLIP, ISHAD_CLIP).astype(np.int8) for wf, qi in zip(Wf, q)
    ]
    net = TernaryNet(seed + 100)
    L = len(net.W)
    alpha0 = list(net.alpha)
    n_dead = 0
    dead_step: Optional[int] = None
    t0 = time.perf_counter()

    for step in range(steps):
        for i in range(L):
            wf = n[i].astype(np.float64) * q[i]
            a = float(np.abs(wf).mean())
            net.alpha[i] = a
            net.W[i] = ternarize_from_shadow(wf)  # thr = 0.7*mean|wf|, as anchor
        idx = rng.integers(0, N_TRAIN, size=BATCH)
        Xb, yb = Xtr[idx], ytr[idx]
        inputs, derivs, logits = net.forward(Xb)
        err = output_error(logits, yb)
        deltas = bp_deltas(net, inputs, derivs, err)  # through ternary, as anchor
        tot = 0.0
        for i in range(L):
            g = inputs[i].T @ deltas[i] / BATCH
            tot += float(np.abs(g).mean())
            x = -LR_SHADOW * g / q[i]
            n[i] = np.clip(
                n[i].astype(np.int32) + _sr(x, rng_mech).astype(np.int32),
                -ISHAD_CLIP,
                ISHAD_CLIP,
            ).astype(np.int8)
        if tot == 0.0:
            n_dead += 1
            if dead_step is None:
                dead_step = step

    ms = (time.perf_counter() - t0) / steps * 1000
    acc = net.acc(Xte, yte)
    sat = float(np.mean([float((np.abs(m) == ISHAD_CLIP).mean()) for m in n]))
    alpha_ratio = float(np.mean([net.alpha[i] / alpha0[i] for i in range(L)]))
    return {
        "arm": f"i8clone_g{G}",
        "role": "oracle",
        "seed": seed,
        "acc": acc,
        "gap_closure": round((acc - REF_BEST_8BIT) / (REF_BP_SHADOW - REF_BEST_8BIT), 3),
        "ms_per_step": ms,
        "state_bits_per_w": 8.0,
        "fb_comm_bits_per_w_step": None,  # oracle: full BP transport
        "n_dead": n_dead,
        "dead_step": dead_step,
        "sat_frac": round(sat, 4),
        "alpha_ratio": round(alpha_ratio, 3),
    }


def _shadow_eff(alpha0: List[float], shadow: List[Array], j: int) -> Array:
    return alpha0[j] * shadow[j].astype(np.float64) / 8.0


def run_shadow_arm(arm: str, seed: int, steps: int = STEPS) -> Dict:
    """bps_eE (oracle, backward-through-shadow) and st512s_e025 (gate arm).

    Update rule is bit-identical to round-4 ishadE; only the delta path
    differs: bps uses exact chained backward through shadow-derived eff,
    st512s uses stale chained B refreshed from shadow-derived eff^T.
    """
    mech, tag = arm.split("_", 1)
    E = {"e025": 0.25, "e05": 0.5}[tag]
    Xtr, ytr, Xte, yte = make_task()
    rng = np.random.default_rng(seed + 900)
    rng_mech = np.random.default_rng(seed + 1300)
    net = TernaryNet(seed + 100)
    L = len(net.W)
    alpha0 = list(net.alpha)  # fixed at init, as rounds 3-4
    shadow = [(8 * w).astype(np.int8) for w in net.W]
    s_ema = [0.0] * L
    B: List[Optional[Array]] = [None] * L
    fb_comm = SHADOW_BITS_PER_REFRESH / ST_K if mech == "st512s" else None
    n_dead = 0
    dead_step: Optional[int] = None
    p_num = p_den = p_num_late = p_den_late = 0.0
    late_from = int(0.9 * steps)
    t0 = time.perf_counter()

    for step in range(steps):
        if mech == "st512s" and step % ST_K == 0:
            B = [None] + [_shadow_eff(alpha0, shadow, j).T.copy() for j in range(1, L)]
        idx = rng.integers(0, N_TRAIN, size=BATCH)
        Xb, yb = Xtr[idx], ytr[idx]
        inputs, derivs, logits = net.forward(Xb)
        err = output_error(logits, yb)
        deltas_bp = bp_deltas(net, inputs, derivs, err)
        if mech == "bps":
            deltas: List[Optional[Array]] = [None] * L
            deltas[L - 1] = err
            d = err
            for i in reversed(range(L - 1)):
                d = (d @ _shadow_eff(alpha0, shadow, i + 1).T) * derivs[i]
                deltas[i] = d
        else:
            deltas = fa_deltas(derivs, err, B)

        tot = 0.0
        for i in range(L):
            g = inputs[i].T @ deltas[i] / BATCH
            mag = np.abs(g)
            tot += float(mag.mean())
            m_mean = float(mag.mean())
            s_ema[i] = m_mean if step == 0 else EMA * s_ema[i] + (1 - EMA) * m_mean
            s_safe = max(s_ema[i], 1e-12)
            if mech == "st512s" and i < L - 1:  # barrier metric, hidden layers
                g_bp = inputs[i].T @ deltas_bp[i] / BATCH
                msk = (g != 0) & (g_bp != 0)
                hits = float((np.sign(g)[msk] == np.sign(g_bp)[msk]).sum())
                p_num += hits
                p_den += float(msk.sum())
                if step >= late_from:
                    p_num_late += hits
                    p_den_late += float(msk.sum())
            x = -E * g / s_safe
            shadow[i] = np.clip(
                shadow[i].astype(np.int32) + _sr(x, rng_mech).astype(np.int32),
                -ISHAD_CLIP,
                ISHAD_CLIP,
            ).astype(np.int8)
            net.W[i] = ternarize_from_shadow(shadow[i])
        if tot == 0.0:
            n_dead += 1
            if dead_step is None:
                dead_step = step

    ms = (time.perf_counter() - t0) / steps * 1000
    acc = net.acc(Xte, yte)
    sat = float(np.mean([float((np.abs(s) == ISHAD_CLIP).mean()) for s in shadow]))
    return {
        "arm": arm,
        "role": "oracle" if mech == "bps" else "gate",
        "seed": seed,
        "acc": acc,
        "gap_closure": round((acc - REF_BEST_8BIT) / (REF_BP_SHADOW - REF_BEST_8BIT), 3),
        "ms_per_step": ms,
        "state_bits_per_w": 8.0,
        "fb_comm_bits_per_w_step": fb_comm,
        "n_dead": n_dead,
        "dead_step": dead_step,
        "sat_frac": round(sat, 4),
        "sign_match_p": (p_num / p_den) if p_den else None,
        "sign_match_p_late": (p_num_late / p_den_late) if p_den_late else None,
    }


def run_arm5(arm: str, seed: int, steps: int = STEPS) -> Dict:
    if arm.startswith("i8clone_g"):
        return run_i8clone(int(arm.rsplit("g", 1)[1]), seed, steps=steps)
    return run_shadow_arm(arm, seed, steps=steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default="data/report/feedflip/feedflip_round5.json")
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            r = run_arm5(arm, seed, steps=args.steps)
            rows.append(r)
            print(
                f"{arm:14s} seed={seed} acc={r['acc']:.3f} GC={r['gap_closure']} "
                f"dead={r['n_dead']} sat={r['sat_frac']} ms/step={r['ms_per_step']:.2f}",
                flush=True,
            )

    by_arm: Dict[str, List[float]] = {}
    dead_by_arm: Dict[str, int] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r["acc"])
        dead_by_arm[r["arm"]] = dead_by_arm.get(r["arm"], 0) + r["n_dead"]

    summary: Dict[str, object] = {}
    valid_oracle_means = {}
    for arm, accs in by_arm.items():
        mean = float(np.mean(accs))
        valid = dead_by_arm[arm] == 0
        summary[arm] = {
            "mean_acc": round(mean, 4),
            "accs": [round(a, 4) for a in accs],
            "valid": valid,
        }
        if arm in ORACLE_ARMS and valid:
            valid_oracle_means[arm] = mean

    if valid_oracle_means:
        c5_arm = max(valid_oracle_means, key=valid_oracle_means.get)  # type: ignore[arg-type]
        c5 = valid_oracle_means[c5_arm]
        if c5 >= CEIL_HIGH:
            verdict = "precision NOT binding at 8 bits/w (P5: gap attributable elsewhere)"
        elif c5 < CEIL_LOW:
            verdict = "precision binding CONFIRMED (8-bit state caps below anchor)"
        else:
            verdict = "unresolved band [0.59, 0.604): partial attribution"
        summary["P5"] = {"C5": round(c5, 4), "arm": c5_arm, "verdict": verdict}
    else:
        summary["P5"] = {"C5": None, "arm": None, "verdict": "GUARD FIRED: no valid oracle"}

    gate = summary.get("st512s_e025", {})
    if isinstance(gate, dict) and "mean_acc" in gate:
        summary["gate_R5"] = {
            "go": bool(gate["mean_acc"] >= GATE_ACC),
            "strict": bool(gate["mean_acc"] >= REF_BP_SHADOW),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
