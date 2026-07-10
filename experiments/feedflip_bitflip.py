"""FeedFlip bit-flip benchmark + pre-registered sign-barrier attack arms.

Rebuilds the deep-MLP FeedFlip harness referenced by data/report/feedflip/README.md
(the original prototype was never committed to git) and attacks the per-weight sign
barrier with new TRANSPORT-FREE vote mechanisms.

PRE-REGISTRATION (written before any attack arm was run):
  Benchmark: teacher-labeled synthetic classification. Architecture, steps, batch,
    and task difficulty were calibrated on BASELINE arms only, then FROZEN at:
    D_IN=32, C=10, deep MLP [32, 64, 64, 64, 64, 10], 8000 steps, batch 16,
    n_train 16384, n=3 seeds (0,1,2), held-out test 2048, teacher noise 0.15.
  FROZEN BASELINE RESULTS (mean acc, n=3 — data/report/feedflip/baselines.json):
    bp_shadow .604 (32 bits/w) | dfa_k32 .536 | dfa_ortho_k32 .534 (7.6 bits/w)
    | feedflip_bp_k8 .489 | dfa_ortho_k8 .483 | dfa_k8 .473 (5.67 bits/w).
    Sign-match p vs BP vote: 0.70-0.71 across all DFA arms (the barrier).
    No attack arm was executed before this block was frozen.
  Baseline arms (harness reconstruction; task difficulty may be calibrated on these
    ONLY, then frozen): bp_shadow, feedflip_bp K8, feedflip_dfa K8/K32,
    feedflip_dfa+ortho K8/K32.
  Attack arms (all transport-free, all <= 8 state bits/weight):
    conf_gate   -- votes only counted where |grad_est| clears a per-layer EMA noise floor
    k_anneal    -- flip threshold K anneals 8 -> 32 over training (fast early, stable late)
    k_layerwise -- larger K for shallower layers (worse alignment => more vote averaging)
    taught_B    -- (2) perturbation-taught feedback B + orthogonal init + K=32
    combo       -- conf_gate + k_anneal + orthogonal B
  GATE (GO/NO-GO): at least one attack arm reaches mean final test acc >= 0.60
    (>= ~70% of the 0.548 -> 0.619 bp_shadow gap) at <= 8 state bits/weight.
  Also logged: per-weight vote sign-match p vs the BP vote (the paper's barrier metric).

State accounting: ternary W (1.58 bits) + signed vote accumulator in [-K, K]
  => bits/w = 1.58 + log2(2K+1). bp_shadow keeps float32 shadow weights (32 bits/w).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from feedflipnets.core.deep_mlp import output_error, relu, softmax_ce_per_sample
from feedflipnets.core.strategies import _orthogonal

Array = np.ndarray

# FROZEN after baseline-only calibration (see PRE-REGISTRATION above).
# Reproduces the README's qualitative landscape: full-precision BP anchor ~0.60
# vs best <=8-bit transport-free ~0.54 (gap ~= the 0.548 -> 0.619 README gap).
DIMS = [32, 64, 64, 64, 64, 10]
STEPS = 8000
BATCH = 16
SEEDS = [0, 1, 2]
N_TRAIN = 16384
N_TEST = 2048
LR_SHADOW = 0.1
GATE_ACC = 0.60
GATE_BITS = 8.0


# ----------------------------------------------------------------- task
def make_task(seed: int = 7):
    """Teacher-MLP labels + label noise. Difficulty calibrated on bp arms only."""
    rng = np.random.default_rng(seed)
    d, c = DIMS[0], DIMS[-1]
    W1 = rng.standard_normal((d, 48)) / np.sqrt(d)
    W2 = rng.standard_normal((48, c)) / np.sqrt(48)

    def label(X: Array, rng: np.random.Generator) -> Array:
        logits = relu(X @ W1) @ W2
        logits = logits + 0.15 * rng.standard_normal(logits.shape)
        return logits.argmax(axis=1)

    Xtr = rng.standard_normal((N_TRAIN, d))
    Xte = rng.standard_normal((N_TEST, d))
    return Xtr, label(Xtr, rng), Xte, label(Xte, rng)


# ----------------------------------------------------------------- ternary net
class TernaryNet:
    """W in {-1,0,+1} per layer + fixed per-layer float scale alpha (absmean of init)."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.W: List[Array] = []
        self.alpha: List[float] = []
        for i in range(len(DIMS) - 1):
            wf = rng.standard_normal((DIMS[i], DIMS[i + 1])) / np.sqrt(DIMS[i])
            a = float(np.abs(wf).mean())
            self.alpha.append(a)
            t = np.zeros_like(wf, dtype=np.int8)
            t[wf > 0.7 * a] = 1
            t[wf < -0.7 * a] = -1
            self.W.append(t)

    def eff(self, idx: int) -> Array:
        return self.alpha[idx] * self.W[idx].astype(np.float64)

    def forward(self, X: Array):
        h = X
        inputs, derivs = [], []
        L = len(self.W)
        for i in range(L):
            inputs.append(h)
            z = h @ self.eff(i)
            if i < L - 1:
                derivs.append((z > 0).astype(np.float64))
                h = relu(z)
            else:
                return inputs, derivs, z
        raise AssertionError

    def acc(self, X: Array, y: Array) -> float:
        _, _, logits = self.forward(X)
        return float((logits.argmax(axis=1) == y).mean())


def bp_deltas(net: TernaryNet, inputs, derivs, err) -> List[Array]:
    """Exact BP deltas through the ternary effective weights (transport)."""
    L = len(net.W)
    deltas = [None] * L
    deltas[L - 1] = err
    d = err
    for i in reversed(range(L - 1)):
        d = (d @ net.eff(i + 1).T) * derivs[i]
        deltas[i] = d
    return deltas


def dfa_deltas(net: TernaryNet, derivs, err, B: List[Array]) -> List[Array]:
    """Transport-free deltas: last layer exact (local), hidden via error @ B."""
    L = len(net.W)
    deltas = [None] * L
    deltas[L - 1] = err
    for i in range(L - 1):
        deltas[i] = (err @ B[i]) * derivs[i]
    return deltas


def make_B(seed: int, ortho: bool) -> List[Array]:
    rng = np.random.default_rng(seed + 500)
    out = DIMS[-1]
    mats = []
    for hd in DIMS[1:-1]:
        if ortho:
            mats.append(_orthogonal(rng, out, hd).astype(np.float64))
        else:
            mats.append(rng.standard_normal((out, hd)) / np.sqrt(out))
    return mats


def taught_B_update(net, B, inputs, err, Xb, yb, step, rng, rho=0.05, lr_B=0.2, n_samp=4):
    """(2) round-robin: nudge B[idx] so err@B aligns with node-perturbation ghat.

    Forward-only (re-runs layers idx+1..L on perturbed activation); never reads W^T.
    """
    L = len(net.W)
    idx = step % (L - 1)
    # recompute activation feeding layer idx+1
    h = Xb
    for i in range(idx + 1):
        z = h @ net.eff(i)
        h = relu(z)
    base_from = idx + 1
    width = h.shape[1]
    batch = Xb.shape[0]

    def loss_from(h_pert):
        g = h_pert
        for i in range(base_from, L):
            z = g @ net.eff(i)
            g = relu(z) if i < L - 1 else z
        return softmax_ce_per_sample(g, yb)

    ghat = np.zeros((batch, width))
    for _ in range(n_samp):
        xi = rng.standard_normal((batch, width)) * rho
        ghat += ((loss_from(h + xi) - loss_from(h - xi)) / (2.0 * rho**2))[:, None] * xi
    ghat /= n_samp
    projected = err @ B[idx]
    ug = ghat / (np.linalg.norm(ghat, axis=1, keepdims=True) + 1e-12)
    up = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-12)
    B[idx] = B[idx] + lr_B * (err.T @ (ug - up)) / batch


# ----------------------------------------------------------------- arms
def run_arm(arm: str, seed: int, steps: int = STEPS) -> Dict[str, float]:
    Xtr, ytr, Xte, yte = make_task()
    rng = np.random.default_rng(seed + 900)

    if arm == "bp_shadow":
        return run_bp_shadow(seed, steps, Xtr, ytr, Xte, yte, rng)

    net = TernaryNet(seed + 100)
    L = len(net.W)
    K_final = {
        "feedflip_bp_k8": 8,
        "feedflip_dfa_k8": 8,
        "feedflip_dfa_ortho_k8": 8,
        "feedflip_dfa_k32": 32,
        "feedflip_dfa_ortho_k32": 32,
        "conf_gate": 32,
        "k_anneal": 32,
        "k_layerwise": 32,
        "taught_B": 32,
        "combo": 32,
    }[arm]
    use_bp_sign = arm == "feedflip_bp_k8"
    ortho = arm in ("feedflip_dfa_ortho_k8", "feedflip_dfa_ortho_k32", "taught_B", "combo")
    B = None if use_bp_sign else make_B(seed, ortho)
    acc_c = [np.zeros_like(w, dtype=np.int32) for w in net.W]
    noise_floor = [0.0] * L
    p_match_num, p_match_den = 0.0, 0.0
    t0 = time.perf_counter()

    for step in range(steps):
        idx = rng.integers(0, N_TRAIN, size=BATCH)
        Xb, yb = Xtr[idx], ytr[idx]
        inputs, derivs, logits = net.forward(Xb)
        err = output_error(logits, yb)

        deltas_bp = bp_deltas(net, inputs, derivs, err)
        if use_bp_sign:
            deltas = deltas_bp
        else:
            if arm == "taught_B" and step >= 50:
                taught_B_update(net, B, inputs, err, Xb, yb, step, rng)
            deltas = dfa_deltas(net, derivs, err, B)

        if arm == "k_anneal" or arm == "combo":
            K_now = 8 + int((K_final - 8) * step / max(1, steps - 1))
        else:
            K_now = K_final

        for i in range(L):
            g = inputs[i].T @ deltas[i] / BATCH
            vote = -np.sign(g).astype(np.int32)
            if not use_bp_sign:  # barrier metric vs BP vote, hidden layers only
                if i < L - 1:
                    g_bp = inputs[i].T @ deltas_bp[i] / BATCH
                    m = (g != 0) & (g_bp != 0)
                    p_match_num += float((np.sign(g)[m] == np.sign(g_bp)[m]).sum())
                    p_match_den += float(m.sum())
            if arm in ("conf_gate", "combo"):
                mag = np.abs(g)
                noise_floor[i] = 0.99 * noise_floor[i] + 0.01 * float(mag.mean())
                vote = np.where(mag > noise_floor[i], vote, 0)
            K_i = K_now
            if arm == "k_layerwise":
                # shallow layers: worst alignment -> most averaging
                K_i = int(round(K_final - (K_final - 8) * i / max(1, L - 1)))
            acc_c[i] += vote
            flip = np.abs(acc_c[i]) >= K_i
            if flip.any():
                net.W[i] = np.clip(
                    net.W[i] + np.sign(acc_c[i]).astype(np.int8) * flip, -1, 1
                ).astype(np.int8)
                acc_c[i][flip] = 0

    ms = (time.perf_counter() - t0) / steps * 1000
    bits = 1.58 + np.log2(2 * K_final + 1)
    return {
        "arm": arm,
        "seed": seed,
        "acc": net.acc(Xte, yte),
        "ms_per_step": ms,
        "bits_per_w": round(float(bits), 2),
        "sign_match_p": (p_match_num / p_match_den) if p_match_den else None,
    }


def run_bp_shadow(seed, steps, Xtr, ytr, Xte, yte, rng) -> Dict[str, float]:
    """Float32 shadow weights, BP through ternary forward (STE), absmean rescale."""
    rng_w = np.random.default_rng(seed + 100)
    Wf = [
        rng_w.standard_normal((DIMS[i], DIMS[i + 1])) / np.sqrt(DIMS[i])
        for i in range(len(DIMS) - 1)
    ]
    net = TernaryNet(seed + 100)
    t0 = time.perf_counter()
    for _ in range(steps):
        for i, wf in enumerate(Wf):
            a = float(np.abs(wf).mean())
            net.alpha[i] = a
            t = np.zeros_like(wf, dtype=np.int8)
            t[wf > 0.7 * a] = 1
            t[wf < -0.7 * a] = -1
            net.W[i] = t
        idx = rng.integers(0, N_TRAIN, size=BATCH)
        Xb, yb = Xtr[idx], ytr[idx]
        inputs, derivs, logits = net.forward(Xb)
        err = output_error(logits, yb)
        deltas = bp_deltas(net, inputs, derivs, err)
        for i in range(len(Wf)):
            Wf[i] -= LR_SHADOW * (inputs[i].T @ deltas[i] / BATCH)
    ms = (time.perf_counter() - t0) / steps * 1000
    return {
        "arm": "bp_shadow",
        "seed": seed,
        "acc": net.acc(Xte, yte),
        "ms_per_step": ms,
        "bits_per_w": 32.0,
        "sign_match_p": None,
    }


BASELINE_ARMS = [
    "bp_shadow",
    "feedflip_bp_k8",
    "feedflip_dfa_k8",
    "feedflip_dfa_k32",
    "feedflip_dfa_ortho_k8",
    "feedflip_dfa_ortho_k32",
]
ATTACK_ARMS = ["conf_gate", "k_anneal", "k_layerwise", "taught_B", "combo"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=BASELINE_ARMS + ATTACK_ARMS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default="data/report/feedflip/feedflip_bitflip.json")
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            r = run_arm(arm, seed, steps=args.steps)
            rows.append(r)
            print(
                f"{arm:24s} seed={seed} acc={r['acc']:.3f} "
                f"bits/w={r['bits_per_w']} ms={r['ms_per_step']:.2f} "
                f"p={r['sign_match_p'] if r['sign_match_p'] is None else round(r['sign_match_p'], 3)}"
            )
    summary = {}
    for arm in args.arms:
        accs = [r["acc"] for r in rows if r["arm"] == arm]
        ps = [r["sign_match_p"] for r in rows if r["arm"] == arm and r["sign_match_p"]]
        summary[arm] = {
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "bits_per_w": [r["bits_per_w"] for r in rows if r["arm"] == arm][0],
            "mean_sign_match_p": float(np.mean(ps)) if ps else None,
        }
    gate = {
        "gate_acc": GATE_ACC,
        "gate_bits": GATE_BITS,
        "go": any(
            v["mean_acc"] >= GATE_ACC and v["bits_per_w"] <= GATE_BITS
            for a, v in summary.items()
            if a in ATTACK_ARMS
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary, "gate": gate}, indent=2) + "\n")
    print(json.dumps({"summary": summary, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
