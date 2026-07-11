"""M2b bits-per-char gate: does ①+② full multi-block LM training close the bpc gap vs tuned BP?

CFG corpus is the controlled primary. Conditions:
  BP        : torch autograd (tuned baseline)
  FIXED_DFA : per-block fixed-random broadcast, A ignored, dW_O/dW_2 exact
  ONE_TWO   : ①+② — per-block A-routed value-exact grads + ② transport-free R_O adaptation

PRE-REGISTERED as a NO-GO: the gate (①+② − BP ≤ 0.15 bpc) is expected to FAIL. Do not tune to pass.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)  # perf: tiny matmuls → BLAS threads cost ~500× (prototype finding)
EPS = 1e-5

from feedflipnets.core.lm import NumpyLM  # noqa: E402
from feedflipnets.data.char_lm import (  # noqa: E402
    bigram_floor,
    build_cfg_corpus,
    build_english_corpus,
    get_batch,
    prep,
)

CONDITIONS = ["BP", "FIXED_DFA", "ONE_TWO"]


# ---- tuned-BP torch baseline (autograd; the reference the gate is measured against) ----
class _Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=EPS, elementwise_affine=False)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, eps=EPS, elementwise_affine=False)
        self.f1 = nn.Linear(d, h, bias=False)
        self.f2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        T, d = x.size(1), x.size(2)
        y = self.ln1(x)
        att = (self.q(y) @ self.k(y).transpose(-2, -1)) / math.sqrt(d)
        att = att.masked_fill(torch.triu(torch.ones(T, T), 1).bool(), float("-inf"))
        x = x + self.o(F.softmax(att, -1) @ self.v(y))
        return x + self.f2(F.relu(self.f1(self.ln2(x))))


class _LM(nn.Module):
    def __init__(self, V, d, h, N, T):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([_Block(d, h) for _ in range(N)])
        self.lnf = nn.LayerNorm(d, eps=EPS, elementwise_affine=False)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, idx):
        T = idx.size(1)
        x = self.emb(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def _train_bp(train, val, V, d, h, N, T, steps, lr, bs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _LM(V, d, h, N, T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    def tb(src):
        xb, yb = get_batch(src, bs, T, rng)
        return torch.tensor(xb), torch.tensor(yb)

    def ev():
        model.eval()
        tot = 0.0
        c = 0
        with torch.no_grad():
            for _ in range(8):
                x, y = tb(val)
                loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
                tot += loss.item() * y.numel()
                c += y.numel()
        model.train()
        return tot / c / math.log(2)

    t0 = time.time()
    best = 9.0
    per_step = None
    for step in range(steps):
        s1 = time.time()
        x, y = tb(train)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 5:
            per_step = time.time() - s1
        if step % max(1, steps // 6) == 0:
            best = min(best, ev())
    final = ev()
    best = min(best, final)
    return final, best, time.time() - t0, per_step


def _train_np(mode, train, val, V, d, h, N, T, steps, lr, bs, seed, adapt_cfg):
    m = NumpyLM(V, d, h, N, T, seed, mode)
    m.set_opt(lr)
    t0 = time.time()
    best = 9.0
    per_step = None
    traj = []
    for step in range(steps):
        s1 = time.time()
        xb, yb = get_batch(train, bs, T, m.rng)
        m.train_step(xb, yb, adapt_cfg if mode == "ONE_TWO" else None)
        if step == 5:
            per_step = time.time() - s1
        if step % max(1, steps // 6) == 0:
            b = m.eval_bpc(val)
            best = min(best, b)
            traj.append(round(b, 3))
    final = m.eval_bpc(val)
    best = min(best, final)
    return final, best, time.time() - t0, per_step, traj


_CORPUS_CACHE = {}


def _corpus(name):
    if name not in _CORPUS_CACHE:
        text = build_english_corpus() if name == "english" else build_cfg_corpus(0)
        train, val, V = prep(text)
        _CORPUS_CACHE[name] = (train, val, V, bigram_floor(train, val, V))
    return _CORPUS_CACHE[name]


def run_condition(condition, d, h, N, T, bs, steps, lr, rho, Ksamp, lrR, seed, corpus="cfg"):
    """Run ONE (condition, seed). Returns {condition, seed, bpc, wall, per_step, traj}."""
    train, val, V, _floor = _corpus(corpus)
    # Pre-registration note: the GO/NO-GO gate metric ("bpc") is the FINAL-checkpoint
    # evaluation, held out from checkpoint selection; min-over-val ("bpc_best_val") is
    # kept only as a selection/diagnostic value. Reporting min() over eval checkpoints
    # for the gate would be an optimistic-selection bias on the pre-registered metric.
    # Both arms (BP and NP conditions) are treated identically.
    if condition == "BP":
        b, b_best, wt, ps = _train_bp(train, val, V, d, h, N, T, steps, lr, bs, seed)
        traj = []
    else:
        b, b_best, wt, ps, traj = _train_np(
            condition, train, val, V, d, h, N, T, steps, lr, bs, seed, (rho, Ksamp, lrR)
        )
    return {
        "condition": condition,
        "seed": seed,
        "bpc": b,
        "bpc_best_val": b_best,
        "wall": wt,
        "per_step": ps,
        "traj": traj,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--h", type=int, default=192)
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rho", type=float, default=0.02)
    ap.add_argument("--Ksamp", type=int, default=4)
    ap.add_argument("--lrR", type=float, default=0.05)
    ap.add_argument("--corpus", type=str, default="cfg", choices=["cfg", "english"])
    args = ap.parse_args()

    _train, _val, V, floor = _corpus(args.corpus)
    print(f"corpus={args.corpus} vocab={V} bigram_floor={floor:.3f} bpc", flush=True)
    results = {c: [] for c in CONDITIONS}
    persteps = {c: [] for c in CONDITIONS}
    rows = []
    for seed in range(args.seeds):
        for cond in CONDITIONS:
            r = run_condition(
                cond,
                args.d,
                args.h,
                args.N,
                args.T,
                args.bs,
                args.steps,
                args.lr,
                args.rho,
                args.Ksamp,
                args.lrR,
                seed,
                corpus=args.corpus,
            )
            results[cond].append(r["bpc"])
            persteps[cond].append(r["per_step"])
            rows.append(r)
            print(
                f"  seed{seed} {cond:10s} bpc={r['bpc']:.3f} traj={r['traj']}",
                flush=True,
            )

    summary = {}
    for cond in CONDITIONS:
        a = np.array(results[cond])
        summary[cond] = {"bpc_mean": float(a.mean()), "bpc_std": float(a.std())}
        print(f"{cond:10s} bpc mean={a.mean():.3f} std={a.std():.3f}", flush=True)
    gap = np.array(results["ONE_TWO"]) - np.array(results["BP"])
    beats_fixed = float(np.mean(results["ONE_TWO"])) < float(np.mean(results["FIXED_DFA"]))
    # pre-registered gate: ①+② − BP ≤ 0.15 bpc, 95% CI excluding 0.25
    ci_half = 1.96 * gap.std(ddof=1) / math.sqrt(len(gap)) if len(gap) > 1 else float("nan")
    gate_pass = bool(gap.mean() <= 0.15 and (gap.mean() + ci_half) < 0.25)
    verdict = {
        "corpus": args.corpus,
        "vocab": V,
        "bigram_floor": floor,
        "steps": args.steps,
        "seeds": args.seeds,
        "summary": summary,
        "gap_one_two_minus_bp_mean": float(gap.mean()),
        "gap_ci_half": float(ci_half),
        "gap_per_seed": [float(x) for x in gap],
        "one_two_beats_fixed_dfa": beats_fixed,
        "gate_threshold_bpc": 0.15,
        "gate_ci_excludes": 0.25,
        "gate_pass": gate_pass,
        "decision": "GO" if gate_pass else "NO-GO",
        "rows": rows,
    }
    out = Path("data/report/m2b")
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2b_lm_ablation.json").write_text(json.dumps(verdict, indent=2))
    print(
        f"\n①+② − BP gap mean={gap.mean():.3f} bpc  beats_fixed_dfa={beats_fixed}  "
        f"gate={'PASS' if gate_pass else 'FAIL (NO-GO)'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
