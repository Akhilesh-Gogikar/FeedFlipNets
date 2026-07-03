#!/usr/bin/env python3
"""Plot histograms of directional agreement p across epochs.

Reads metrics_train.jsonl files produced by alignment probes and builds
per-variant histograms of p over epochs for a given dataset/mode under
`runs/bench/<dataset>/<mode>/<variant>/seed*/metrics_train.jsonl`.

Output: data/report/plots/p_hist_<dataset>_<mode>.png

Usage:
  python scripts/plot_p_hist.py --dataset ag_news --mode real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


# Prefer aggregated bench layout; fall back to top-level runs when absent
ROOT_BENCH = Path("runs/bench")
ROOT_TOP = Path("runs")
PLOTS = Path("data/report/plots")


def _read_jsonl(path: Path) -> List[Mapping[str, object]]:
    out: List[Mapping[str, object]] = []
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    except Exception:
        return []
    return out


def _extract_p(rec: Mapping[str, object]) -> float | None:
    """Pull p estimate from a training metrics record.

    Priority: p_global_mean -> p_mean -> mean of p_l{i} keys.
    """
    def _as_float(v: object) -> float | None:
        return float(v) if isinstance(v, (int, float)) else None

    v = _as_float(rec.get("p_global_mean"))
    if v is not None:
        return v
    v = _as_float(rec.get("p_mean"))
    if v is not None:
        return v
    # average layer-wise p_l{i}
    vals: List[float] = []
    for k, val in rec.items():
        if isinstance(k, str) and k.startswith("p_l"):
            f = _as_float(val)
            if f is not None:
                vals.append(f)
    if vals:
        return float(np.mean(vals))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", default="real", choices=["real", "offline"])
    ap.add_argument("--out", default=None, help="Override output path")
    args = ap.parse_args()

    # Try bench layout first
    root = ROOT_BENCH / args.dataset / args.mode
    paths: list[Path]
    if root.exists():
        paths = [p for p in root.iterdir() if p.is_dir()]
    else:
        # Fallback: look for runs matching dataset keyword under top-level runs
        cand = [p for p in ROOT_TOP.iterdir() if p.is_dir()]
        paths = [p for p in cand if args.dataset in p.name]
        if not paths:
            print(f"no data root: {root} and no top-level matches for {args.dataset}")
            return

    import matplotlib.pyplot as plt  # type: ignore

    variants = sorted(paths)
    data: Dict[str, List[float]] = {}
    for var in variants:
        ps: List[float] = []
        # bench: seed*/metrics_train.jsonl; top-level: metrics_train.jsonl directly
        seed_paths = list(sorted(var.glob("seed*/metrics_train.jsonl")))
        if not seed_paths:
            seed_paths = [var / "metrics_train.jsonl"] if (var / "metrics_train.jsonl").exists() else []
        for train_path in seed_paths:
            recs = _read_jsonl(train_path)
            for rec in recs:
                p = _extract_p(rec)
                if p is not None:
                    ps.append(float(p))
        if ps:
            data[var.name] = ps

    if not data:
        print("no p data found")
        return

    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.2))
    bins = np.linspace(0.0, 1.0, 21)
    for name, ps in sorted(data.items()):
        plt.hist(ps, bins=bins, alpha=0.4, density=True, label=name, edgecolor="k")
    plt.xlabel("p = Pr[⟨ΔV_DFA, ΔV_BP⟩ > 0]")
    plt.ylabel("Density")
    plt.title(f"Directional agreement p — {args.dataset} ({args.mode})")
    plt.grid(True, ls=":", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = (
        Path(args.out)
        if args.out is not None
        else PLOTS / f"p_hist_{args.dataset}_{args.mode}.png"
    )
    plt.savefig(out, dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
