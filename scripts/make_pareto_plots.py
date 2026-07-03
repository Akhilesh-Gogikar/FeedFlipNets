#!/usr/bin/env python3
"""Build Pareto-style plots of accuracy vs sparsity, size ∝ throughput.

Scans given run directories (or all immediate children of runs/ by default),
extracts:
- Accuracy (or R^2 for regression) from test metrics
- Throughput (samples/sec) from test timings
- Sparsity (ternary_zero_ratio) from test metrics or checkpoint

Writes a scatter plot to data/report/plots/pareto_scatter.png.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _test_accuracy_or_r2(run: pathlib.Path) -> Optional[float]:
    path = run / "metrics_test.jsonl"
    if not path.exists():
        return None
    acc: Optional[float] = None
    r2: Optional[float] = None
    # Take the last epoch entry
    with path.open("r") as fh:
        for line in fh:
            rec = json.loads(line)
            acc = float(rec.get("accuracy", acc)) if "accuracy" in rec else acc
            r2 = float(rec.get("r2", r2)) if "r2" in rec else r2
    return r2 if r2 is not None else acc


def _test_sparsity(run: pathlib.Path) -> Optional[float]:
    path = run / "metrics_test.jsonl"
    if not path.exists():
        return None
    val: Optional[float] = None
    with path.open("r") as fh:
        for line in fh:
            rec = json.loads(line)
            if "ternary_zero_ratio" in rec:
                val = float(rec["ternary_zero_ratio"])  # overwrite until last
    return val


def _test_throughput(run: pathlib.Path) -> Optional[float]:
    timing_path = run / "timing.json"
    test_metrics_path = run / "metrics_test.jsonl"
    if not timing_path.exists() or not test_metrics_path.exists():
        return None
    try:
        timing = json.loads(timing_path.read_text())
        total = float(timing.get("test", {}).get("total_sec", 0.0))
        test_epochs = 0
        samples_per_epoch = None
        with test_metrics_path.open("r") as fh:
            for line in fh:
                test_epochs += 1
                if samples_per_epoch is None:
                    rec = json.loads(line)
                    samples_per_epoch = int(rec.get("sample_count", 0))
        if not test_epochs or not samples_per_epoch or total <= 0:
            return None
        total_samples = test_epochs * samples_per_epoch
        return float(total_samples) / total
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="Run directories to include")
    ap.add_argument("--out", default="data/report/plots/pareto_scatter.png")
    args = ap.parse_args()

    runs: List[pathlib.Path]
    if args.runs:
        runs = [pathlib.Path(p) for p in args.runs]
    else:
        base = pathlib.Path("runs")
        runs = [p for p in base.iterdir() if p.is_dir()]

    xs: List[float] = []  # sparsity (zero ratio)
    ys: List[float] = []  # accuracy / R2
    sizes: List[float] = []  # marker size ∝ throughput
    labels: List[str] = []

    for run in runs:
        thr = _test_throughput(run)
        acc = _test_accuracy_or_r2(run)
        sp = _test_sparsity(run)
        if thr is None or acc is None or sp is None:
            continue
        xs.append(sp)
        ys.append(acc)
        # scale marker area so that typical throughputs are visible
        sizes.append(max(20.0, min(200.0, float(thr) * 0.02)))
        labels.append(run.name)

    if not xs:
        print("no data for pareto plot")
        return

    plt.figure(figsize=(7, 4.5))
    sc = plt.scatter(xs, ys, s=sizes, c=xs, cmap="viridis", edgecolors="k", alpha=0.85)
    cbar = plt.colorbar(sc)
    cbar.set_label("Ternary sparsity (zero ratio)")
    plt.xlabel("Sparsity (zero ratio)")
    plt.ylabel("Accuracy / R$^2$")
    plt.title("Accuracy vs Sparsity (size ∝ throughput)")
    plt.grid(True, ls=":", alpha=0.5)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
