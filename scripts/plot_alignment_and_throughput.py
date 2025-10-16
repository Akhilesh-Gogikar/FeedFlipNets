#!/usr/bin/env python3
"""Generate alignment curves (rho over epochs) and throughput bar charts.

Reads runs under runs/bench/<dataset>/<mode>/<variant>/seed*/ and emits:
- alignment_curve_{dataset}_{mode}.png: rho_mean vs epoch per variant.
- throughput_bar_{dataset}_{mode}.png: train throughput (samples/s) mean±std per variant.

Usage:
  python scripts/plot_alignment_and_throughput.py --dataset ag_news --mode real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

ROOT = Path("runs/bench")
PLOTS = Path("data/report/plots")


def _read_jsonl(path: Path) -> List[Mapping[str, object]]:
    out: List[Mapping[str, object]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _last_sample_count(train_jsonl: Path) -> float | None:
    if not train_jsonl.exists():
        return None
    last_line = ""
    with train_jsonl.open("r", encoding="utf-8") as h:
        for last_line in h:
            pass
    try:
        rec = json.loads(last_line)
    except Exception:
        return None
    sc = rec.get("sample_count") if isinstance(rec, dict) else None
    return float(sc) if isinstance(sc, (int, float)) else None


def _alignment_curves(
    dataset: str, mode: str
) -> Tuple[List[str], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Return mapping variant -> (epochs, rho_mean_by_epoch) averaged over seeds."""
    root = ROOT / dataset / mode
    if not root.exists():
        return [], {}
    variants = sorted([p.name for p in root.iterdir() if p.is_dir()])
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    chosen: List[str] = []
    for var in variants:
        runs = sorted((root / var).glob("seed*/metrics_train.jsonl"))
        if not runs:
            continue
        # Collect per-seed rho_mean per epoch length
        series: List[np.ndarray] = []
        max_len = 0
        for train_path in runs:
            recs = _read_jsonl(train_path)
            rhos = [float(r.get("rho_mean", np.nan)) for r in recs]
            arr = np.asarray(rhos, dtype=np.float64)
            if np.all(np.isnan(arr)):
                continue
            # replace NaNs with previous value to smooth
            for i in range(len(arr)):
                if np.isnan(arr[i]):
                    arr[i] = arr[i - 1] if i > 0 else np.nan
            series.append(arr)
            max_len = max(max_len, len(arr))
        if not series:
            continue
        # Pad to same length with last valid value
        padded: List[np.ndarray] = []
        for s in series:
            if len(s) < max_len:
                fill = s[-1] if len(s) else np.nan
                s = np.pad(s, (0, max_len - len(s)), constant_values=fill)
            padded.append(s)
        mat = np.vstack(padded)
        mean = np.nanmean(mat, axis=0)
        epochs = np.arange(1, len(mean) + 1)
        result[var] = (epochs, mean)
        chosen.append(var)
    return chosen, result


def _train_throughput(dataset: str, mode: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return variants, mean, 95% CI of train throughput (samples/sec).

    Falls back to 0 error bar when only a single seed is available.
    """
    root = ROOT / dataset / mode
    if not root.exists():
        return [], np.array([]), np.array([])
    variants = sorted([p.name for p in root.iterdir() if p.is_dir()])
    vals: List[float] = []
    labs: List[str] = []
    cis: List[float] = []
    for var in variants:
        throughputs: List[float] = []
        for seed_dir in sorted((root / var).glob("seed*")):
            timing = seed_dir / "timing.json"
            if not timing.exists():
                continue
            data = json.loads(timing.read_text())
            train = data.get("train") if isinstance(data, dict) else None
            total = train.get("total_sec") if isinstance(train, dict) else None
            if not isinstance(total, (int, float)) or total <= 0:
                continue
            sc = _last_sample_count(seed_dir / "metrics_train.jsonl")
            if isinstance(sc, float) and sc > 0:
                throughputs.append(sc / float(total))
        if throughputs:
            labs.append(var)
            mu = float(np.mean(throughputs))
            vals.append(mu)
            if len(throughputs) > 1:
                # Student's t 95% CI
                try:
                    from scipy import stats  # type: ignore

                    tcrit = float(stats.t.ppf(1 - 0.05 / 2.0, df=len(throughputs) - 1))
                except Exception:
                    tcrit = 1.96
                sem = float(np.std(throughputs, ddof=1)) / (len(throughputs) ** 0.5)
                cis.append(tcrit * sem)
            else:
                cis.append(0.0)
    return labs, np.asarray(vals), np.asarray(cis)


def _plot_alignment(dataset: str, mode: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    variants, curves = _alignment_curves(dataset, mode)
    if not variants:
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 3.4))
    for var in sorted(curves.keys()):
        x, y = curves[var]
        plt.plot(x, y, label=var, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("rho (mean across layers)")
    plt.title(f"Alignment (rho) — {dataset} ({mode})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = PLOTS / f"alignment_curve_{dataset}_{mode}.png"
    plt.savefig(out, dpi=200)
    plt.close()


def _plot_throughput(dataset: str, mode: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    labels, means, ci95 = _train_throughput(dataset, mode)
    if not labels:
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    plt.figure(figsize=(6.0, 3.4))
    plt.bar(x, means, yerr=ci95, capsize=3)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Train throughput (samples/s)")
    plt.title(f"Training Throughput — {dataset} ({mode})")
    plt.tight_layout()
    out = PLOTS / f"throughput_bar_{dataset}_{mode}.png"
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", default="real", choices=["real", "offline"])
    args = ap.parse_args()
    _plot_alignment(args.dataset, args.mode)
    _plot_throughput(args.dataset, args.mode)


if __name__ == "__main__":
    main()
