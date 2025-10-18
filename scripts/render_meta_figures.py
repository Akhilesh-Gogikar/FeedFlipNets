#!/usr/bin/env python3
"""
Render redesigned meta figures from data/report/meta_convergence_summary.csv:
 - meta_epochs_to_90pct_acc.png (horizontal bars)
 - meta_early_slopes.png (two clean panels: Acc. slope and R^2 slope)

The script is deterministic and uses a consistent style with compact labels.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/report/meta_convergence_summary.csv"
OUT_DIR = ROOT / "data/report/plots"


def _compact_label(run: str) -> str:
    # Example: "20newsgroups-bow-mlp-dfa-per-epoch" -> "20NG DFA (pe)"
    ds = run.split("-")[0]
    if ds.startswith("20news"):
        ds_label = "20NG"
    elif ds.startswith("mnist"):
        ds_label = "MNIST"
    elif ds.startswith("california"):
        ds_label = "Cal. Housing"
    elif ds.startswith("ucr"):
        ds_label = "UCR"
    else:
        ds_label = ds
    pe = "per-epoch" in run
    ps = ("-dfa" in run) and not pe
    sched = " (pe)" if pe else (" (ps)" if ps else "")
    variant = "DFA"
    return f"{ds_label} {variant}{sched}"


def _read_meta() -> (
    Tuple[List[str], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]
):
    runs: List[str] = []
    e90: Dict[str, float] = {}
    acc_slope: Dict[str, float] = {}
    e90r2: Dict[str, float] = {}
    r2_slope: Dict[str, float] = {}
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row["run"].strip()
            runs.append(run)
            try:
                e = float(row.get("epochs_to_90pct_best_acc") or "nan")
            except ValueError:
                e = float("nan")
            try:
                s = float(row.get("early_acc_slope_e3") or "nan")
            except ValueError:
                s = float("nan")
            try:
                e_r2 = float(row.get("epochs_to_90pct_r2") or "nan")
            except ValueError:
                e_r2 = float("nan")
            try:
                s_r2 = float(row.get("early_r2_slope_e3") or "nan")
            except ValueError:
                s_r2 = float("nan")
            e90[run] = e
            acc_slope[run] = s
            e90r2[run] = e_r2
            r2_slope[run] = s_r2
    return runs, e90, acc_slope, e90r2, r2_slope


def _fmt_small(x: float, pos=None) -> str:
    # Format 0.000 -> .000, -0.008 -> -.008
    try:
        s = f"{x:.3f}"
    except Exception:
        return ""
    if s.startswith("-0"):
        return "-." + s.split(".")[1]
    if s.startswith("0"):
        return "." + s.split(".")[1]
    return s


def plot_epochs_to_point(runs: List[str], e90: Dict[str, float]) -> None:
    labels = [_compact_label(r) for r in runs if e90.get(r) == e90.get(r)]
    values = [e90[r] for r in runs if e90.get(r) == e90.get(r)]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.barh(labels, values, color="#5B8FD9")
    ax.set_xlabel("Epochs to 90% of best accuracy")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "meta_epochs_to_90pct_acc.png", dpi=200)
    plt.close(fig)


def plot_early_slopes(
    runs: List[str], acc_slope: Dict[str, float], r2_slope: Dict[str, float]
) -> None:
    # Single-panel, sorted, with value annotations (accuracy only)
    pairs = [(r, acc_slope.get(r, float("nan"))) for r in runs]
    pairs = [p for p in pairs if p[1] == p[1]]  # drop NaNs
    # Sort descending by slope (faster early learning at top)
    pairs.sort(key=lambda x: x[1], reverse=True)
    labels = [_compact_label(r) for r, _ in pairs]
    acc_vals = [v for _, v in pairs]

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bars = ax.barh(labels, acc_vals, color="#69C283")
    ax.set_title("Early accuracy slope (e1→e3)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: _fmt_small(v)))
    ax.grid(axis="x", alpha=0.2)

    # Annotate values on bars
    for bar, v in zip(bars, acc_vals):
        x = bar.get_width()
        ax.text(
            x + (0.002 if x >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            _fmt_small(v),
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
            color="#333333",
        )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "meta_early_slopes.png", dpi=200)
    plt.close(fig)


def main() -> None:
    runs, e90, acc_slope, _, r2_slope = _read_meta()
    # Fix a stable order: by dataset keyword
    order = sorted(
        runs,
        key=lambda r: (
            "20news" not in r,
            "mnist" not in r,
            "california" not in r,
            "ucr" not in r,
            r,
        ),
    )
    plot_epochs_to_point(order, e90)
    plot_early_slopes(order, acc_slope, r2_slope)


if __name__ == "__main__":
    main()
