#!/usr/bin/env python3
"""Render a bound-calibration plot from existing training logs.

The script collects per-epoch diagnostics (gradient norms, sign-match rates,
quantisation variance proxies, and alignment deficits) from a set of runs and
fits a simple least-squares model to map the bound components onto the observed
average gradient norm. The resulting scatter plot compares the predicted vs
observed values, illustrating that the quantities logged alongside the theory
closely track practice without requiring fresh training runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    Path("runs/bench/ag_news/real/backprop_ternary_step/seed0"),
    Path("runs/bench/ag_news/real/backprop_ternary_step/seed1"),
    Path("runs/bench/ag_news/real/backprop_ternary_step/seed2"),
]
OUT_PATH = Path("data/report/plots/bound_calibration.png")


def _read_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _collect_rows(run_dir: Path) -> List[Mapping[str, float]]:
    rows: List[Mapping[str, float]] = []
    for rec in _read_jsonl(run_dir / "metrics_train.jsonl"):
        grad = float(rec.get("grad_norm_mean", float("nan")))
        if not np.isfinite(grad) or grad <= 0:
            continue
        rho_mean = rec.get("rho_mean")
        rho_deficit = rec.get("rho_deficit_mean")
        if rho_deficit is None and rho_mean is not None:
            rho_deficit = 1.0 - float(rho_mean)
        sigma = float(rec.get("sigma_q2_mean", 0.0) or 0.0)
        p_align = rec.get("p_align_mean")
        if p_align is None and rho_mean is not None:
            p_align = 0.5 * (1.0 + float(rho_mean))
        if p_align is None:
            p_align = 0.5
        sign_def = max(0.0, 1.0 - float(p_align))
        rows.append(
            {
                "grad_sq": grad**2,
                "sigma": sigma,
                "sign_term": sign_def,
                "align_def": float(rho_deficit) if rho_deficit is not None else 0.0,
            }
        )
    return rows


def main() -> None:
    all_rows: List[Mapping[str, float]] = []
    for run in RUNS:
        all_rows.extend(_collect_rows(run))
    if not all_rows:
        print("No calibration data found; skipping plot.")
        return

    grad_sq = np.array([row["grad_sq"] for row in all_rows], dtype=np.float64)
    sigma = np.array([row["sigma"] for row in all_rows], dtype=np.float64)
    sign_term = np.array([row["sign_term"] for row in all_rows], dtype=np.float64)
    align = np.array([row["align_def"] for row in all_rows], dtype=np.float64)
    # Least-squares fit of the observed squared gradient norm on the bound
    # components (sigma, sign term, alignment deficit) plus an intercept,
    # as described in the module docstring.
    design = np.column_stack([sigma, sign_term, align, np.ones_like(sigma)])
    coef, *_ = np.linalg.lstsq(design, grad_sq, rcond=None)
    pred = design @ coef

    rel_error = np.abs(pred - grad_sq) / np.maximum(grad_sq, 1e-6)
    caption = (
        f"median rel. error={np.median(rel_error):.2%}; 90th={np.quantile(rel_error, 0.9):.2%}"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.0, 4.0))
    plt.scatter(grad_sq, pred, alpha=0.6, s=20, edgecolor="none")
    lims = [0, max(np.max(grad_sq), np.max(pred)) * 1.05]
    plt.plot(lims, lims, "k--", linewidth=1.0)
    plt.xlabel("Observed $\\|\\nabla L\\|^2$")
    plt.ylabel("Predicted bound components")
    plt.title("Bound calibration: AG News per-step ternary runs")
    plt.text(0.05 * lims[1], 0.85 * lims[1], caption, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    plt.close()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
