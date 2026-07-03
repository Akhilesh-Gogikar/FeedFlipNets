#!/usr/bin/env python3
"""Lightweight metrics toolkit for FeedFlipNets experiments.

This module provides standalone helpers that mirror the statistics used in the
paper/reporting pipeline:

* ``epochs_to_threshold`` – first epoch reaching a target metric value.
* ``stability_cv`` – coefficient of variation as a stability proxy.
* ``alignment_auc`` – area under an alignment curve.
* ``tost_equivalence`` – two one-sided tests (TOST) for equivalence to a margin.

The functions intentionally avoid external dependencies beyond NumPy so that
they can be imported into notebooks, CLI scripts, or reporting pipelines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_T_CRIT = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_critical(df: int) -> float:
    """Return the 0.975 quantile for a Student-t distribution with ``df`` degrees."""

    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    return _T_CRIT.get(df + 1, 1.96)


def epochs_to_threshold(history: Sequence[float], threshold: float) -> int:
    """Return the first epoch index (1-based) where ``history`` meets ``threshold``.

    If the threshold is never met, returns ``len(history) + 1`` to mirror the
    convention used in the paper.
    """

    for idx, value in enumerate(history, start=1):
        if value >= threshold:
            return idx
    return len(history) + 1


def stability_cv(values: Sequence[float]) -> float:
    """Coefficient of variation (standard deviation divided by mean)."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return float("inf")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return std / mean


def alignment_auc(alignment: Sequence[float], *, steps: Sequence[float] | None = None) -> float:
    """Area under an alignment curve using trapezoidal integration.

    Args:
        alignment: Sequence of alignment values (e.g., per epoch cosine similarities).
        steps: Optional x-axis locations. If omitted, assumes unit spacing.
    """

    y = np.asarray(alignment, dtype=np.float64)
    if y.size == 0:
        return 0.0
    if steps is None:
        x = np.arange(y.size, dtype=np.float64)
    else:
        x = np.asarray(steps, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("steps must match alignment shape")
    return float(np.trapz(y, x))


@dataclass
class TOSTResult:
    equivalent: bool
    t_lower: float
    t_upper: float
    t_crit: float
    df: int


def tost_equivalence(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    margin: float,
) -> TOSTResult:
    """Perform a simple two one-sided test (TOST) for equivalence.

    Args:
        sample_a, sample_b: Observations from the two conditions.
        margin: Maximum allowed absolute difference (in the metric units).

    Returns:
        ``TOSTResult`` indicating whether the difference is within ``margin``.

    Notes:
        This implementation uses the pooled-standard-error formulation with a
        t-critical lookup (df = n1 + n2 - 2). For small sample sizes the lookup
        matches the CI calculations used throughout the reporting scripts.
    """

    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("TOST requires at least two observations per sample.")

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    se = math.sqrt(var_a / a.size + var_b / b.size)
    if se < 1e-12:
        # Identical runs imply equivalence if the difference is within margin.
        delta = mean_a - mean_b
        equivalent = abs(delta) <= margin
        return TOSTResult(
            equivalent, float("inf"), float("-inf"), float("inf"), a.size + b.size - 2
        )

    delta = mean_a - mean_b
    df = a.size + b.size - 2
    tcrit = _t_critical(df)
    t_lower = (delta + margin) / se
    t_upper = (delta - margin) / se
    equivalent = (t_lower > tcrit) and (t_upper < -tcrit)
    return TOSTResult(
        equivalent=bool(equivalent), t_lower=t_lower, t_upper=t_upper, t_crit=tcrit, df=df
    )


__all__ = [
    "TOSTResult",
    "alignment_auc",
    "epochs_to_threshold",
    "stability_cv",
    "tost_equivalence",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick demo for metrics toolkit.")
    parser.add_argument("--margin", type=float, default=0.01, help="TOST equivalence margin.")
    args = parser.parse_args()

    sample_a = [0.89, 0.90, 0.905, 0.892]
    sample_b = [0.888, 0.891, 0.893, 0.897]
    tost = tost_equivalence(sample_a, sample_b, margin=args.margin)
    print(
        f"TOST equivalent: {tost.equivalent} (df={tost.df}, t_lower={tost.t_lower:.3f}, "
        f"t_upper={tost.t_upper:.3f}, t_crit={tost.t_crit:.3f})"
    )
