"""Alignment metrics: per-matrix cosine, theta, and depth-slope (spec section 7)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray


def per_matrix_cosine(grads_a: Dict[str, Array], grads_b: Dict[str, Array]) -> Dict[str, float]:
    """Cosine between matching weight-matrix gradients, per matrix (never flattened together)."""
    out: Dict[str, float] = {}
    for key in grads_a:
        if key not in grads_b:
            continue
        a = grads_a[key].ravel()
        b = grads_b[key].ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        out[key] = float(np.dot(a, b) / denom)
    return out


def theta_deg(cosine: float) -> float:
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def min_theta_over_layers(cosines: Dict[str, float]) -> float:
    """Worst-aligned layer = largest angle."""
    return max(theta_deg(c) for c in cosines.values())


def t_quantile_975(df: int) -> float:
    """Two-sided 95% Student-t critical value for small df (hardcoded table; no scipy
    dependency). Falls back to the normal quantile for large df."""
    table = {
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
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if df <= 0:
        return float("inf")
    if df in table:
        return table[df]
    keys = sorted(table)
    for k in keys:
        if df < k:
            return table[k]  # conservative: use the next-smaller-df (larger) quantile
    return 1.96


def depth_slope(depths: List[int], theta_mins: List[float]) -> Tuple[float, float]:
    """OLS slope of theta_min vs L (degrees per layer) with a 95% CI half-width
    (Student-t with n-2 df, appropriate for the small n used here)."""
    x = np.asarray(depths, dtype=np.float64)
    yv = np.asarray(theta_mins, dtype=np.float64)
    n = len(x)
    xm, ym = x.mean(), yv.mean()
    sxx = float(((x - xm) ** 2).sum())
    slope = float(((x - xm) * (yv - ym)).sum() / (sxx + 1e-12))
    if n <= 2:
        return slope, float("inf")
    resid = yv - (ym + slope * (x - xm))
    s2 = float((resid**2).sum() / (n - 2))
    se = np.sqrt(s2 / (sxx + 1e-12))
    return slope, float(t_quantile_975(n - 2) * se)


# --- M2: attention-block theta + per-path breakdown (spec section 7) ---
_ATTN = ["Wq", "Wk", "Wv", "Wo"]


def attention_block_theta(cosines: Dict[str, float]) -> float:
    """Attention-block theta = MAX angle over {Wq,Wk,Wv,Wo} (worst-aligned). MLP excluded."""
    return max(theta_deg(cosines[k]) for k in _ATTN if k in cosines)


def per_path_theta(cosines: Dict[str, float]) -> Dict[str, float]:
    """Per-path worst angle: value path {Wv,Wo} vs score path {Wq,Wk} (spec section 3)."""
    value = [theta_deg(cosines[k]) for k in ["Wv", "Wo"] if k in cosines]
    score = [theta_deg(cosines[k]) for k in ["Wq", "Wk"] if k in cosines]
    return {
        "value": max(value) if value else float("nan"),
        "score": max(score) if score else float("nan"),
    }
