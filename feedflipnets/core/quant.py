"""Ternary quantisation helpers.

Conventions:
- Inclusive-zero boundary for ternary mapping: values with ``|x| \le tau``
  map to ``0``; only magnitudes strictly exceeding the threshold are mapped
  to ``\pm 1``. This matches the paper's Definition (and Eq. (1)) and reduces
  flip "chatter" from boundary noise.
"""

from __future__ import annotations

import numpy as np

from .types import Array


def ternary(x: Array) -> Array:
    """Return the sign of ``x`` as ternary floats."""

    out = np.zeros_like(x, dtype=float)
    out[x > 0.0] = 1.0
    out[x < 0.0] = -1.0
    return out


def quantize_ternary_det(weights: Array, tau: float) -> Array:
    """Deterministic ternary quantisation with threshold ``tau``.

    Boundary rule: ``|w| \le tau -> 0`` (inclusive).
    """

    out = np.zeros_like(weights, dtype=float)
    out[weights > tau] = 1.0
    out[weights < -tau] = -1.0
    return out


def quantize_ternary_stoch(weights: Array, tau: float, rng: np.random.Generator) -> Array:
    """Stochastic ternary quantisation matching the legacy behaviour.

    Boundary rule: ``|w + \varepsilon| \le tau -> 0`` (inclusive) after
    adding zero-mean dither ``\varepsilon \sim \mathcal{U}[-\tau, \tau]``.
    """

    noise = rng.uniform(-tau, tau, size=weights.shape)
    jittered = weights + noise
    out = np.zeros_like(weights, dtype=float)
    out[jittered > tau] = 1.0
    out[jittered < -tau] = -1.0
    return out


def pack_ternary(weights: Array) -> np.ndarray:
    """Pack ternary weights into ``int8`` vectors for logging or storage."""

    return ternary(weights).astype(np.int8)
