"""Finite-difference gradient check for feedback strategies (GATE-0)."""
from __future__ import annotations

import numpy as np
import torch

from ..core.deep_mlp import DeepMLP, output_error, softmax_ce_per_sample

Array = np.ndarray


class _Desc:
    """Minimal ``ModelDescription``-like shim exposing ``layer_dims``."""

    def __init__(self, dims):
        self.layer_dims = dims


def _strategy_grads(strategy, model: DeepMLP, X: Array, y: Array):
    state, logits, _acts = model.forward(X)
    err = output_error(logits, y)
    sstate = strategy.init(_Desc(list(model.layer_dims)))
    grads, _ = strategy.backward(state, err, sstate)
    return grads


def max_rel_err_vs_finite_diff(
    strategy, model: DeepMLP, X: Array, y: Array, eps: float = 1e-6
) -> float:
    """Max relative error between a strategy's grads and central finite differences.

    The finite-difference reference is the TRUE loss gradient (exact BP). Use only with
    strategies expected to match BP (e.g. Backprop) as the GATE-0 correctness gate.
    """
    grads = _strategy_grads(strategy, model, X, y)
    n = X.shape[0]
    worst = 0.0
    for idx, W in enumerate(model.weights):
        key = f"W{idx}"
        analytic = grads[key]
        num = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])
        while not it.finished:
            i, j = it.multi_index
            orig = W[i, j]
            W[i, j] = orig + eps
            lp = softmax_ce_per_sample(model.forward(X)[1], y).sum() / n
            W[i, j] = orig - eps
            lm = softmax_ce_per_sample(model.forward(X)[1], y).sum() / n
            W[i, j] = orig
            num[i, j] = (lp - lm) / (2 * eps)
            it.iternext()
        denom = np.maximum(np.abs(analytic) + np.abs(num), 1e-8)
        worst = max(worst, float(np.max(np.abs(analytic - num) / denom)))
    return worst


# --- M2 additions: block-level grad-check + value-path-exact check ---


def _block_scalar_loss(block, x_np, e_block_np):
    x2, _ = block.forward_torch(x_np, requires_grad_params=False)
    return float((torch.tensor(e_block_np, dtype=torch.float64) * x2).sum().detach())


def block_grad_max_rel_err(block, x_np, e_block_np, ref_grads, eps: float = 1e-6) -> float:
    """Max rel-err between the autograd reference and central finite differences on the block."""
    worst = 0.0
    for name in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]:
        W = getattr(block, name)
        a = ref_grads[name]
        num = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])
        while not it.finished:
            i, j = it.multi_index
            o = W[i, j]
            W[i, j] = o + eps
            lp = _block_scalar_loss(block, x_np, e_block_np)
            W[i, j] = o - eps
            lm = _block_scalar_loss(block, x_np, e_block_np)
            W[i, j] = o
            num[i, j] = (lp - lm) / (2 * eps)
            it.iternext()
        denom = np.maximum(np.abs(a) + np.abs(num), 1e-8)
        worst = max(worst, float(np.max(np.abs(a - num) / denom)))
    return worst


def value_path_exact_err(one_grads, ref_grads) -> float:
    """Max abs-err of ①'s dW_O and dW_2 vs the autograd reference (must be < 1e-5)."""
    return max(
        float(np.max(np.abs(one_grads["Wo"] - ref_grads["Wo"]))),
        float(np.max(np.abs(one_grads["W2"] - ref_grads["W2"]))),
    )
