"""Configurable-depth NumPy MLP testbed for M1 alignment experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from .types import ActivationState

Array = np.ndarray


def relu(x: Array) -> Array:
    return np.maximum(0.0, x)


def softmax(z: Array) -> Array:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y: Array, num_classes: int) -> Array:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def softmax_ce_per_sample(logits: Array, y: Array) -> Array:
    """Per-sample cross-entropy, shape (batch,)."""
    p = softmax(logits)
    yoh = one_hot(y, logits.shape[1])
    return -(yoh * np.log(p + 1e-12)).sum(axis=1)


def output_error(logits: Array, y: Array) -> Array:
    """dL/dz at the output = softmax - one_hot, shape (batch, C). NOT divided by batch."""
    return softmax(logits) - one_hot(y, logits.shape[1])


@dataclass
class DeepMLP:
    """Bias-free ReLU MLP. layer_dims = [d_in, h1, ..., h_{L-1}, C]."""

    layer_dims: List[int]
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights: List[Array] = [
            (
                rng.standard_normal((self.layer_dims[i], self.layer_dims[i + 1]))
                / np.sqrt(self.layer_dims[i])
            ).astype(np.float64)
            for i in range(len(self.layer_dims) - 1)
        ]

    @property
    def num_layers(self) -> int:
        return len(self.weights)

    def forward(self, X: Array) -> Tuple[ActivationState, Array, List[Array]]:
        L = self.num_layers
        h = X.astype(np.float64)
        layer_inputs: List[Array] = []
        layer_derivs: List[Array] = []
        acts: List[Array] = [h]  # acts[j] = input activation feeding layer j (acts[0] = X)
        logits = h
        for idx in range(L):
            layer_inputs.append(h)
            z = h @ self.weights[idx]
            if idx < L - 1:
                layer_derivs.append((z > 0).astype(np.float64))
                h = relu(z)
                acts.append(h)
            else:
                logits = z
        state = ActivationState(
            layer_inputs=layer_inputs,
            layer_derivs=layer_derivs,
            weights=[w.copy() for w in self.weights],
        )
        return state, logits, acts

    def forward_from(self, start_idx: int, h_start: Array) -> Array:
        """Re-run layers start_idx..L-1 given the activation feeding layer start_idx."""
        L = self.num_layers
        h = h_start
        logits = h
        for idx in range(start_idx, L):
            z = h @ self.weights[idx]
            if idx < L - 1:
                h = relu(z)
            else:
                logits = z
        return logits


def make_perturb_loss_fn(
    model: DeepMLP, X: Array, y: Array
) -> Tuple[Callable[[int, Array], Array], Array, ActivationState]:
    """Return (fn, base_per_sample_loss, activation_state).

    fn(hidden_idx, delta_h) perturbs the post-activation acts[hidden_idx] by delta_h and
    returns the per-sample loss after re-running the rest of the network. Samples are
    independent in an MLP, so per-sample node perturbation is exact.
    """
    state, logits, acts = model.forward(X)
    base = softmax_ce_per_sample(logits, y)

    def fn(hidden_idx: int, delta_h: Array) -> Array:
        h_pert = acts[hidden_idx] + delta_h
        logits2 = model.forward_from(hidden_idx, h_pert)
        return softmax_ce_per_sample(logits2, y)

    return fn, base, state
