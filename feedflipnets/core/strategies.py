"""Feedback strategies for FeedFlipNets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np

from .quant import quantize_ternary_det
from .types import ActivationState, Array, Gradients, ModelDescription, StrategyState


class FeedbackStrategy(Protocol):
    """Protocol implemented by feedback-alignment strategies."""

    def init(self, model: ModelDescription) -> StrategyState:
        """Initialise internal state for ``model``."""

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        """Return parameter gradients and the (possibly updated) state."""


@dataclass
class _SimpleState(StrategyState):
    """State container for simple strategies."""

    feedback: List[Array] = field(default_factory=list)


@dataclass
class Backprop:
    """Classical backpropagation using the model's forward weights."""

    def init(self, model: ModelDescription) -> StrategyState:  # noqa: D401
        return StrategyState()

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        weights = activations.weights
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs

        grads: Gradients = {}
        batch = error.shape[0]
        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            delta = (delta @ weights[idx + 1].T) * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta / batch
        return grads, state


@dataclass
class DFA:
    """Direct Feedback Alignment with fixed random matrices."""

    rng: np.random.Generator

    def init(self, model: ModelDescription) -> StrategyState:
        dims = model.layer_dims
        output_dim = dims[-1]
        matrices: List[Array] = []
        for hidden_dim in dims[1:-1]:
            scale = 1.0 / np.sqrt(output_dim)
            matrices.append(
                self.rng.standard_normal((output_dim, hidden_dim)).astype(np.float32) * scale
            )
        return _SimpleState(feedback=matrices)

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        weights = activations.weights
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs
        feedback_mats = list(getattr(state, "feedback", []))

        grads: Gradients = {}
        batch = error.shape[0]
        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            feedback = feedback_mats[idx]
            projected = error @ feedback
            delta_layer = projected * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta_layer / batch
        return grads, state


@dataclass
class TernaryDFA:
    """Direct feedback alignment with ternary-quantised feedback."""

    rng: np.random.Generator
    threshold: float = 0.1

    def init(self, model: ModelDescription) -> StrategyState:
        dims = model.layer_dims
        output_dim = dims[-1]
        matrices: List[Array] = []
        for hidden_dim in dims[1:-1]:
            scale = 1.0 / np.sqrt(output_dim)
            matrix = self.rng.standard_normal((output_dim, hidden_dim)).astype(np.float32) * scale
            matrices.append(quantize_ternary_det(matrix, self.threshold))
        return _SimpleState(feedback=matrices)

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        feedback = list(getattr(state, "feedback", []))
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs
        weights = activations.weights

        grads: Gradients = {}
        batch = error.shape[0]
        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            matrix = feedback[idx]
            projected = error @ matrix
            delta_layer = projected * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta_layer / batch
        return grads, state


@dataclass
class StructuredFeedback:
    """Feedback alignment using structured random matrices."""

    rng: np.random.Generator
    structure_type: str = "orthogonal"
    refresh: str = "fixed"
    rank: int | None = None
    blocks: int | None = None

    def init(self, model: ModelDescription) -> StrategyState:
        feedback, layer_shapes, layer_seeds = self._build_stack(model.layer_dims)
        return StrategyState(
            metadata={
                "signature": self._signature(model.layer_dims),
                "pending_refresh": False,
                "layer_dims": list(model.layer_dims),
                "layer_shapes": layer_shapes,
                "layer_seeds": layer_seeds,
                "structure_type": self.structure_type,
            },
            feedback=feedback,
        )

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        feedback = list(state.feedback)
        metadata = dict(state.metadata)
        dims = metadata.get("layer_dims", [])

        if self._needs_refresh(feedback, dims, metadata):
            feedback, layer_shapes, layer_seeds = self._build_stack(dims)
            metadata["pending_refresh"] = False
            metadata["layer_shapes"] = layer_shapes
            metadata["layer_seeds"] = layer_seeds

        weights = activations.weights
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs

        grads: Gradients = {}
        batch = error.shape[0]
        last_idx = len(weights) - 1
        # Output layer gradient uses true error
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ error / batch
        # For hidden layers, project output error directly to each layer using its B_l
        for idx in reversed(range(last_idx)):
            matrix = feedback[idx]
            projected = error @ matrix
            delta_layer = projected * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta_layer / batch

        new_state = StrategyState(feedback=feedback, metadata=metadata)
        return grads, new_state

    # ------------------------------------------------------------------
    # Helpers

    def _needs_refresh(
        self,
        feedback: Sequence[Array],
        dims: Sequence[int],
        metadata: Dict[str, object],
    ) -> bool:
        if not feedback:
            return True
        if self.refresh == "per_step":
            return True
        if self.refresh == "per_epoch":
            return bool(metadata.get("pending_refresh", False))
        return False

    def _signature(self, dims: Sequence[int]) -> List[Tuple[int, int]]:
        """Return (rows, cols) for each hidden layer's feedback B_l.

        We project the output-layer error (dim = ``dims[-1]``) directly to each
        hidden layer (dim = ``dims[1:-1]``), as in classical DFA. Hence each
        feedback matrix has shape ``(dims[-1], dims[l])`` for hidden layer ``l``.
        """
        output_dim = dims[-1]
        return [(output_dim, hidden_dim) for hidden_dim in dims[1:-1]]

    def _build_stack(self, dims: Sequence[int]) -> Tuple[List[Array], List[Tuple[int, int]], List[int]]:
        matrices: List[Array] = []
        shapes: List[Tuple[int, int]] = []
        seeds: List[int] = []
        for out_dim, in_dim in self._signature(dims):
            subseed = int(self.rng.integers(0, 2**31 - 1))
            subrng = np.random.default_rng(subseed)
            matrices.append(self._make_matrix_with_rng(subrng, out_dim, in_dim))
            shapes.append((out_dim, in_dim))
            seeds.append(subseed)
        return matrices, shapes, seeds

    def _make_matrix_with_rng(self, subrng: np.random.Generator, out_dim: int, in_dim: int) -> Array:
        # Delegate based on structure type using a sub-RNG for reproducibility
        if self.structure_type == "orthogonal":
            return _orthogonal(subrng, out_dim, in_dim)
        if self.structure_type == "hadamard":
            return _hadamard_matrix(subrng, out_dim, in_dim)
        if self.structure_type == "blockdiag":
            return _blockdiag_orthogonal(subrng, in_dim, out_dim, self.blocks)
        if self.structure_type == "lowrank":
            return _lowrank(subrng, out_dim, in_dim, self.rank)
        raise ValueError(f"Unknown structure_type: {self.structure_type}")

    def _make_matrix(self, out_dim: int, in_dim: int) -> Array:
        if self.structure_type == "orthogonal":
            return _orthogonal(self.rng, out_dim, in_dim)
        if self.structure_type == "hadamard":
            return _hadamard_matrix(self.rng, out_dim, in_dim)
        if self.structure_type == "blockdiag":
            return _blockdiag_orthogonal(self.rng, in_dim, out_dim, self.blocks)
        if self.structure_type == "lowrank":
            return _lowrank(self.rng, out_dim, in_dim, self.rank)
        raise ValueError(f"Unknown structure_type: {self.structure_type}")


def _orthogonal(rng: np.random.Generator, out_dim: int, in_dim: int) -> Array:
    """Haar (QR) construction with row- or col-orthonormality.

    - If ``out_dim <= in_dim``: return an ``(out_dim, in_dim)`` matrix with
      orthonormal rows (exact near-isometry for error projection).
    - If ``out_dim > in_dim``: return an ``(out_dim, in_dim)`` matrix with
      orthonormal columns (non-expansive mapping; spectral norm 1).

    Uses QR on a Gaussian matrix with sign correction (Mezzadri; Higham).
    """
    m, n = out_dim, in_dim
    if m <= n:
        # Draw A in R^{n x m}, Q (n x m) with orthonormal columns, then return Q^T.
        A = rng.standard_normal((n, m), dtype=np.float64)
        Q, R = np.linalg.qr(A, mode="reduced")
        d = np.sign(np.diag(R))
        d[d == 0] = 1.0
        Q = Q * d  # column sign-fix
        B = Q.T  # (m x n), rows orthonormal
    else:
        # m > n: draw A in R^{m x n}, Q (m x n) with orthonormal columns, return Q.
        A = rng.standard_normal((m, n), dtype=np.float64)
        Q, R = np.linalg.qr(A, mode="reduced")
        d = np.sign(np.diag(R))
        d[d == 0] = 1.0
        Q = Q * d  # column sign-fix
        B = Q  # (m x n), columns orthonormal
    return B.astype(np.float32)


def _hadamard_matrix(rng: np.random.Generator, out_dim: int, in_dim: int) -> Array:
    """Subsampled randomized Hadamard (SRHT-style) rectangle with debiasing.

    Builds an ``(out_dim, in_dim)`` matrix by sampling rows and columns from a
    normalized Hadamard of size ``s = 2^⌈log2(in_dim)⌉`` and applying random
    column signs. A factor ``√(s / in_dim)`` preserves expected row norms when
    ``s > in_dim``. This yields near-isometry (JL/FJLT intuition) when
    ``out_dim <= in_dim`` and remains non-expansive on average otherwise.
    """
    m, n = out_dim, in_dim
    s = 1 << (max(1, n) - 1).bit_length()
    Hs = _hadamard(s)  # (s x s), orthonormal: H H^T = I
    # Column selection and random signs
    cols = rng.permutation(s)[:n]
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    # Row sampling
    rows = rng.permutation(s)[:m]
    B = Hs[np.ix_(rows, cols)]  # (m x n)
    B = (B * signs)  # apply column-wise random signs
    if s > n:
        B = B * np.sqrt(np.float32(s / n))  # debias cropping
    return B.astype(np.float32)


def _lowrank(
    rng: np.random.Generator,
    out_dim: int,
    in_dim: int,
    rank: int | None,
) -> Array:
    r = rank or max(1, min(out_dim, in_dim) // 16)
    U = rng.standard_normal((out_dim, r))
    V = rng.standard_normal((in_dim, r))
    B = U @ V.T
    norms = np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return (B / norms).astype(np.float32)


def _hadamard(n: int) -> Array:
    m = 1 << (max(1, n) - 1).bit_length()
    H = np.array([[1]], dtype=np.float32)
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])
    scale = np.sqrt(np.float32(H.shape[0]))
    return H / scale


def _blockdiag_orthogonal(
    rng: np.random.Generator,
    dim_in: int,
    dim_out: int,
    blocks: int | None,
) -> Array:
    blocks = blocks or max(1, np.gcd(dim_in, dim_out))
    rows = np.array_split(np.arange(dim_out), blocks)
    cols = np.array_split(np.arange(dim_in), blocks)
    B = np.zeros((dim_out, dim_in), dtype=np.float32)
    for r_idx, c_idx in zip(rows, cols):
        if len(r_idx) == 0 or len(c_idx) == 0:
            continue
        A = rng.standard_normal((len(r_idx), len(c_idx)))
        Q, _ = np.linalg.qr(A.T)
        blk = Q.T[: len(r_idx), : len(c_idx)].astype(np.float32)
        B[np.ix_(r_idx, c_idx)] = blk
    return B


__all__ = [
    "FeedbackStrategy",
    "Backprop",
    "DFA",
    "TernaryDFA",
    "StructuredFeedback",
]
