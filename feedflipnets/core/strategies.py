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
        feedback = self._build_stack(model.layer_dims)
        return StrategyState(
            metadata={
                "signature": self._signature(model.layer_dims),
                "pending_refresh": False,
                "layer_dims": list(model.layer_dims),
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
            feedback = self._build_stack(dims)
            metadata["pending_refresh"] = False

        weights = activations.weights
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs

        grads: Gradients = {}
        batch = error.shape[0]
        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            matrix = feedback[idx]
            delta = (delta @ matrix) * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta / batch

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
        pairs: List[Tuple[int, int]] = []
        for idx in range(len(dims) - 2):
            rows = dims[idx + 2]
            cols = dims[idx + 1]
            pairs.append((rows, cols))
        return pairs

    def _build_stack(self, dims: Sequence[int]) -> List[Array]:
        matrices: List[Array] = []
        for out_dim, in_dim in self._signature(dims):
            matrices.append(self._make_matrix(out_dim, in_dim))
        return matrices

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
    A = rng.standard_normal((out_dim, in_dim))
    Q, _ = np.linalg.qr(A.T)
    return Q.T[:out_dim, :in_dim].astype(np.float32)


def _hadamard_matrix(rng: np.random.Generator, out_dim: int, in_dim: int) -> Array:
    size = max(out_dim, in_dim)
    H = _hadamard(size)
    if H.shape[0] < size:
        return _orthogonal(rng, out_dim, in_dim)
    return H[:out_dim, :in_dim].astype(np.float32)


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


@dataclass
class PerturbationTaughtFeedback:
    """② Transport-free feedback learning.

    Fast path is DFA. Each step, one feedback matrix (round-robin) is nudged so that
    ``error @ B_idx`` aligns with a node-perturbation estimate of dL/dh at the corresponding
    hidden activation. The estimate uses only the broadcast error and a scalar loss delta
    obtained from ``state.metadata['perturb_loss_fn']`` — it never reads a downstream weight.
    Update is direction-only (scale-invariant); antithetic ±xi cancels O(rho^2) curvature bias.
    """

    rng: np.random.Generator
    rho: float = 0.05
    lr_B: float = 0.1
    samples_per_step: int = 4

    def init(self, model: ModelDescription) -> StrategyState:
        dims = model.layer_dims
        out = dims[-1]
        mats: List[Array] = [
            (self.rng.standard_normal((out, hd)) / np.sqrt(out)).astype(np.float64)
            for hd in dims[1:-1]
        ]
        return StrategyState(feedback=mats, metadata={"step": 0})

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        weights = activations.weights
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs
        B = list(state.feedback)
        md = dict(state.metadata)

        grads: Gradients = {}
        batch = error.shape[0]
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ error / batch
        for idx in reversed(range(last_idx)):
            projected = error @ B[idx]
            delta_layer = projected * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta_layer / batch

        fn = md.get("perturb_loss_fn")
        if fn is not None and last_idx >= 1:
            step = int(md.get("step", 0))
            idx = step % last_idx  # feedback index 0..last_idx-1
            hidden_idx = idx + 1  # perturb post-activation acts[hidden_idx]
            width = B[idx].shape[1]
            ghat = np.zeros((batch, width), dtype=np.float64)
            for _ in range(self.samples_per_step):
                xi = self.rng.standard_normal((batch, width)) * self.rho
                loss_plus = fn(hidden_idx, xi)
                loss_minus = fn(hidden_idx, -xi)
                ghat += ((loss_plus - loss_minus) / (2.0 * self.rho**2))[:, None] * xi
            ghat /= self.samples_per_step

            projected = error @ B[idx]
            unit_g = ghat / (np.linalg.norm(ghat, axis=1, keepdims=True) + 1e-12)
            unit_p = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-12)
            dB = error.T @ (unit_g - unit_p) / batch
            B[idx] = B[idx] + self.lr_B * dB
            md["step"] = step + 1

        return grads, StrategyState(feedback=B, metadata=md)


# --- M2: ① Activation-Routed DFA (torch testbed) ---
from .transformer import forward_np_cache as _fwd_np  # noqa: E402
from .transformer import softmax_jac_apply as _sjac  # noqa: E402


@dataclass
class ActivationRoutedDFA:
    """① Activation-Routed DFA for a pre-LN single-head decoder block.

    Reuses cached data-maps EXACTLY (A, softmax-Jac J_A, LN VJP, phi' mask). Replaces ONLY the two
    weight-transposes W_O^T, W_2^T with fixed-random surrogates R_O, R_2 (1-alone) -- 2 adapts them.
    Two DFA broadcast points (one per pre-LN sublayer): e_attn ~= dL/dx1, e_mlp ~= dL/dx2, so that
    dL/dW_O = Ctx^T.e_attn and dL/dW_2 = H^T.e_mlp value-EXACT. Never dereferences W_O^T / W_2^T.

    adapt_R_O=True enables the 1+2 surrogate update (Kolen-Pollack, transport-free): the runner
    supplies a genuine node-perturbation estimate ghat_ctx of dL/dCtx (block-forward only, no
    autograd, no W_O^T) and this nudges R_O TOWARD W_O^T's direction -- PARTIALLY (validated:
    ||R_O-W_O^T|| stays ~1.0, cos~0.69 at d=32; the M1 1/sqrt(d) variance wall), never reaching
    exactness, and never reading W_O (see experiments/m2_attention_alignment.py::adapt_R_O_honest).
    """

    R_O: Array
    R_2: Array
    adapt_R_O: bool = False
    lr_R: float = 0.02

    def block_grads(
        self,
        block,
        x_np: Array,
        e_attn: Array,
        e_mlp: Array,
        ghat_ctx: Array | None = None,
    ) -> Gradients:
        c = _fwd_np(block, x_np)
        d = block.d
        # --- MLP sublayer (x2 = x1 + M): dW2 EXACT, dH via surrogate R_2, phi' reused exactly ---
        dM = e_mlp
        dW2 = c["H"].T @ dM  # EXACT
        dH = dM @ self.R_2  # surrogate R_2 for W_2ᵀ
        dpre = dH * c["phi_mask"]  # phi' mask reused EXACTLY
        dW1 = c["z2"].T @ dpre
        # --- attention sublayer (x1 = x + O): dWo EXACT, dCtx via surrogate R_O, A/J_A reused ---
        dO = e_attn  # dL/dO = dL/dx1
        dWo = c["Ctx"].T @ dO  # EXACT (no transport)
        dCtx = dO @ self.R_O  # surrogate R_O for W_Oᵀ (ONLY attention transport)
        dV = c["A"].T @ dCtx  # A cached (exact)
        dA = dCtx @ c["V"].T  # V cached (exact)
        dS = np.empty_like(c["A"])
        for r in range(block.T):
            dS[r] = _sjac(c["A"][r], dA[r])  # softmax Jac from cached A (exact)
        scale = 1.0 / np.sqrt(d)
        dQ = (dS @ c["K"]) * scale  # K cached
        dK = (dS.T @ c["Q"]) * scale  # Q cached
        dWq = c["y"].T @ dQ  # y cached
        dWk = c["y"].T @ dK
        dWv = c["y"].T @ dV
        if self.adapt_R_O and ghat_ctx is not None:
            # transport-free Kolen-Pollack: nudge R_O TOWARD W_O^T's direction via regression of the
            # (surrogate) dCtx onto the node-perturbation estimate ghat_ctx. PARTIAL, never reaches
            # W_O^T (1/sqrt(d) wall); ghat_ctx MUST be forward-only, NEVER autograd(x2, Ctx).
            pred = dO @ self.R_O
            self.R_O = self.R_O + self.lr_R * (dO.T @ (ghat_ctx - pred)) / block.T
        return {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo, "W1": dW1, "W2": dW2, "dCtx": dCtx}


def fixed_dfa_block_grads(
    block,
    x_np: Array,
    e_attn: Array,
    e_mlp: Array,
    Bq: Array,
    Bk: Array,
    Bv: Array,
    B1: Array,
) -> Gradients:
    """Fixed-DFA-in-a-block baseline: broadcast e to Q/K/V/pre via independent random matrices,
    IGNORING A entirely. dW_O/dW_2 use the exact residual error (same as ①), so the ablation
    isolates the win to the A-routed score/value path.
    """
    c = _fwd_np(block, x_np)
    dO = e_attn
    dWo = c["Ctx"].T @ dO
    dWq = c["y"].T @ (e_attn @ Bq)
    dWk = c["y"].T @ (e_attn @ Bk)
    dWv = c["y"].T @ (e_attn @ Bv)
    dpre = (e_attn @ B1) * c["phi_mask"]
    dW1 = c["z2"].T @ dpre
    dW2 = c["H"].T @ e_mlp
    return {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo, "W1": dW1, "W2": dW2}


__all__ = [
    "FeedbackStrategy",
    "Backprop",
    "DFA",
    "TernaryDFA",
    "StructuredFeedback",
    "PerturbationTaughtFeedback",
    "ActivationRoutedDFA",
    "fixed_dfa_block_grads",
]
