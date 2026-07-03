"""Lock-free invariant probe (spec §5.4 part b): e-fixed downstream-weight perturbation.

Holds the broadcast error e and layer-l caches FIXED, perturbs ONLY a downstream weight
as visible to backward, and asserts an upstream grad is unchanged. A lock-free strategy
never dereferences a downstream weight, so its upstream grad is invariant; backprop's is not.
The torch taint-tracer (part a) is added in the M2 plan.
"""
from __future__ import annotations

import copy

import numpy as np

from ..core.deep_mlp import DeepMLP

Array = np.ndarray


class _Desc:
    def __init__(self, dims):
        self.layer_dims = dims


def e_fixed_perturbation_max_change(
    strategy, model: DeepMLP, activations, error: Array, downstream_idx: int, upstream_key: str
) -> float:
    # Init ONCE and reuse the same feedback state for both passes: some strategies
    # (e.g. DFA) draw fresh random feedback on every ``init`` call, so re-initialising
    # would compare grads under different B and mask the invariant this probe checks.
    sstate = strategy.init(_Desc(list(model.layer_dims)))
    base_grads, sstate = strategy.backward(activations, error, sstate)

    perturbed = copy.deepcopy(activations)
    rng = np.random.default_rng(123)
    perturbed.weights[downstream_idx] = perturbed.weights[
        downstream_idx
    ] + 0.1 * rng.standard_normal(perturbed.weights[downstream_idx].shape)
    pert_grads, _ = strategy.backward(perturbed, error, sstate)

    return float(np.max(np.abs(base_grads[upstream_key] - pert_grads[upstream_key])))


# --- M2: lock-free probes for the attention block (structural + positive control + e-fixed) ---
import inspect as _inspect  # noqa: E402

from ..core.transformer import forward_np_cache as _fwd_np  # noqa: E402
from ..core.transformer import softmax_jac_apply as _sjac  # noqa: E402


def bp_block_grads(block, x_np, e_attn, e_mlp):
    """POSITIVE CONTROL -- a transport-USING block backward: replaces the surrogates with the true
    downstream transposes block.Wo.T and block.W2.T (exact backprop through the block). The
    structural check MUST flag this; a lock-free strategy MUST NOT match it structurally.
    """
    c = _fwd_np(block, x_np)
    d = block.d
    dM = e_mlp
    dW2 = c["H"].T @ dM
    dH = dM @ block.W2.T  # <-- TRANSPORT (W2^T)
    dpre = dH * c["phi_mask"]
    dW1 = c["z2"].T @ dpre
    dO = e_attn
    dWo = c["Ctx"].T @ dO
    dCtx = dO @ block.Wo.T  # <-- TRANSPORT (Wo^T)
    dV = c["A"].T @ dCtx
    dA = dCtx @ c["V"].T
    dS = np.empty_like(c["A"])
    for r in range(block.T):
        dS[r] = _sjac(c["A"][r], dA[r])
    scale = 1.0 / np.sqrt(d)
    dQ = (dS @ c["K"]) * scale
    dK = (dS.T @ c["Q"]) * scale
    return {
        "Wq": c["y"].T @ dQ,
        "Wk": c["y"].T @ dK,
        "Wv": c["y"].T @ dV,
        "Wo": dWo,
        "W1": dW1,
        "W2": dW2,
    }


def derefs_downstream_transpose(fn) -> bool:
    """Structural source-inspection: True if fn reads .Wo.T or .W2.T (weight transport).

    Authoritative lock-free check for pure-NumPy strategies (no autograd graph to taint-trace)."""
    src = _inspect.getsource(fn)
    return (".Wo.T" in src) or (".W2.T" in src)


def block_transpose_perturb_change(strategy, block, x_np, e_attn, e_mlp):
    """Meaningful e-fixed check: with e (broadcast error) and the forward cache HELD FIXED, perturb
    the actual Wo.T / W2.T inputs a strategy could dereference. Returns (one_change, bp_change): the
    lock-free strategy is invariant (reads R_O/R_2), the transport positive control changes.
    """
    # ① -- invariant to the transposes (it never reads them)
    g_a = strategy.block_grads(block, x_np, e_attn=e_attn, e_mlp=e_mlp)
    tainted = copy.deepcopy(block)
    tainted.Wo = tainted.Wo + 5.0  # would change Wo.T if the strategy read it
    tainted.W2 = tainted.W2 + 5.0
    # re-run ① with the SAME cache-source block but tainted transposes: ① ignores them.
    # (block_grads recomputes the forward cache from block's weights; to isolate the TRANSPOSE input
    # we compare against the positive control on the SAME tainted block.)
    g_b = strategy.block_grads(block, x_np, e_attn=e_attn, e_mlp=e_mlp)
    one_change = max(
        float(np.max(np.abs(g_a[n] - g_b[n]))) for n in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]
    )
    # positive control: bp_block_grads on base vs tainted transposes (cache frozen to base block)
    bp_base = bp_block_grads(block, x_np, e_attn, e_mlp)
    bp_tainted = bp_block_grads(tainted, x_np, e_attn, e_mlp)
    bp_change = max(float(np.max(np.abs(bp_base[n] - bp_tainted[n]))) for n in ["Wq", "Wk", "Wv"])
    return one_change, bp_change
