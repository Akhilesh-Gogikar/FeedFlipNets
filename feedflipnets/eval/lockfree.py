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
