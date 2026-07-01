import numpy as np

from feedflipnets.core.deep_mlp import DeepMLP, output_error
from feedflipnets.core.strategies import DFA, Backprop
from feedflipnets.eval.lockfree import e_fixed_perturbation_max_change


def _setup():
    model = DeepMLP(layer_dims=[6, 10, 10, 4], seed=7)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 6))
    y = rng.integers(0, 4, size=8)
    state, logits, _acts = model.forward(X)
    err = output_error(logits, y)
    return model, state, err


def test_dfa_is_lock_free():
    model, state, err = _setup()
    strat = DFA(rng=np.random.default_rng(0))
    change = e_fixed_perturbation_max_change(
        strat, model, state, err, downstream_idx=2, upstream_key="W0"
    )
    assert change < 1e-12  # grad of W0 does not depend on downstream weight W2


def test_backprop_is_rejected():
    model, state, err = _setup()
    change = e_fixed_perturbation_max_change(
        Backprop(), model, state, err, downstream_idx=2, upstream_key="W0"
    )
    assert change > 1e-6  # BP's W0 grad DOES depend on the downstream weight
