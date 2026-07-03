import numpy as np

from feedflipnets.core.deep_mlp import DeepMLP
from feedflipnets.core.strategies import Backprop
from feedflipnets.eval.gradcheck import max_rel_err_vs_finite_diff


def test_backprop_matches_finite_difference():
    model = DeepMLP(layer_dims=[6, 10, 4], seed=5)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 6))
    y = rng.integers(0, 4, size=12)
    err = max_rel_err_vs_finite_diff(Backprop(), model, X, y, eps=1e-6)
    assert err < 1e-4
