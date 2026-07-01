import numpy as np

from feedflipnets.core.deep_mlp import DeepMLP, make_perturb_loss_fn, output_error


def _data(n=16, d=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, c, size=n)
    return X, y


def test_forward_shapes_and_activationstate():
    model = DeepMLP(layer_dims=[8, 16, 16, 3], seed=1)
    X, y = _data()
    state, logits, acts = model.forward(X)
    assert logits.shape == (16, 3)
    # one weight per layer, one input per layer, one deriv per hidden layer
    assert len(state.weights) == 3
    assert len(state.layer_inputs) == 3
    assert len(state.layer_derivs) == 2
    assert acts[0].shape == (16, 8) and acts[-1].shape == (16, 16)


def test_perturb_loss_fn_zero_delta_matches_base():
    model = DeepMLP(layer_dims=[8, 16, 16, 3], seed=2)
    X, y = _data()
    fn, base, _state = make_perturb_loss_fn(model, X, y)
    zero = np.zeros((X.shape[0], model.layer_dims[2]))
    got = fn(2, zero)
    assert np.allclose(got, base, atol=1e-10)


def test_output_error_is_softmax_minus_onehot():
    model = DeepMLP(layer_dims=[8, 16, 3], seed=3)
    X, y = _data()
    _state, logits, _acts = model.forward(X)
    err = output_error(logits, y)
    assert err.shape == (16, 3)
    # rows sum to zero: (p - onehot) sums to (1 - 1) = 0
    assert np.allclose(err.sum(axis=1), 0.0, atol=1e-10)
