import numpy as np

from feedflipnets.core.activations import hadamard_pre, relu
from feedflipnets.core.quant import (
    quantize_ternary_det,
    quantize_ternary_stoch,
    ternary,
)
from feedflipnets.utils import make_dataset


def test_relu_and_hadamard_padding():
    x = np.array([-1.0, 0.0, 2.5], dtype=np.float32)
    relu_out = relu(x)
    assert np.allclose(relu_out, np.array([0.0, 0.0, 2.5]))

    # hadamard_pre pads to the next power of two and divides by sqrt(next_pow/size),
    # so each row's norm shrinks by exactly that factor (padding adds zeros).
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((2, 3)).astype(np.float32)
    padded = hadamard_pre(rows)
    assert padded.shape[-1] == 4
    scale = np.sqrt(4 / 3)
    for i in range(2):
        assert np.allclose(padded[i, :3] * scale, rows[i], atol=1e-6)
        assert np.allclose(padded[i, 3:], 0.0)
        assert np.isclose(np.linalg.norm(padded[i]), np.linalg.norm(rows[i]) / scale, atol=1e-6)


def test_quantization_deterministic_vs_stochastic():
    weights = np.array([[0.1, -0.2], [0.06, -0.01]], dtype=np.float32)
    det = quantize_ternary_det(weights, tau=0.05)
    assert set(np.unique(det)) <= {-1.0, 0.0, 1.0}

    rng = np.random.default_rng(0)
    stoch = quantize_ternary_stoch(weights, tau=0.05, rng=rng)
    assert stoch.shape == weights.shape
    assert not np.allclose(det, stoch)
    assert np.array_equal(ternary(weights), np.sign(weights))


def test_make_dataset_synthetic_quickstart():
    x, y = make_dataset(freq=3, n=8)
    assert x.shape[1] == 8
    assert y.shape == x.shape
