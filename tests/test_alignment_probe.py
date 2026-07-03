import numpy as np

from feedflipnets.eval.alignment import (
    depth_slope,
    min_theta_over_layers,
    per_matrix_cosine,
    theta_deg,
)


def test_identical_grads_give_cosine_one():
    a = {"W0": np.ones((3, 4)), "W1": np.array([[1.0, -2.0]])}
    cos = per_matrix_cosine(a, a)
    assert np.allclose(list(cos.values()), 1.0, atol=1e-9)


def test_orthogonal_grads_give_cosine_zero():
    a = {"W0": np.array([[1.0, 0.0]])}
    b = {"W0": np.array([[0.0, 1.0]])}
    cos = per_matrix_cosine(a, b)
    assert abs(cos["W0"]) < 1e-9


def test_theta_and_slope():
    assert abs(theta_deg(1.0) - 0.0) < 1e-9
    assert abs(theta_deg(0.0) - 90.0) < 1e-9
    # theta increasing with depth -> positive slope of theta vs L
    slope, _ci = depth_slope([2, 4, 8], [10.0, 20.0, 40.0])
    assert slope > 0


def test_min_theta_over_layers_picks_worst():
    cos = {"W0": 0.9, "W1": 0.1}
    assert abs(min_theta_over_layers(cos) - theta_deg(0.1)) < 1e-9
