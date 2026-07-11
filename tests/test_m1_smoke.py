# tests/test_m1_smoke.py
import numpy as np

from experiments.m1_depth_sweep import depth_slope, run_condition


def test_run_condition_deterministic():
    a = run_condition(strategy="dfa", depth=4, seed=0, steps=20, samples_per_step=2)
    b = run_condition(strategy="dfa", depth=4, seed=0, steps=20, samples_per_step=2)
    assert np.isclose(a["min_theta"], b["min_theta"])


def test_fixed_dfa_theta_slope_positive_after_training():
    # Fixed random DFA's worst-layer angle should grow with depth (theta slope vs depth > 0)
    # on TRAINED runs (steps>0). The library helper fixed_dfa_slope_is_negative() is misnamed
    # (it returns slope > 0) and runs steps=0, so we compute the slope here directly to pin
    # an actual training regression while staying fast.
    thetas = [
        run_condition(strategy="dfa", depth=d, seed=0, steps=20, samples_per_step=2)["min_theta"]
        for d in [2, 4, 8]
    ]
    slope, _ = depth_slope([2, 4, 8], thetas)
    assert slope > 0
