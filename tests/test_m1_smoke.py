# tests/test_m1_smoke.py
import numpy as np

from experiments.m1_depth_sweep import fixed_dfa_slope_is_negative, run_condition


def test_run_condition_deterministic():
    a = run_condition(strategy="dfa", depth=4, seed=0, steps=20, samples_per_step=2)
    b = run_condition(strategy="dfa", depth=4, seed=0, steps=20, samples_per_step=2)
    assert np.isclose(a["min_theta"], b["min_theta"])


def test_fixed_dfa_negative_control():
    # Fixed random DFA's worst-layer angle should grow with depth (theta slope > 0).
    assert fixed_dfa_slope_is_negative() is True
