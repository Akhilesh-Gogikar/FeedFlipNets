# tests/test_m2_ablation_smoke.py
import numpy as np

from experiments.m2_attention_alignment import CONDITIONS, run_condition


def test_conditions_registered():
    assert set(CONDITIONS) == {"fixed_dfa", "one_only", "two_only", "one_two"}


def test_run_condition_deterministic():
    a = run_condition("one_only", depth=1, seed=0, n_draws=8)
    b = run_condition("one_only", depth=1, seed=0, n_draws=8)
    assert np.isclose(a["attn_block_theta"], b["attn_block_theta"])


def test_value_path_exact_for_one():
    r = run_condition("one_only", depth=1, seed=0, n_draws=4)
    assert r["value_path_exact_err"] < 1e-5


def test_one_two_beats_fixed_dfa_on_attention_block():
    # ①+② with the HONEST transport-free adapted R_O beats fixed-DFA attention-block theta by
    # >= 5 deg (LEVEL gate, spec §6/M2). R_O is NOT exact — exactness is not required, not achieved.
    fixed = run_condition("fixed_dfa", depth=1, seed=0, n_draws=20)
    onetwo = run_condition("one_two", depth=1, seed=0, n_draws=20, adapt_steps=2000, adapt_K=16)
    assert (fixed["attn_block_theta"] - onetwo["attn_block_theta"]) >= 5.0
    assert onetwo["R_O_rel_err"] > 0.3  # partial recovery only (no transport smuggling)


def test_two_only_adapts_transport_free_without_a_reuse():
    # ②-only = A-free adapted broadcast (no ① routing); it should still beat fixed_dfa somewhat but
    # by LESS than ①+② (which adds A-reuse). Registered ordering, not exactness. At K=16 the A-free
    # ĝ_V estimator is still at the ~90° noise floor (1/√d variance wall); raise adapt_K (variance
    # ∝ d) rather than loosening the assertion, per the plan's margin guidance.
    fixed = run_condition("fixed_dfa", depth=1, seed=0, n_draws=20)
    two = run_condition("two_only", depth=1, seed=0, n_draws=20, adapt_steps=2000, adapt_K=32)
    assert two["attn_block_theta"] <= fixed["attn_block_theta"]
