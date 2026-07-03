import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from experiments.m2b_lm_ablation import CONDITIONS, run_condition  # noqa: E402


def test_conditions_registered():
    assert set(CONDITIONS) == {"BP", "FIXED_DFA", "ONE_TWO"}


def test_one_two_learns_and_beats_fixed_dfa_smoke():
    # reduced-budget smoke: ①+② should already beat fixed-DFA in bpc (structure vs no-structure),
    # and BP should beat both. Full n≥5 gate is Task 7.
    cfg = dict(
        d=32,
        h=64,
        N=2,
        T=24,
        bs=16,
        steps=200,
        lr=3e-3,
        rho=0.02,
        Ksamp=4,
        lrR=0.05,
        seed=0,
    )
    bp = run_condition("BP", **cfg)
    fx = run_condition("FIXED_DFA", **cfg)
    ot = run_condition("ONE_TWO", **cfg)
    assert ot["bpc"] < fx["bpc"]  # ①+② beats fixed-DFA (A-reuse learns structure)
    assert bp["bpc"] < ot["bpc"]  # honest: BP still wins (the NO-GO signal)
