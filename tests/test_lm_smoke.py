# tests/test_lm_smoke.py
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from feedflipnets.core.lm import NumpyLM  # noqa: E402
from feedflipnets.data.char_lm import build_cfg_corpus, get_batch, prep  # noqa: E402


def _train_a_bit(mode, steps=60):
    train, _val, V = prep(build_cfg_corpus(seed=0))
    m = NumpyLM(V=V, d=32, h=64, N=2, T=24, seed=0, mode=mode)
    m.set_opt(lr=3e-3)
    losses = []
    for _ in range(steps):
        xb, yb = get_batch(train, bs=16, T=24, rng=m.rng)
        losses.append(m.train_step(xb, yb, adapt_cfg=(0.02, 4, 0.05)))
    return np.array(losses)


def test_one_two_loss_decreases():
    losses = _train_a_bit("ONE_TWO")
    # loss (bpc) drops over training — the ①+② scheme actually learns
    assert losses[:5].mean() - losses[-5:].mean() > 0.2


def test_fixed_dfa_loss_decreases():
    losses = _train_a_bit("FIXED_DFA")
    assert losses[:5].mean() - losses[-5:].mean() > 0.1


def test_train_step_returns_bpc_scalar():
    losses = _train_a_bit("ONE_TWO", steps=5)
    assert np.isfinite(losses).all() and (losses > 0).all()
