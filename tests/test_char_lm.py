# tests/test_char_lm.py
import numpy as np

from feedflipnets.data.char_lm import bigram_floor, build_cfg_corpus, get_batch, prep


def test_cfg_corpus_is_deterministic_and_structured():
    a = build_cfg_corpus(seed=0)
    b = build_cfg_corpus(seed=0)
    assert a == b  # seeded, reproducible
    assert len(a) > 100_000  # enough chars to train a small LM
    assert "(" in a and ")" in a  # nested brackets → long-range structure


def test_prep_splits_and_vocab():
    text = build_cfg_corpus(seed=0)
    train, val, V = prep(text)
    assert V == len(set(text))
    assert train.dtype == np.int64 and val.dtype == np.int64
    assert len(val) == len(text) - len(train)
    assert int(train.max()) < V and int(val.max()) < V


def test_bigram_floor_above_two_bpc_on_cfg():
    train, val, V = prep(build_cfg_corpus(seed=0))
    floor = bigram_floor(train, val, V)
    # A bigram cannot capture the nested grammar → floor is high (prototype ≈ 2.881 bpc).
    assert 2.0 < floor < 4.0


def test_get_batch_shapes_and_next_char_target():
    rng = np.random.default_rng(0)
    train, _val, _V = prep(build_cfg_corpus(seed=0))
    xb, yb = get_batch(train, bs=8, T=16, rng=rng)
    assert xb.shape == (8, 16) and yb.shape == (8, 16)
    # yb is xb shifted by one (next-char prediction)
    assert np.array_equal(xb[:, 1:], yb[:, :-1])
