# tests/test_activation_routed_dfa.py
import numpy as np
import torch

from feedflipnets.core.strategies import ActivationRoutedDFA, fixed_dfa_block_grads
from feedflipnets.core.transformer import ProtoBlock, autograd_ref
from feedflipnets.eval.gradcheck import value_path_exact_err


def _setup():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    d, d_ff, T = 6, 12, 5
    block = ProtoBlock(d, d_ff, T, seed=1)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((T, d))
    e_block = rng.standard_normal((T, d))
    ref, _ = autograd_ref(block, x, e_block)
    return block, x, e_block, ref, rng


def test_value_path_exact():
    block, x, e_block, ref, rng = _setup()
    d, d_ff = block.d, block.d_ff
    R_O = rng.standard_normal((d, d)) / np.sqrt(d)
    R_2 = rng.standard_normal((d, d_ff)) / np.sqrt(d)
    strat = ActivationRoutedDFA(R_O=R_O, R_2=R_2)
    # two DFA broadcast points: e_attn = dL/dx1 (exact ref), e_mlp = dL/dx2 = e_block
    grads = strat.block_grads(block, x, e_attn=ref["x1"], e_mlp=e_block)
    assert value_path_exact_err(grads, ref) < 1e-5


def test_a_reuse_ceiling_exact_surrogate_gives_zero_angle():
    # With R_O = Wo^T (forbidden in production; ceiling probe), value-path dV is exact -> cos 1.
    block, x, e_block, ref, rng = _setup()
    d, d_ff = block.d, block.d_ff
    strat = ActivationRoutedDFA(
        R_O=block.Wo.T.copy(), R_2=rng.standard_normal((d, d_ff)) / np.sqrt(d)
    )
    grads = strat.block_grads(block, x, e_attn=ref["x1"], e_mlp=e_block)
    a, b = grads["Wv"].ravel(), ref["Wv"].ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    assert cos > 0.999  # A-reuse recovers dW_V exactly when the surrogate is exact


def test_one_beats_fixed_dfa_on_surrogate_matrices():
    # Averaged over surrogate draws, ①'s |cos| on Wv/Wq/Wk exceeds fixed-DFA's (it reuses A).
    block, x, e_block, ref, rng = _setup()
    d, d_ff = block.d, block.d_ff

    def cosine(g, r):
        a, b = g.ravel(), r.ravel()
        return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))

    one_acc = {n: [] for n in ["Wv", "Wq", "Wk"]}
    base_acc = {n: [] for n in ["Wv", "Wq", "Wk"]}
    for s in range(30):
        rr = np.random.default_rng(100 + s)
        R_O = rr.standard_normal((d, d)) / np.sqrt(d)
        R_2 = rr.standard_normal((d, d_ff)) / np.sqrt(d)
        g = ActivationRoutedDFA(R_O=R_O, R_2=R_2).block_grads(
            block, x, e_attn=ref["x1"], e_mlp=e_block
        )
        B = {k: rr.standard_normal((d, d)) / np.sqrt(d) for k in ["Bq", "Bk", "Bv"]}
        B["B1"] = rr.standard_normal((d, d_ff)) / np.sqrt(d)
        bs = fixed_dfa_block_grads(block, x, ref["x1"], e_block, **B)
        for n in one_acc:
            one_acc[n].append(cosine(g[n], ref[n]))
            base_acc[n].append(cosine(bs[n], ref[n]))
    for n in ["Wv", "Wq", "Wk"]:
        assert np.mean(one_acc[n]) > np.mean(base_acc[n]), n
