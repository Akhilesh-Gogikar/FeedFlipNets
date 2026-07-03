import numpy as np
import torch

from feedflipnets.core.transformer import ProtoBlock, autograd_ref, forward_np_cache


def test_forward_shapes_and_cache():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    d, d_ff, T = 6, 12, 5
    block = ProtoBlock(d, d_ff, T, seed=1)
    x = np.random.default_rng(0).standard_normal((T, d))
    x2, cache = block.forward_torch(x)
    assert tuple(x2.shape) == (T, d)
    for key in ["y", "Q", "K", "V", "A", "Ctx"]:
        assert key in cache
    # A rows are a softmax (sum to 1)
    A = cache["A"].detach().numpy()
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-10)


def test_np_cache_matches_torch_forward():
    d, d_ff, T = 6, 12, 5
    block = ProtoBlock(d, d_ff, T, seed=2)
    x = np.random.default_rng(1).standard_normal((T, d))
    _x2, tcache = block.forward_torch(x)
    ncache = forward_np_cache(block, x)
    for key in ["A", "Ctx", "V", "y"]:
        assert np.allclose(ncache[key], tcache[key].detach().numpy(), atol=1e-10), key


def test_autograd_ref_returns_param_and_x1_grads():
    d, d_ff, T = 6, 12, 5
    block = ProtoBlock(d, d_ff, T, seed=3)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((T, d))
    e_block = rng.standard_normal((T, d))
    ref, _cache = autograd_ref(block, x, e_block)
    for key in ["Wq", "Wk", "Wv", "Wo", "W1", "W2", "Ctx", "x1"]:
        assert key in ref
    # dL/dW_O = Ctx^T @ (dL/dx1); reference must satisfy it
    ncache = forward_np_cache(block, x)
    assert np.allclose(ncache["Ctx"].T @ ref["x1"], ref["Wo"], atol=1e-9)
