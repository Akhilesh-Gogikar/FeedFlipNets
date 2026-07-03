import numpy as np
import torch

from feedflipnets.core.transformer import ProtoBlock, autograd_ref
from feedflipnets.eval.gradcheck import block_grad_max_rel_err


def test_autograd_block_matches_finite_difference():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    block = ProtoBlock(6, 12, 5, seed=1)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 6))
    e_block = rng.standard_normal((5, 6))
    ref, _ = autograd_ref(block, x, e_block)
    err = block_grad_max_rel_err(block, x, e_block, ref, eps=1e-6)
    assert err < 1e-3  # attention/LN amplify float error (spec §6 GATE-0 tol)
