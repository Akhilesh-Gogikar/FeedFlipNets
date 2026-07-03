# tests/test_attention_lockfree.py
import numpy as np
import torch

from feedflipnets.core.strategies import ActivationRoutedDFA
from feedflipnets.core.transformer import ProtoBlock, autograd_ref
from feedflipnets.eval.lockfree import (
    block_transpose_perturb_change,
    bp_block_grads,
    derefs_downstream_transpose,
)


def _setup():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    d, d_ff, T = 6, 12, 5
    block = ProtoBlock(d, d_ff, T, seed=1)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((T, d))
    e_block = rng.standard_normal((T, d))
    ref, _ = autograd_ref(block, x, e_block)
    R_O = rng.standard_normal((d, d)) / np.sqrt(d)
    R_2 = rng.standard_normal((d, d_ff)) / np.sqrt(d)
    return block, x, ref, e_block, ActivationRoutedDFA(R_O=R_O, R_2=R_2)


def test_structural_check_passes_one_flags_positive_control():
    # ① uses R_O/R_2 (no transport); the bp_block_grads positive control uses Wo.T/W2.T and MUST
    # be flagged, else the structural check is unfalsifiable.
    assert derefs_downstream_transpose(ActivationRoutedDFA.block_grads) is False
    assert derefs_downstream_transpose(bp_block_grads) is True


def test_e_fixed_transpose_perturbation():
    # Perturb the ACTUAL Wo.T/W2.T inputs with e + forward cache FIXED. ① is invariant (reads
    # R_O/R_2); the transport-using positive control changes.
    block, x, ref, e_block, strat = _setup()
    one_change, bp_change = block_transpose_perturb_change(strat, block, x, ref["x1"], e_block)
    assert one_change < 1e-12
    assert bp_change > 1e-6
