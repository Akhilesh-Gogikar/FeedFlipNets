# tests/test_lm_value_path_exact.py
"""GUARD 1 — value-path-exactness of ① in the CAUSAL multi-block LM.

PLAN-BUG FIX: the plan's Task-4 test compares the CAUSAL M2b kernel (block_forward_np applies a
np.tril mask to A) against the M2 ProtoBlock / autograd_ref, which are NON-causal (full softmax, no
mask). Those legitimately differ in A, so that reference CANNOT pass (round-1 measured diff ~5.6).
The kernel is correct — the M2 reference is the wrong yardstick for a causal block.

Fix: build a tiny torch CAUSAL single-block whose forward matches block_forward_np exactly
(pre-LN, single head, masked_fill(triu(ones,1), -inf), LN with unbiased=False and eps=1e-5), and
assert ①'s value-exact dW_O / dW_2 equal torch.autograd.grad of the matching sublayer to < 1e-5.
The batched-consistency test compares the batched kernel to a loop over M2b's OWN causal kernel,
never M2's ProtoBlock.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from feedflipnets.core.lm import EPS, block_forward_np, one_block_grads  # noqa: E402

DT = torch.float64


def _rand_params(d, h, rng):
    """M2b block param dict, same shapes/scales as NumpyLM blocks."""
    s = 1.0 / math.sqrt(d)
    return dict(
        Wq=rng.standard_normal((d, d)) * s,
        Wk=rng.standard_normal((d, d)) * s,
        Wv=rng.standard_normal((d, d)) * s,
        Wo=rng.standard_normal((d, d)) * s,
        W1=rng.standard_normal((d, h)) / math.sqrt(d),
        W2=rng.standard_normal((h, d)) / math.sqrt(h),
    )


def _ln_torch(x):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + EPS)


def _causal_block_torch(P, x_np):
    """CAUSAL single-head pre-LN block in torch — mirrors block_forward_np exactly.

    Returns (Wo, W2, x1, x2) with grad tracking on Wo/W2 so autograd can produce the value-exact
    references for the two residual-additive weights.
    """
    d = x_np.shape[-1]
    x = torch.tensor(x_np, dtype=DT)  # (1,T,d); x is the residual, NOT differentiated here
    Wo = torch.tensor(P["Wo"], dtype=DT, requires_grad=True)
    W2 = torch.tensor(P["W2"], dtype=DT, requires_grad=True)
    Wq = torch.tensor(P["Wq"], dtype=DT)
    Wk = torch.tensor(P["Wk"], dtype=DT)
    Wv = torch.tensor(P["Wv"], dtype=DT)
    W1 = torch.tensor(P["W1"], dtype=DT)
    T = x.shape[1]
    y = _ln_torch(x)
    Q, K, V = y @ Wq, y @ Wk, y @ Wv
    S = (Q @ K.transpose(-2, -1)) / math.sqrt(d)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    S = S.masked_fill(mask, float("-inf"))
    A = torch.softmax(S, dim=-1)
    Ctx = A @ V
    x1 = x + Ctx @ Wo
    z2 = _ln_torch(x1)
    H = torch.relu(z2 @ W1)
    x2 = x1 + H @ W2
    return Wo, W2, x1, x2


def test_value_path_exact_dWo_dW2_multiblock():
    """① dW_O and dW_2 are value-EXACT vs a CAUSAL autograd reference for EVERY block in a stack.

    e_attn seeds dL/dx1 (attention residual), e_mlp seeds dL/dx2 (MLP residual). The value-exact
    grads dW_O = Ctxᵀ·e_attn and dW_2 = Hᵀ·e_mlp carry no transpose, so they must match autograd of
    the causal block to ~machine precision.
    """
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    d, h, T, N = 8, 16, 6, 3
    rng = np.random.default_rng(0)
    for bi in range(N):
        P = _rand_params(d, h, rng)
        x = rng.standard_normal((T, d))
        e_attn = rng.standard_normal((T, d))  # dL/dx1
        e_mlp = rng.standard_normal((T, d))  # dL/dx2

        Wo_t, W2_t, x1_t, x2_t = _causal_block_torch(P, x[None])
        # dW_O is the grad of x1 w.r.t Wo seeded by e_attn=dL/dx1 (attention sublayer only)
        (ref_Wo,) = torch.autograd.grad(
            x1_t, Wo_t, grad_outputs=torch.tensor(e_attn[None], dtype=DT), retain_graph=True
        )
        # dW_2 is the grad of x2 w.r.t W2 seeded by e_mlp=dL/dx2 (MLP sublayer only)
        (ref_W2,) = torch.autograd.grad(
            x2_t, W2_t, grad_outputs=torch.tensor(e_mlp[None], dtype=DT)
        )
        ref_Wo = ref_Wo.numpy()
        ref_W2 = ref_W2.numpy()

        R_O = rng.standard_normal((d, d)) / math.sqrt(d)
        R_2 = rng.standard_normal((d, h)) / math.sqrt(d)
        c = block_forward_np(P, x[None])
        g, _dx = one_block_grads(P, c, e_attn[None], e_mlp[None], R_O, R_2)

        err_wo = float(np.max(np.abs(g["Wo"] - ref_Wo)))
        err_w2 = float(np.max(np.abs(g["W2"] - ref_W2)))
        assert err_wo < 1e-5, ("Wo", bi, err_wo)
        assert err_w2 < 1e-5, ("W2", bi, err_w2)


def test_batched_one_block_matches_own_causal_kernel_per_sequence():
    """The batched (B,T,d) kernel == a loop over M2b's OWN causal kernel on each sequence (B=1).

    Consistency check for the batched einsums against the single-sequence path of the SAME causal
    math — not against M2's non-causal ProtoBlock (which would legitimately differ in A).
    """
    d, h, T, B = 8, 16, 6, 4
    rng = np.random.default_rng(3)
    P = _rand_params(d, h, rng)
    R_O = rng.standard_normal((d, d)) / math.sqrt(d)
    R_2 = rng.standard_normal((d, h)) / math.sqrt(d)
    x = rng.standard_normal((B, T, d))
    e_attn = rng.standard_normal((B, T, d))
    e_mlp = rng.standard_normal((B, T, d))

    c = block_forward_np(P, x)
    g_batched, dx_batched = one_block_grads(P, c, e_attn, e_mlp, R_O, R_2)

    # per-sequence loop over the SAME causal kernel; param grads sum over the batch
    g_loop = {k: np.zeros_like(v) for k, v in g_batched.items()}
    dx_loop = np.zeros_like(dx_batched)
    for b in range(B):
        cb = block_forward_np(P, x[b : b + 1])
        gb, dxb = one_block_grads(P, cb, e_attn[b : b + 1], e_mlp[b : b + 1], R_O, R_2)
        for k in g_loop:
            g_loop[k] += gb[k]
        dx_loop[b : b + 1] = dxb

    for k in g_batched:
        assert np.allclose(g_batched[k], g_loop[k], atol=1e-10), k
    assert np.allclose(dx_batched, dx_loop, atol=1e-10)
