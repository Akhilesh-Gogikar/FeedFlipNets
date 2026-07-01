# feedflipnets/core/lm.py
"""Stackable char-LM in batched NumPy (float64) + the lock-free ①+② / fixed-DFA training step (M2b).

The block math is the batched (B,T,d) restatement of the M2 single-sequence construction in
feedflipnets/core/{transformer,strategies}.py:
  - ① one_block_grads reuses A / causal softmax-Jac / LN-VJP / φ' EXACTLY; dW_O=Ctxᵀ·e_attn and
    dW_2=Hᵀ·e_mlp are value-EXACT; only W_Oᵀ→R_O and W_2ᵀ→R_2 are surrogated.
  - ② ghat_ctx_nodepert estimates dL/dCtx by antithetic node-perturbation via a block-FORWARD
    recompute of the local scalar loss — NO autograd.grad, NO W_Oᵀ (reads P["Wo"] forward only).
Torch is NOT imported here; the tuned-BP baseline lives in experiments/m2b_lm_ablation.py.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

Array = np.ndarray
EPS = 1e-5


def ln_np(x: Array) -> Tuple[Array, Array]:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, ddof=0, keepdims=True)
    sig = np.sqrt(var + EPS)
    return (x - mu) / sig, sig


def ln_vjp(x: Array, g: Array) -> Array:
    """VJP of LN(x) (no affine): dx = (1/σ)(g − mean(g) − x̂·mean(g·x̂))."""
    xhat, sig = ln_np(x)
    gm = g.mean(-1, keepdims=True)
    gxm = (g * xhat).mean(-1, keepdims=True)
    return (g - gm - xhat * gxm) / sig


_MASK: Dict[int, Array] = {}


def causal_mask(T: int) -> Array:
    if T not in _MASK:
        _MASK[T] = np.tril(np.ones((T, T)))
    return _MASK[T]


def block_forward_np(P: Dict[str, Array], x: Array) -> Dict[str, Array]:
    """Batched causal single-head pre-LN block forward. x:(B,T,d). Returns cache of data-maps."""
    B, T, d = x.shape
    y, _ = ln_np(x)
    Q = y @ P["Wq"]
    K = y @ P["Wk"]
    Vv = y @ P["Wv"]  # (B,T,d)
    S = np.einsum("btd,bsd->bts", Q, K) / math.sqrt(d)  # (B,T,T)
    m = causal_mask(T)
    S = np.where(m > 0, S, -1e30)
    S = S - S.max(-1, keepdims=True)
    A = np.exp(S) * m
    A = A / A.sum(-1, keepdims=True)  # (B,T,T)
    Ctx = np.einsum("bts,bsd->btd", A, Vv)  # (B,T,d)
    attn_out = Ctx @ P["Wo"]
    x1 = x + attn_out
    z2, _ = ln_np(x1)
    pre = z2 @ P["W1"]
    H = np.maximum(0.0, pre)
    phi = (pre > 0).astype(np.float64)
    M = H @ P["W2"]
    x2 = x1 + M
    return dict(x=x, y=y, Q=Q, K=K, V=Vv, A=A, Ctx=Ctx, x1=x1, z2=z2, pre=pre, H=H, phi=phi, x2=x2)


def softmax_jac_causal(A: Array, dA: Array) -> Array:
    """Batched row-softmax Jacobian: dS = A*(dA − <A,dA>_rows). A,dA:(B,T,T)."""
    dot = (A * dA).sum(-1, keepdims=True)
    return A * (dA - dot)


def one_block_grads(P, c, e_attn, e_mlp, R_O, R_2):
    """① A-routed, value-exact dW_O/dW_2, surrogate R_O/R_2. Returns (param grads summed over batch, dL/dx).

    e_attn ≈ dL/dx1 (attention broadcast), e_mlp ≈ dL/dx2 (MLP broadcast). Batched restatement of
    ActivationRoutedDFA.block_grads. Never dereferences W_Oᵀ / W_2ᵀ.
    """
    B, T, d = c["x"].shape
    # MLP sublayer: x2 = x1 + M ; e_mlp seeds dL/dx2
    dM = e_mlp
    dW2 = np.einsum("btk,btd->kd", c["H"], dM)  # EXACT
    dH = dM @ R_2  # surrogate R_2 for W_2ᵀ
    dpre = dH * c["phi"]  # φ' reused EXACTLY
    dW1 = np.einsum("btd,btk->dk", c["z2"], dpre)
    dz2 = dpre @ P["W1"].T
    dx1_from_mlp = ln_vjp(c["x1"], dz2) + dM  # dL/dx1 (LN2 VJP + residual)
    # attention sublayer: x1 = x + O ; e_attn seeds dL/dx1
    dO = e_attn
    dWo = np.einsum("btd,bte->de", c["Ctx"], dO)  # EXACT (no transport)
    dCtx = dO @ R_O  # surrogate R_O for W_Oᵀ (ONLY attn transport)
    dV = np.einsum("bts,btd->bsd", c["A"], dCtx)
    dA = np.einsum("btd,bsd->bts", dCtx, c["V"])
    dS = softmax_jac_causal(c["A"], dA)
    scale = 1.0 / math.sqrt(d)
    dQ = np.einsum("bts,bsd->btd", dS, c["K"]) * scale
    dK = np.einsum("bts,btd->bsd", dS, c["Q"]) * scale  # dK[s] = Σ_t dS[t,s]·Q[t]
    dWq = np.einsum("btd,bte->de", c["y"], dQ)
    dWk = np.einsum("btd,bte->de", c["y"], dK)
    dWv = np.einsum("btd,bte->de", c["y"], dV)
    dy = dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T
    grads = dict(Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo, W1=dW1, W2=dW2)
    dx = ln_vjp(c["x"], dy) + dx1_from_mlp  # total dL/d(block input)
    return grads, dx


def fixed_dfa_block_grads(P, c, e_attn, e_mlp, Bq, Bk, Bv, B1):
    """Fixed-DFA-in-a-block baseline: broadcast e to Q/K/V/pre via independent random matrices,
    IGNORING A entirely. dW_O/dW_2 still use the exact activations (Ctx, H)."""
    dWo = np.einsum("btd,bte->de", c["Ctx"], e_attn)
    dWq = np.einsum("btd,bte->de", c["y"], e_attn @ Bq)
    dWk = np.einsum("btd,bte->de", c["y"], e_attn @ Bk)
    dWv = np.einsum("btd,bte->de", c["y"], e_attn @ Bv)
    dpre = (e_attn @ B1) * c["phi"]
    dW1 = np.einsum("btd,btk->dk", c["z2"], dpre)
    dW2 = np.einsum("btk,btd->kd", c["H"], e_mlp)
    grads = dict(Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo, W1=dW1, W2=dW2)
    dx = e_mlp  # residual-dominated cheap input error for the baseline
    return grads, dx


def ghat_ctx_nodepert(P, c, e_top_block, rho, K_samp, rng):
    """② transport-free estimate of dL/dCtx: antithetic node-perturbation of Ctx using ONLY a
    block-FORWARD recompute of the local scalar loss L = <e_top_block, x2>. Never calls autograd;
    never reads W_Oᵀ (reads P["Wo"] forward only)."""
    Ctx0 = c["Ctx"]
    x_res = c["x"]

    def local_loss(Ctx_pert):
        attn_out = Ctx_pert @ P["Wo"]
        x1 = x_res + attn_out
        z2, _ = ln_np(x1)
        H = np.maximum(0.0, z2 @ P["W1"])
        x2 = x1 + H @ P["W2"]
        return (e_top_block * x2).sum(axis=(1, 2))  # per-sample scalar (B,)

    ghat = np.zeros_like(Ctx0)
    for _ in range(K_samp):
        xi = rng.standard_normal(Ctx0.shape) * rho
        lp = local_loss(Ctx0 + xi)
        lm = local_loss(Ctx0 - xi)
        coef = ((lp - lm) / (2.0 * rho * rho))[:, None, None]
        ghat += coef * xi
    return ghat / K_samp
