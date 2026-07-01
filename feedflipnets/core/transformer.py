"""torch-CPU pre-LN single-head decoder block + autograd exact-BP reference (M2 testbed).

Single sequence of ``T`` token rows, model dim ``d``, single head (head dim = d). Pre-LN:
    y = LN(x); Q=yWq; K=yWk; V=yWv; A=softmax(QKᵀ/√d); Ctx=AV; O=Ctx·Wo; x1 = x + O
    z2 = LN2(x1); H = relu(z2·W1); M = H·W2; x2 = x1 + M
autograd is used ONLY for the exact-BP reference/alignment target; the ① strategy stays explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import torch

DT = torch.float64
Array = np.ndarray


def _ln_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
    mu = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)


def _ln_np(x: Array, eps: float) -> Tuple[Array, Array]:
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, ddof=0, keepdims=True)
    sigma = np.sqrt(var + eps)
    return (x - mu) / sigma, sigma


def ln_vjp(x: Array, g_out: Array, eps: float) -> Array:
    """VJP of LN(x) w.r.t x (no affine). dx = (1/σ)(g − mean(g) − x̂·mean(g·x̂))."""
    xhat, sigma = _ln_np(x, eps)
    g_mean = g_out.mean(axis=1, keepdims=True)
    gx_mean = (g_out * xhat).mean(axis=1, keepdims=True)
    return (g_out - g_mean - xhat * gx_mean) / sigma


def softmax_jac_apply(a_row: Array, g_row: Array) -> Array:
    """Row-softmax Jacobian J_A = diag(a) − aaᵀ applied to g: a·(g − a·g). (J_A symmetric.)"""
    dot = float((a_row * g_row).sum())
    return a_row * (g_row - dot)


@dataclass
class ProtoBlock:
    d: int
    d_ff: int
    T: int
    seed: int = 0
    eps: float = 1e-5
    Wq: Array = field(default=None, repr=False)
    Wk: Array = field(default=None, repr=False)
    Wv: Array = field(default=None, repr=False)
    Wo: Array = field(default=None, repr=False)
    W1: Array = field(default=None, repr=False)
    W2: Array = field(default=None, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        s = 1.0 / np.sqrt(self.d)
        self.Wq = rng.standard_normal((self.d, self.d)) * s
        self.Wk = rng.standard_normal((self.d, self.d)) * s
        self.Wv = rng.standard_normal((self.d, self.d)) * s
        self.Wo = rng.standard_normal((self.d, self.d)) * s
        self.W1 = rng.standard_normal((self.d, self.d_ff)) / np.sqrt(self.d)
        self.W2 = rng.standard_normal((self.d_ff, self.d)) / np.sqrt(self.d_ff)

    def forward_torch(self, x_np: Array, requires_grad_params: bool = True):
        d = self.d
        x = torch.tensor(x_np, dtype=DT, requires_grad=True)
        params = {
            n: torch.tensor(getattr(self, n), dtype=DT, requires_grad=requires_grad_params)
            for n in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]
        }
        y = _ln_torch(x, self.eps)
        Q = y @ params["Wq"]
        K = y @ params["Wk"]
        V = y @ params["Wv"]
        S = (Q @ K.T) / np.sqrt(d)
        A = torch.softmax(S, dim=1)
        Ctx = A @ V
        O_out = Ctx @ params["Wo"]
        x1 = x + O_out
        z2 = _ln_torch(x1, self.eps)
        pre = z2 @ params["W1"]
        H = torch.relu(pre)
        M = H @ params["W2"]
        x2 = x1 + M
        cache = dict(
            x=x,
            y=y,
            Q=Q,
            K=K,
            V=V,
            S=S,
            A=A,
            Ctx=Ctx,
            O=O_out,
            x1=x1,
            z2=z2,
            pre=pre,
            H=H,
            M=M,
            x2=x2,
            params=params,
        )
        return x2, cache


def forward_np_cache(block: ProtoBlock, x_np: Array) -> Dict[str, Array]:
    """Recompute the block_cache in numpy (data-maps only), exactly matching forward_torch."""
    d = block.d
    x = x_np
    y, sigma_y = _ln_np(x, block.eps)
    Q = y @ block.Wq
    K = y @ block.Wk
    V = y @ block.Wv
    S = (Q @ K.T) / np.sqrt(d)
    S = S - S.max(axis=1, keepdims=True)
    A = np.exp(S)
    A = A / A.sum(axis=1, keepdims=True)
    Ctx = A @ V
    O_out = Ctx @ block.Wo
    x1 = x + O_out
    z2, sigma_z2 = _ln_np(x1, block.eps)
    pre = z2 @ block.W1
    H = np.maximum(0.0, pre)
    phi_mask = (pre > 0).astype(np.float64)
    return dict(
        x=x,
        y=y,
        sigma_y=sigma_y,
        Q=Q,
        K=K,
        V=V,
        A=A,
        Ctx=Ctx,
        O=O_out,
        x1=x1,
        z2=z2,
        sigma_z2=sigma_z2,
        pre=pre,
        H=H,
        phi_mask=phi_mask,
    )


def autograd_ref(block: ProtoBlock, x_np: Array, e_block_np: Array):
    """Exact-BP reference: seed dL/dx2 = e_block, backprop via autograd. float64.

    Returns {Wq,Wk,Wv,Wo,W1,W2, Ctx, O, x1} where x1 = dL/dx1 (the attention broadcast target).
    """
    _x2, cache = block.forward_torch(x_np, requires_grad_params=True)
    e_block = torch.tensor(e_block_np, dtype=DT)
    names = ["Wq", "Wk", "Wv", "Wo", "W1", "W2", "Ctx", "O", "x1"]
    targets = [cache["params"][n] for n in names[:6]] + [cache["Ctx"], cache["O"], cache["x1"]]
    grads = torch.autograd.grad(_x2, targets, grad_outputs=e_block, retain_graph=True)
    return {n: g.detach().numpy() for n, g in zip(names, grads)}, cache
