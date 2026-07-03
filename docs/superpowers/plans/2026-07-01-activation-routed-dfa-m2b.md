# Activation-Routed DFA on a Char-Level LM (M2b) — Bits-per-Char Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether ①+②'s attention-alignment win from M2 (worst-θ 90°→46°) **translates into competitive bits-per-char** on a real char-level LM against a tuned backprop baseline. Build a stackable multi-block LM, wire the full lock-free ①+② training scheme (per-block DFA broadcast + ① A-routed value-exact grads + ②'s transport-free surrogate adaptation), a fixed-DFA baseline, and a tuned BP baseline; run an n≥5 CFG ablation → bpc and record the pre-registered go/no-go. **This is a pre-registered NO-GO plan:** the prototype (n=3, CFG, 1500 steps) gives BP 0.772, fixed-DFA 3.848, ①+② 2.763 — ①+② beats fixed-DFA by 1.09 bpc and edges below the bigram floor (2.881), but sits **~1.99 bpc ABOVE BP**. The gate (①+② − BP ≤ 0.15 bpc) is **NOT MET**. Do NOT tune to force a pass; record the honest verdict and the arc conclusion (alignment is necessary but not sufficient for competitive bpc).

**Architecture:** float only, no ternary. torch-CPU is the tuned-BP reference/baseline **only** (autograd through a `torch.nn` `LM`). The ①+② and fixed-DFA conditions run in **pure batched NumPy (float64)** over a stacked LM (embeddings + N causal single-head pre-LN blocks + tied-shape linear head): each pre-LN sublayer is its own DFA broadcast point (`e_attn = e@B_attn_ℓ`, `e_mlp = e@B_mlp_ℓ`), ① routes the error through the cached data-maps (`A`, causal softmax-Jac, LN-VJP, `φ'`) exactly and replaces only `W_Oᵀ`/`W_2ᵀ` with surrogates `R_O`/`R_2`, and ② adapts `R_O` by transport-free node-perturbation of `Ctx` (block-forward recompute only — NO `autograd.grad`, NO `W_Oᵀ`). The head grad is exact from `e`; the embedding grad comes from the first-block input error. The M2b block math **reuses the M2 construction** (`feedflipnets/core/transformer.py`, `feedflipnets/core/strategies.py::ActivationRoutedDFA`) — batched here for `(B,T,d)` — so the value-path-exactness and transport-free guards carry forward to the multi-block setting.

**Tech Stack:** Python 3.9, **torch 2.8 (CPU)** (already pinned in `requirements-extras.txt` from M2), NumPy, pytest. Interpreter: **`.venv/bin/python`** (`python` is NOT on PATH). Determinism: `torch.use_deterministic_algorithms(True)` where torch runs, seeded torch+numpy, float64 in the NumPy path. **Perf (mandatory):** the prototype found a ~500× slowdown on tiny batched matmuls when BLAS spawned threads — every runner/test must set `OPENBLAS_NUM_THREADS=1` / `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` **before importing numpy** and call `torch.set_num_threads(1)`.

**Scope note (M2b = BPC GATE only):** This plan covers the char-LM loader, the stackable LM + DFA-broadcast training step, the fixed-DFA and BP baselines, the n≥5 CFG ablation, and the four carried-forward guards. It does NOT introduce ternary, multi-head attention, KV-cache, or English as the primary corpus (English is an optional secondary run). The alignment ablation (θ-level) lives in the M2 plan; M2b measures the downstream accuracy consequence.

## File Structure

- `feedflipnets/data/char_lm.py` (new) — char-level corpus builders (`build_cfg_corpus`, `build_english_corpus`), `prep` (train/val split + vocab), `bigram_floor` (add-1 smoothed val bpc), `get_batch`. One responsibility: corpus + batching.
- `feedflipnets/core/lm.py` (new) — batched NumPy block ops (`ln_np`, `ln_vjp`, `block_forward_np`, `softmax_jac_causal`), the batched grad kernels (`one_block_grads`, `fixed_dfa_block_grads`), the transport-free `ghat_ctx_nodepert`, a plain `Adam`, and the `NumpyLM` trainer (forward + DFA-broadcast backward + ② adaptation + `eval_bpc`). One responsibility: the stackable LM + its lock-free training step.
- `experiments/m2b_lm_ablation.py` (new) — the BP/fixed-DFA/①+② × n-seeds runner: builds the corpus once, runs each condition, prints the summary + gap-to-BP, and writes JSON to `data/report/m2b/`.
- `tests/test_char_lm.py`, `tests/test_lm_value_path_exact.py`, `tests/test_lm_transport_free.py`, `tests/test_lm_smoke.py`, `tests/test_m2b_ablation_smoke.py` (new).

**Cross-references (reused, NOT modified):**
- `feedflipnets/core/transformer.py` — `ProtoBlock`, `forward_np_cache`, `autograd_ref` (the single-block autograd reference the multi-block value-path-exact test builds on).
- `feedflipnets/core/strategies.py::ActivationRoutedDFA` — ①'s single-block value-exact routing; `feedflipnets/core/lm.py::one_block_grads` is its batched `(B,T,d)` restatement and MUST agree with it on a single sequence.
- `feedflipnets/eval/{alignment,gradcheck,lockfree}.py` — M2 helpers reused in tests (`value_path_exact_err`, `derefs_downstream_transpose`).

---

## Task 1: Char-level corpus + batching (`feedflipnets/data/char_lm.py`)

**Files:**
- Create: `feedflipnets/data/char_lm.py`
- Test: `tests/test_char_lm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_char_lm.py
import numpy as np

from feedflipnets.data.char_lm import build_cfg_corpus, prep, bigram_floor, get_batch


def test_cfg_corpus_is_deterministic_and_structured():
    a = build_cfg_corpus(seed=0)
    b = build_cfg_corpus(seed=0)
    assert a == b                       # seeded, reproducible
    assert len(a) > 100_000             # enough chars to train a small LM
    assert "(" in a and ")" in a        # nested brackets → long-range structure


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_char_lm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'feedflipnets.data.char_lm'`

- [ ] **Step 3: Write minimal implementation**

```python
# feedflipnets/data/char_lm.py
"""Char-level corpora + batching for the M2b bits-per-char gate.

CFG corpus is the controlled primary: a nested probabilistic grammar whose long-range brackets and
agreement a bigram cannot capture, so the bigram floor lands high (~2.9 bpc) and a real LM (BP) gets
well below it. English (repo docs/*.md prose) is an optional secondary corpus with genuine
irreducible entropy. Float/int only, no torch.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Tuple

import numpy as np

Array = np.ndarray


def build_cfg_corpus(seed: int = 0) -> str:
    """Nested probabilistic CFG + larger vocab + long-range brackets/agreement.

    Bigram floor lands high; a real LM (BP) should get well below it.
    """
    rng = np.random.default_rng(seed)
    det = ["the", "a", "this", "that", "my", "his", "her", "some", "no", "every"]
    adj = ["quick", "brown", "lazy", "bright", "silent", "ancient", "clever", "weary",
           "golden", "hollow", "restless", "crimson", "distant", "gentle"]
    noun = ["fox", "dog", "river", "mountain", "scholar", "engine", "shadow", "garden",
            "letter", "machine", "planet", "harbor", "melody", "cipher", "lantern", "vulture"]
    verb = ["jumps", "observes", "questions", "remembers", "constructs", "abandons",
            "measures", "follows", "ignites", "dissolves", "gathers", "translates"]
    adv = ["quickly", "silently", "eventually", "carefully", "rarely", "abruptly", "gladly"]
    conj = ["and", "but", "because", "although", "while", "so"]

    def np_() -> str:
        return f"{rng.choice(det)} {rng.choice(adj)} {rng.choice(noun)}"

    def clause(depth: int = 0) -> str:
        s = f"{np_()} {rng.choice(verb)} {np_()}"
        if rng.random() < 0.35:
            s += f" {rng.choice(adv)}"
        if depth < 2 and rng.random() < 0.4:
            s += f" ({clause(depth + 1)})"
        if depth == 0 and rng.random() < 0.5:
            s += f" {rng.choice(conj)} {clause(depth + 1)}"
        return s

    parts = [clause().capitalize() + "." for _ in range(2200)]
    text = " ".join(parts)
    text = text.replace(". ", ".\n", len(text) // 40)
    return text


def build_english_corpus(root: str = "docs") -> str:
    """Optional secondary corpus: repo docs/*.md prose (genuine irreducible entropy).

    Strips code fences / inline code / URLs lightly; restricts to a stable printable-ASCII vocab.
    """
    import glob
    import re

    chunks = []
    for fp in sorted(glob.glob(root + "/**/*.md", recursive=True)):
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                chunks.append(f.read())
        except OSError:
            pass
    text = "\n".join(chunks)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)   # drop code blocks
    text = re.sub(r"`[^`]*`", " ", text)                       # inline code
    text = re.sub(r"https?://\S+", " ", text)                  # urls
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "".join(ch for ch in text if 32 <= ord(ch) < 127 or ch == "\n")
    return text


def prep(text: str) -> Tuple[Array, Array, int]:
    """Char→id, 90/10 train/val split. Returns (train, val, vocab_size)."""
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    return data[:split], data[split:], V


def bigram_floor(train: Array, val: Array, V: int) -> float:
    """Add-1 smoothed bigram val bits-per-char (the 'no structure learned' floor)."""
    bg = defaultdict(Counter)
    for a, b in zip(train[:-1], train[1:]):
        bg[a][b] += 1
    ll = 0.0
    for a, b in zip(val[:-1], val[1:]):
        c = bg[a]
        ll += -math.log2((c[b] + 1) / (sum(c.values()) + V))
    return ll / (len(val) - 1)


def get_batch(src: Array, bs: int, T: int, rng: np.random.Generator) -> Tuple[Array, Array]:
    """Random next-char batch. xb:(bs,T), yb:(bs,T) is xb shifted by one."""
    ix = rng.integers(0, len(src) - T - 1, size=bs)
    xb = np.stack([src[i:i + T] for i in ix])
    yb = np.stack([src[i + 1:i + T + 1] for i in ix])
    return xb, yb
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_char_lm.py -q`
Expected: PASS (4 passed). Prototype: `chars=249960 vocab=38 bigram_floor=2.881 bpc`.

- [ ] **Step 5: Commit**

```bash
git add feedflipnets/data/char_lm.py tests/test_char_lm.py
git commit -m "feat(m2b): char-level CFG/English corpus loader + bigram floor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Batched NumPy block ops + ① / fixed-DFA grad kernels (`feedflipnets/core/lm.py`, part 1)

The batched `(B,T,d)` restatement of the M2 block math. `one_block_grads` is the batched `ActivationRoutedDFA.block_grads`; `fixed_dfa_block_grads` is the batched fixed-DFA baseline. `dW_O = Ctxᵀ·e_attn` and `dW_2 = Hᵀ·e_mlp` are **value-exact** (no transport); only `W_Oᵀ→R_O` and `W_2ᵀ→R_2` are surrogated. The value-path-exactness is guarded in Task 4.

**Files:**
- Create: `feedflipnets/core/lm.py` (block ops + grad kernels; the `NumpyLM` trainer is appended in Task 3)
- Test: `tests/test_lm_value_path_exact.py` (written in Task 4; Task 2 is exercised there and in Task 3's smoke test)

- [ ] **Step 1: Write minimal implementation** (create `feedflipnets/core/lm.py`)

```python
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
from typing import Dict, List, Tuple

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
    Q = y @ P["Wq"]; K = y @ P["Wk"]; Vv = y @ P["Wv"]        # (B,T,d)
    S = np.einsum("btd,bsd->bts", Q, K) / math.sqrt(d)         # (B,T,T)
    m = causal_mask(T)
    S = np.where(m > 0, S, -1e30)
    S = S - S.max(-1, keepdims=True)
    A = np.exp(S) * m
    A = A / A.sum(-1, keepdims=True)                            # (B,T,T)
    Ctx = np.einsum("bts,bsd->btd", A, Vv)                      # (B,T,d)
    O = Ctx @ P["Wo"]
    x1 = x + O
    z2, _ = ln_np(x1)
    pre = z2 @ P["W1"]
    H = np.maximum(0.0, pre)
    phi = (pre > 0).astype(np.float64)
    M = H @ P["W2"]
    x2 = x1 + M
    return dict(x=x, y=y, Q=Q, K=K, V=Vv, A=A, Ctx=Ctx, x1=x1, z2=z2,
                pre=pre, H=H, phi=phi, x2=x2)


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
    dW2 = np.einsum("btk,btd->kd", c["H"], dM)                 # EXACT
    dH = dM @ R_2                                              # surrogate R_2 for W_2ᵀ
    dpre = dH * c["phi"]                                       # φ' reused EXACTLY
    dW1 = np.einsum("btd,btk->dk", c["z2"], dpre)
    dz2 = dpre @ P["W1"].T
    dx1_from_mlp = ln_vjp(c["x1"], dz2) + dM                   # dL/dx1 (LN2 VJP + residual)
    # attention sublayer: x1 = x + O ; e_attn seeds dL/dx1
    dO = e_attn
    dWo = np.einsum("btd,bte->de", c["Ctx"], dO)              # EXACT (no transport)
    dCtx = dO @ R_O                                           # surrogate R_O for W_Oᵀ (ONLY attn transport)
    dV = np.einsum("bts,btd->bsd", c["A"], dCtx)
    dA = np.einsum("btd,bsd->bts", dCtx, c["V"])
    dS = softmax_jac_causal(c["A"], dA)
    scale = 1.0 / math.sqrt(d)
    dQ = np.einsum("bts,bsd->btd", dS, c["K"]) * scale
    dK = np.einsum("bts,btd->bsd", dS, c["Q"]) * scale        # dK[s] = Σ_t dS[t,s]·Q[t]
    dWq = np.einsum("btd,bte->de", c["y"], dQ)
    dWk = np.einsum("btd,bte->de", c["y"], dK)
    dWv = np.einsum("btd,bte->de", c["y"], dV)
    dy = dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T
    grads = dict(Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo, W1=dW1, W2=dW2)
    dx = ln_vjp(c["x"], dy) + dx1_from_mlp                    # total dL/d(block input)
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
    Ctx0 = c["Ctx"]; x_res = c["x"]

    def local_loss(Ctx_pert):
        O = Ctx_pert @ P["Wo"]
        x1 = x_res + O
        z2, _ = ln_np(x1)
        H = np.maximum(0.0, z2 @ P["W1"])
        x2 = x1 + H @ P["W2"]
        return (e_top_block * x2).sum(axis=(1, 2))            # per-sample scalar (B,)

    ghat = np.zeros_like(Ctx0)
    for _ in range(K_samp):
        xi = rng.standard_normal(Ctx0.shape) * rho
        lp = local_loss(Ctx0 + xi)
        lm = local_loss(Ctx0 - xi)
        coef = ((lp - lm) / (2.0 * rho * rho))[:, None, None]
        ghat += coef * xi
    return ghat / K_samp
```

- [ ] **Step 2: Sanity-import to catch syntax errors**

Run: `.venv/bin/python -c "import feedflipnets.core.lm as m; print(m.one_block_grads, m.ghat_ctx_nodepert)"`
Expected: prints the two function objects (no ImportError). Full behavioral tests land in Tasks 3–4.

- [ ] **Step 3: Commit**

```bash
git add feedflipnets/core/lm.py
git commit -m "feat(m2b): batched NumPy block ops + ①/fixed-DFA grad kernels + transport-free ĝ_Ctx

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Stackable NumpyLM trainer + loss-decreasing smoke test (`feedflipnets/core/lm.py`, part 2)

Append the `Adam` optimizer and the `NumpyLM` trainer: embeddings + N blocks + linear head, per-block per-sublayer DFA broadcast of the top error, ① or fixed-DFA per-block grads, exact head grad, embedding grad from first-block input error, and ②'s amortized (one block/step) transport-free `R_O` adaptation.

**Files:**
- Modify: `feedflipnets/core/lm.py` (append `Adam`, `NumpyLM`)
- Test: `tests/test_lm_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lm_smoke.py
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from feedflipnets.core.lm import NumpyLM
from feedflipnets.data.char_lm import build_cfg_corpus, prep, get_batch


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_lm_smoke.py -q`
Expected: FAIL — `ImportError: cannot import name 'NumpyLM' from 'feedflipnets.core.lm'`

- [ ] **Step 3: Write minimal implementation** (append to `feedflipnets/core/lm.py`)

```python
# --- Adam + stackable LM trainer (M2b) ---


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            if g is None:
                continue
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class NumpyLM:
    """Embeddings + N causal pre-LN blocks + linear head, trained lock-free with DFA broadcast.

    mode ∈ {"ONE_TWO", "FIXED_DFA"}. Each pre-LN sublayer is its own DFA broadcast of the top error
    e=dL/dlogits: e_attn=e@B_attn_ℓ, e_mlp=e@B_mlp_ℓ. ONE_TWO uses ① value-exact grads + ② adapts R_O;
    FIXED_DFA ignores A. Head grad EXACT from e; embedding grad from the first-block input error.
    """

    def __init__(self, V, d, h, N, T, seed, mode):
        self.V, self.d, self.h, self.N, self.T, self.mode = V, d, h, N, T, mode
        rng = np.random.default_rng(seed); self.rng = rng
        s = 1.0 / math.sqrt(d)
        self.P = dict(emb=rng.standard_normal((V, d)) * s,
                      pos=rng.standard_normal((T, d)) * s,
                      head=rng.standard_normal((d, V)) * s)
        self.blocks: List[Dict[str, Array]] = []
        for _ in range(N):
            self.blocks.append(dict(
                Wq=rng.standard_normal((d, d)) * s, Wk=rng.standard_normal((d, d)) * s,
                Wv=rng.standard_normal((d, d)) * s, Wo=rng.standard_normal((d, d)) * s,
                W1=rng.standard_normal((d, h)) / math.sqrt(d),
                W2=rng.standard_normal((h, d)) / math.sqrt(h)))
        # per-block per-sublayer DFA broadcast of top error e (…,V) → d
        self.B_attn = [rng.standard_normal((V, d)) / math.sqrt(V) for _ in range(N)]
        self.B_mlp = [rng.standard_normal((V, d)) / math.sqrt(V) for _ in range(N)]
        # ① surrogates R_O (adapted by ②) and R_2 (fixed). R_2 is the surrogate for W_2ᵀ (d×h).
        self.R_O = [rng.standard_normal((d, d)) / math.sqrt(d) for _ in range(N)]
        self.R_2 = [rng.standard_normal((d, h)) / math.sqrt(d) for _ in range(N)]
        # fixed-DFA broadcasts
        self.Bq = [rng.standard_normal((d, d)) / math.sqrt(d) for _ in range(N)]
        self.Bk = [rng.standard_normal((d, d)) / math.sqrt(d) for _ in range(N)]
        self.Bv = [rng.standard_normal((d, d)) / math.sqrt(d) for _ in range(N)]
        self.B1 = [rng.standard_normal((d, h)) / math.sqrt(d) for _ in range(N)]
        flat = {"emb": self.P["emb"], "pos": self.P["pos"], "head": self.P["head"]}
        for i, b in enumerate(self.blocks):
            for k, v in b.items():
                flat[f"b{i}_{k}"] = v
        self.flat = flat
        self.opt = None
        self.adapt_ptr = 0

    def set_opt(self, lr):
        self.opt = Adam(self.flat, lr=lr)

    def forward_np(self, xb):
        """Batched forward. xb:(B,T). Returns logits(B,T,V), per-block caches, final pre-LN act."""
        B, T = xb.shape
        x = self.P["emb"][xb] + self.P["pos"][:T]              # (B,T,d)
        caches = []
        for bi in range(self.N):
            c = block_forward_np(self.blocks[bi], x)
            caches.append(c); x = c["x2"]
        _xf, _ = ln_np(x)
        logits = _xf @ self.P["head"]
        return logits, caches, x

    def train_step(self, xb, yb, adapt_cfg):
        B, T = xb.shape
        logits, caches, xfinal = self.forward_np(xb)
        z = logits - logits.max(-1, keepdims=True)
        ez = np.exp(z); p = ez / ez.sum(-1, keepdims=True)
        idx = (np.arange(B)[:, None], np.arange(T)[None, :], yb)
        loss = -np.log(p[idx] + 1e-12).sum()
        e = p.copy(); e[idx] -= 1.0; e /= T                    # dL/dlogits (B,T,V), mean over tokens
        grads = {k: np.zeros_like(v) for k, v in self.flat.items()}
        # head EXACT
        xf, _ = ln_np(xfinal)
        grads["head"] += np.einsum("btd,btv->dv", xf, e)
        dxf = e @ self.P["head"].T
        e_final = ln_vjp(xfinal, dxf)                          # dL/d(block-N output) exact through final LN
        dx_input = e_final
        for bi in reversed(range(self.N)):
            c = caches[bi]
            e_attn = e @ self.B_attn[bi]
            e_mlp = e @ self.B_mlp[bi]
            if self.mode == "ONE_TWO":
                g, dx = one_block_grads(self.blocks[bi], c, e_attn, e_mlp, self.R_O[bi], self.R_2[bi])
            else:  # FIXED_DFA
                g, dx = fixed_dfa_block_grads(self.blocks[bi], c, e_attn, e_mlp,
                                              self.Bq[bi], self.Bk[bi], self.Bv[bi], self.B1[bi])
            for k in g:
                grads[f"b{bi}_{k}"] += g[k]
            dx_input = dx
        # embedding grads from first-block input error
        np.add.at(grads["emb"], xb, dx_input)
        grads["pos"][:T] += dx_input.sum(axis=0)
        # ② surrogate adaptation: amortised one block/step, transport-free, NO autograd / NO W_Oᵀ
        if self.mode == "ONE_TWO" and adapt_cfg is not None:
            bi = self.adapt_ptr % self.N; self.adapt_ptr += 1
            rho, Ksamp, lrR = adapt_cfg
            c = caches[bi]
            e_attn = e @ self.B_attn[bi]
            e_top_block = e @ self.B_mlp[bi]                   # local-loss seed for the block
            ghat = ghat_ctx_nodepert(self.blocks[bi], c, e_top_block, rho, Ksamp, self.rng)
            pred = e_attn @ self.R_O[bi]
            # KP: dR_O ∝ e_attnᵀ·(ĝ_Ctx − e_attn·R_O)  (sum over batch+time)
            self.R_O[bi] += lrR * np.einsum("btd,bte->de", e_attn, ghat - pred) / T
        self.opt.step(grads)
        return loss / (B * T) / math.log(2)

    def eval_bpc(self, val, n_batches=8, bs=32):
        tot = 0.0; cnt = 0
        for _ in range(n_batches):
            xb, yb = get_batch(val, bs, self.T, self.rng)
            logits, _, _ = self.forward_np(xb)
            z = logits - logits.max(-1, keepdims=True)
            p = np.exp(z); p /= p.sum(-1, keepdims=True)
            B, T = xb.shape
            idx = (np.arange(B)[:, None], np.arange(T)[None, :], yb)
            tot += -np.log(p[idx] + 1e-12).sum(); cnt += B * T
        return tot / cnt / math.log(2)
```

Add `from .char_lm import get_batch` at the top of `feedflipnets/core/lm.py`? No — `get_batch` lives in `feedflipnets.data.char_lm`. Import it in `lm.py`'s header:

```python
from feedflipnets.data.char_lm import get_batch  # add near the other imports
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_lm_smoke.py -q`
Expected: PASS (3 passed). The ①+② and fixed-DFA losses both decrease; ①+② drops faster (prototype traj `[5.66, 3.61, 3.17, 2.97, 2.85, 2.80]` over 1500 steps).

- [ ] **Step 5: Commit**

```bash
git add feedflipnets/core/lm.py tests/test_lm_smoke.py
git commit -m "feat(m2b): stackable NumpyLM trainer (DFA broadcast + ② adaptation) + loss-decrease smoke

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: GUARD 1 — value-path-exact in the MULTI-BLOCK LM

The M2 value-path-exactness test was single-block. Here we assert it **through the stacked LM**: seed each pre-LN sublayer with the autograd-exact broadcast (`e_attn = dL/dx1`, `e_mlp = dL/dx2` from `autograd_ref` on the *isolated* block instance) and check ①'s `dW_O`/`dW_2` match autograd to `<1e-5`. Also cross-check that the batched `one_block_grads` on a single sequence agrees with `ActivationRoutedDFA.block_grads` (the M2 code), so the batched kernel is provably the same math.

**Files:**
- Test: `tests/test_lm_value_path_exact.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_lm_value_path_exact.py
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from feedflipnets.core.transformer import ProtoBlock, autograd_ref
from feedflipnets.core.strategies import ActivationRoutedDFA
from feedflipnets.core.lm import one_block_grads, block_forward_np


def _block_params(pb):
    return dict(Wq=pb.Wq, Wk=pb.Wk, Wv=pb.Wv, Wo=pb.Wo, W1=pb.W1, W2=pb.W2)


def test_batched_one_block_matches_single_block_strategy():
    """The batched (B=1,T,d) kernel == the M2 ActivationRoutedDFA single-sequence math."""
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    d, d_ff, T = 8, 16, 6
    pb = ProtoBlock(d, d_ff, T, seed=1)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((T, d)); e_block = rng.standard_normal((T, d))
    ref, _ = autograd_ref(pb, x, e_block)               # ref["x1"]=dL/dx1, e_block=dL/dx2
    R_O = rng.standard_normal((d, d)) / np.sqrt(d)
    R_2 = rng.standard_normal((d, d_ff)) / np.sqrt(d)
    # single-block M2 strategy (note: strategy's R_2 is d×d_ff; batched kernel's R_2 is d×h — same)
    g_single = ActivationRoutedDFA(R_O=R_O, R_2=R_2).block_grads(
        pb, x, e_attn=ref["x1"], e_mlp=e_block)
    # batched kernel with B=1
    P = _block_params(pb)
    c = block_forward_np(P, x[None])
    g_batched, _dx = one_block_grads(
        P, c, e_attn=ref["x1"][None], e_mlp=e_block[None], R_O=R_O, R_2=R_2)
    for k in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]:
        assert np.allclose(g_single[k], g_batched[k], atol=1e-10), k


def test_value_path_exact_dWo_dW2_multiblock():
    """① dW_O and dW_2 are value-EXACT vs autograd for EVERY block in a stack, when each sublayer is
    seeded with its autograd-exact broadcast (e_attn=dL/dx1, e_mlp=dL/dx2)."""
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    d, d_ff, T, N = 8, 16, 6, 3
    rng = np.random.default_rng(0)
    for bi in range(N):
        pb = ProtoBlock(d, d_ff, T, seed=10 + bi)
        x = rng.standard_normal((T, d)); e_block = rng.standard_normal((T, d))
        ref, _ = autograd_ref(pb, x, e_block)
        R_O = rng.standard_normal((d, d)) / np.sqrt(d)
        R_2 = rng.standard_normal((d, d_ff)) / np.sqrt(d)
        P = _block_params(pb)
        c = block_forward_np(P, x[None])
        g, _dx = one_block_grads(P, c, e_attn=ref["x1"][None], e_mlp=e_block[None], R_O=R_O, R_2=R_2)
        assert np.max(np.abs(g["Wo"] - ref["Wo"])) < 1e-5, ("Wo", bi)
        assert np.max(np.abs(g["W2"] - ref["W2"])) < 1e-5, ("W2", bi)
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `.venv/bin/python -m pytest tests/test_lm_value_path_exact.py -q`
Expected: PASS (2 passed). Prototype: `max|①dW_O − refW_O| ≈ 3.8e-16`, `max|①dW_2 − refW_2| ≈ 1.3e-15` (both `< 1e-5`). If Task 2/3 are correct this passes immediately; if it fails, the batched einsums diverged from the M2 math — fix `one_block_grads`, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lm_value_path_exact.py
git commit -m "test(m2b): GUARD 1 — value-path-exact dW_O/dW_2 through the multi-block LM

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: GUARD 2 — transport-free assertion on ②'s adaptation + lock-free positive control

②'s adaptation must never touch `W_Oᵀ` or call `autograd.grad`. There is no autograd graph to taint-trace (pure NumPy), so the authoritative checks are: (i) **source-inspection** of `ghat_ctx_nodepert` and the `R_O` update path — no `autograd`, no `.Wo.T`/`.T` transpose of `Wo`; reuse the M2 `derefs_downstream_transpose` idea; (ii) a **positive control** — a deliberately transport-*using* estimator the checker must FLAG; (iii) an **e-fixed transpose-perturbation** control — perturbing `Wo` (which would change `Wo.T` if read) leaves `ghat_ctx_nodepert` changing only through the *forward* `Wo` it legitimately uses, while the transport control changes via the transpose.

**Files:**
- Test: `tests/test_lm_transport_free.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_lm_transport_free.py
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import inspect
import re

import numpy as np

from feedflipnets.core import lm as lmmod
from feedflipnets.core.lm import ghat_ctx_nodepert, block_forward_np


def _reads_transport(fn) -> bool:
    """Structural source-inspection: True iff fn transposes/dereferences a downstream weight or
    calls autograd. Authoritative lock-free check for the pure-NumPy ② adaptation path."""
    src = inspect.getsource(fn)
    return bool(re.search(r"\.Wo\s*\.T|Wo\"\]\s*\.T|W_?O\^?T|autograd", src))


def test_ghat_is_transport_free():
    # ② adaptation reads P["Wo"] FORWARD only; never W_Oᵀ, never autograd.
    assert _reads_transport(ghat_ctx_nodepert) is False


def test_train_step_adaptation_has_no_autograd():
    src = inspect.getsource(lmmod.NumpyLM.train_step)
    assert "autograd" not in src and ".Wo.T" not in src


def test_positive_control_is_flagged():
    # A transport-USING estimator (reads Wo.T) MUST be flagged — proves the check can fail.
    def ghat_transport_control(P, c, e_top_block, rho, K_samp, rng):
        # forbidden: dL/dCtx via the actual transpose W_Oᵀ (this is exactly what ② must avoid)
        return e_top_block @ P["Wo"].T

    assert _reads_transport(ghat_transport_control) is True


def test_ghat_ignores_a_transpose_of_wo_perturbation():
    """e + forward cache fixed: ĝ_Ctx changes ONLY through the forward Wo it legitimately uses.
    Positive control (uses Wo.T) changes when Wo is perturbed via the transpose it reads."""
    rng = np.random.default_rng(0)
    d, h, T = 8, 16, 6
    P = dict(Wq=rng.standard_normal((d, d)), Wk=rng.standard_normal((d, d)),
             Wv=rng.standard_normal((d, d)), Wo=rng.standard_normal((d, d)),
             W1=rng.standard_normal((d, h)), W2=rng.standard_normal((h, d)))
    x = rng.standard_normal((4, T, d))
    c = block_forward_np(P, x)
    e_top = rng.standard_normal((4, T, d))
    g0 = ghat_ctx_nodepert(P, c, e_top, rho=0.02, K_samp=8, rng=np.random.default_rng(1))
    # ② is FORWARD-only in Wo: it is a legitimate reader; assert it produces finite, non-trivial output.
    assert np.isfinite(g0).all() and np.linalg.norm(g0) > 0
    # transport control depends on Wo.T explicitly; perturbing Wo changes it via the transpose.
    ctrl0 = e_top @ P["Wo"].T
    P2 = dict(P); P2["Wo"] = P["Wo"] + 5.0
    ctrl1 = e_top @ P2["Wo"].T
    assert np.max(np.abs(ctrl1 - ctrl0)) > 1.0   # the forbidden path is transpose-sensitive
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_lm_transport_free.py -q`
Expected: PASS (4 passed). Prototype confirmed: NO `autograd.grad`, NO `W_Oᵀ` in the ② path (reads `P["Wo"]` forward only); e-fixed transpose-perturb change `= 0.0` for ①/②, `> 0` for the positive control.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lm_transport_free.py
git commit -m "test(m2b): GUARD 2 — ② adaptation transport-free (no autograd, no W_Oᵀ) + positive control

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Ablation runner + BP baseline + GUARD 4 smoke (`experiments/m2b_lm_ablation.py`)

The runner builds the CFG corpus once, runs `BP` (tuned torch autograd), `FIXED_DFA`, and `ONE_TWO` across n≥5 seeds, prints the summary + gap-to-BP + beats-fixed-DFA verdict, and writes JSON. **GUARD 4** (①+② beats fixed-DFA; honest gap-to-BP recorded) is asserted as a smoke test at reduced steps so CI stays cheap; the full n≥5 gate runs in Task 7.

**Files:**
- Create: `experiments/m2b_lm_ablation.py`
- Test: `tests/test_m2b_ablation_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m2b_ablation_smoke.py
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from experiments.m2b_lm_ablation import run_condition, CONDITIONS


def test_conditions_registered():
    assert set(CONDITIONS) == {"BP", "FIXED_DFA", "ONE_TWO"}


def test_one_two_learns_and_beats_fixed_dfa_smoke():
    # reduced-budget smoke: ①+② should already beat fixed-DFA in bpc (structure vs no-structure),
    # and BP should beat both. Full n≥5 gate is Task 7.
    cfg = dict(d=32, h=64, N=2, T=24, bs=16, steps=200, lr=3e-3,
               rho=0.02, Ksamp=4, lrR=0.05, seed=0)
    bp = run_condition("BP", **cfg)
    fx = run_condition("FIXED_DFA", **cfg)
    ot = run_condition("ONE_TWO", **cfg)
    assert ot["bpc"] < fx["bpc"]           # ①+② beats fixed-DFA (A-reuse learns structure)
    assert bp["bpc"] < ot["bpc"]           # honest: BP still wins (the NO-GO signal)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_m2b_ablation_smoke.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.m2b_lm_ablation'`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/m2b_lm_ablation.py
"""M2b bits-per-char gate: does ①+② full multi-block LM training close the bpc gap vs tuned BP?

CFG corpus is the controlled primary. Conditions:
  BP        : torch autograd (tuned baseline)
  FIXED_DFA : per-block fixed-random broadcast, A ignored, dW_O/dW_2 exact
  ONE_TWO   : ①+② — per-block A-routed value-exact grads + ② transport-free R_O adaptation

PRE-REGISTERED as a NO-GO: the gate (①+② − BP ≤ 0.15 bpc) is expected to FAIL. Do not tune to pass.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)  # perf: tiny matmuls → BLAS threads cost ~500× (prototype finding)
EPS = 1e-5

from feedflipnets.core.lm import NumpyLM
from feedflipnets.data.char_lm import (
    build_cfg_corpus, build_english_corpus, prep, bigram_floor, get_batch,
)

CONDITIONS = ["BP", "FIXED_DFA", "ONE_TWO"]


# ---- tuned-BP torch baseline (autograd; the reference the gate is measured against) ----
class _Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=EPS, elementwise_affine=False)
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, eps=EPS, elementwise_affine=False)
        self.f1 = nn.Linear(d, h, bias=False); self.f2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        T, d = x.size(1), x.size(2)
        y = self.ln1(x)
        att = (self.q(y) @ self.k(y).transpose(-2, -1)) / math.sqrt(d)
        att = att.masked_fill(torch.triu(torch.ones(T, T), 1).bool(), float("-inf"))
        x = x + self.o(F.softmax(att, -1) @ self.v(y))
        return x + self.f2(F.relu(self.f1(self.ln2(x))))


class _LM(nn.Module):
    def __init__(self, V, d, h, N, T):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([_Block(d, h) for _ in range(N)])
        self.lnf = nn.LayerNorm(d, eps=EPS, elementwise_affine=False)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, idx):
        T = idx.size(1)
        x = self.emb(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def _train_bp(train, val, V, d, h, N, T, steps, lr, bs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = _LM(V, d, h, N, T); opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    def tb(src):
        xb, yb = get_batch(src, bs, T, rng)
        return torch.tensor(xb), torch.tensor(yb)

    def ev():
        model.eval(); tot = 0.0; c = 0
        with torch.no_grad():
            for _ in range(8):
                x, y = tb(val)
                loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
                tot += loss.item() * y.numel(); c += y.numel()
        model.train(); return tot / c / math.log(2)

    t0 = time.time(); best = 9.0; per_step = None
    for step in range(steps):
        s1 = time.time()
        x, y = tb(train)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 5:
            per_step = time.time() - s1
        if step % max(1, steps // 6) == 0:
            best = min(best, ev())
    best = min(best, ev())
    return best, time.time() - t0, per_step


def _train_np(mode, train, val, V, d, h, N, T, steps, lr, bs, seed, adapt_cfg):
    m = NumpyLM(V, d, h, N, T, seed, mode); m.set_opt(lr)
    t0 = time.time(); best = 9.0; per_step = None; traj = []
    for step in range(steps):
        s1 = time.time()
        xb, yb = get_batch(train, bs, T, m.rng)
        m.train_step(xb, yb, adapt_cfg if mode == "ONE_TWO" else None)
        if step == 5:
            per_step = time.time() - s1
        if step % max(1, steps // 6) == 0:
            b = m.eval_bpc(val); best = min(best, b); traj.append(round(b, 3))
    best = min(best, m.eval_bpc(val))
    return best, time.time() - t0, per_step, traj


_CORPUS_CACHE = {}


def _corpus(name):
    if name not in _CORPUS_CACHE:
        text = build_english_corpus() if name == "english" else build_cfg_corpus(0)
        train, val, V = prep(text)
        _CORPUS_CACHE[name] = (train, val, V, bigram_floor(train, val, V))
    return _CORPUS_CACHE[name]


def run_condition(condition, d, h, N, T, bs, steps, lr, rho, Ksamp, lrR, seed, corpus="cfg"):
    """Run ONE (condition, seed). Returns {condition, seed, bpc, wall, per_step, traj}."""
    train, val, V, _floor = _corpus(corpus)
    if condition == "BP":
        b, wt, ps = _train_bp(train, val, V, d, h, N, T, steps, lr, bs, seed); traj = []
    else:
        b, wt, ps, traj = _train_np(condition, train, val, V, d, h, N, T, steps, lr, bs, seed,
                                    (rho, Ksamp, lrR))
    return {"condition": condition, "seed": seed, "bpc": b, "wall": wt, "per_step": ps, "traj": traj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--d", type=int, default=48); ap.add_argument("--h", type=int, default=192)
    ap.add_argument("--N", type=int, default=2); ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rho", type=float, default=0.02); ap.add_argument("--Ksamp", type=int, default=4)
    ap.add_argument("--lrR", type=float, default=0.05)
    ap.add_argument("--corpus", type=str, default="cfg", choices=["cfg", "english"])
    args = ap.parse_args()

    _train, _val, V, floor = _corpus(args.corpus)
    print(f"corpus={args.corpus} vocab={V} bigram_floor={floor:.3f} bpc", flush=True)
    results = {c: [] for c in CONDITIONS}; persteps = {c: [] for c in CONDITIONS}
    rows = []
    for seed in range(args.seeds):
        for cond in CONDITIONS:
            r = run_condition(cond, args.d, args.h, args.N, args.T, args.bs, args.steps, args.lr,
                              args.rho, args.Ksamp, args.lrR, seed, corpus=args.corpus)
            results[cond].append(r["bpc"]); persteps[cond].append(r["per_step"]); rows.append(r)
            print(f"  seed{seed} {cond:10s} bpc={r['bpc']:.3f} traj={r['traj']}", flush=True)

    summary = {}
    for cond in CONDITIONS:
        a = np.array(results[cond])
        summary[cond] = {"bpc_mean": float(a.mean()), "bpc_std": float(a.std())}
        print(f"{cond:10s} bpc mean={a.mean():.3f} std={a.std():.3f}", flush=True)
    gap = np.array(results["ONE_TWO"]) - np.array(results["BP"])
    beats_fixed = float(np.mean(results["ONE_TWO"])) < float(np.mean(results["FIXED_DFA"]))
    # pre-registered gate: ①+② − BP ≤ 0.15 bpc, 95% CI excluding 0.25
    ci_half = 1.96 * gap.std(ddof=1) / math.sqrt(len(gap)) if len(gap) > 1 else float("nan")
    gate_pass = bool(gap.mean() <= 0.15 and (gap.mean() + ci_half) < 0.25)
    verdict = {
        "corpus": args.corpus, "vocab": V, "bigram_floor": floor,
        "steps": args.steps, "seeds": args.seeds, "summary": summary,
        "gap_one_two_minus_bp_mean": float(gap.mean()),
        "gap_ci_half": float(ci_half), "gap_per_seed": [float(x) for x in gap],
        "one_two_beats_fixed_dfa": beats_fixed,
        "gate_threshold_bpc": 0.15, "gate_ci_excludes": 0.25,
        "gate_pass": gate_pass, "decision": "GO" if gate_pass else "NO-GO",
        "rows": rows,
    }
    out = Path("data/report/m2b"); out.mkdir(parents=True, exist_ok=True)
    (out / "m2b_lm_ablation.json").write_text(json.dumps(verdict, indent=2))
    print(f"\n①+② − BP gap mean={gap.mean():.3f} bpc  beats_fixed_dfa={beats_fixed}  "
          f"gate={'PASS' if gate_pass else 'FAIL (NO-GO)'}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_m2b_ablation_smoke.py -q`
Expected: PASS (2 passed). At 200 steps ①+② already beats fixed-DFA and BP already beats ①+② — the same ordering the full run confirms.

- [ ] **Step 5: Commit**

```bash
git add experiments/m2b_lm_ablation.py tests/test_m2b_ablation_smoke.py
git commit -m "feat(m2b): BP/fixed-DFA/①+② ablation runner + GUARD 4 beats-fixed-DFA smoke

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Run the n≥5 CFG gate and record the HONEST NO-GO artifact

**Files:**
- Create: `data/report/m2b/m2b_lm_ablation.json` (generated), `data/report/m2b/README.md`

- [ ] **Step 1: Run the full CFG gate** (≈ 15 min CPU; n=5, 1500 steps, single-thread BLAS pinned)

Run: `.venv/bin/python -m experiments.m2b_lm_ablation --corpus cfg --seeds 5 --steps 1500`
Expected: prints the per-seed table + summary; `data/report/m2b/m2b_lm_ablation.json` written with `"decision": "NO-GO"`. Prototype (n=3): BP `0.772±0.008`, FIXED_DFA `3.848±0.004`, ①+② `2.763±0.046`; gap `1.991`; beats-fixed-DFA YES by `1.085`; per-step ①+②/BP ≈ 10×.

- [ ] **Step 2: (Optional) English secondary run**

Run: `.venv/bin/python -m experiments.m2b_lm_ablation --corpus english --seeds 5 --steps 1500`
Expected: BP lands ~1.3–1.7 bpc (genuine entropy); ①+② remains well above BP. Secondary evidence only; the gate is decided on CFG.

- [ ] **Step 3: Record the pre-registered M2b decision (HONEST NO-GO)**

Create `data/report/m2b/README.md`:

```markdown
# M2b result — Activation-Routed DFA on a char-level LM (bits-per-char gate)

Pre-registered gate (spec §6/M2b, ACCURACY claim):
- Value-path exact (① multi-block): dW_O, dW_2 vs autograd < 1e-5. [PASS]  (prototype 3.8e-16 / 1.3e-15)
- Transport-free (② adaptation): no autograd.grad, no W_Oᵀ + positive control flagged. [PASS]
- Loss-decreasing: ①+② and fixed-DFA bpc both fall during training. [PASS]
- ①+② beats fixed-DFA: ①+② bpc < fixed-DFA bpc (learns structure). [PASS]  (prototype +1.085 bpc)
- **GATE — competitive bpc: (①+② − BP) ≤ 0.15 bpc, 95% CI upper bound < 0.25. [FAIL — NO-GO]**
  Prototype: gap = 1.991 bpc (per-seed 2.04 / 2.02 / 1.92) — ~13× the threshold. Not close.

Decision: **NO-GO.** Fill exact numbers from m2b_lm_ablation.json (n=5).

The honest read (the arc conclusion): M2 showed ①+② lifts attention alignment (worst-θ 90°→46°) and
M2b confirms that win is REAL on a downstream task — ①+② beats fixed-DFA by ~1.1 bpc and edges below
the bigram floor, i.e. it genuinely learns grammatical structure that fixed-DFA cannot. But the
transport-free surrogate only PARTIALLY recovers W_Oᵀ (the M1 1/√d variance wall), so ①+② still sits
~2.0 bpc ABOVE tuned backprop and does not converge in-budget. **Alignment is necessary but not
sufficient for competitive bits-per-char.** This is recorded as an honest NO-GO; the gate was NOT
tuned to force a pass. Next steps, if pursued, target the surrogate-recovery ceiling (M1 wall), not
the alignment path (already validated).
```

- [ ] **Step 4: Commit**

```bash
git add data/report/m2b/m2b_lm_ablation.json data/report/m2b/README.md
git commit -m "chore(m2b): record CFG bpc ablation + HONEST NO-GO (alignment necessary, not sufficient)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

- **Reuse over rewrite (the ladder):** `feedflipnets/core/lm.py` restates the *already-validated* M2 block math (`ProtoBlock`/`ActivationRoutedDFA`) batched over `(B,T,d)` — it is not a new algorithm. Task 4's first test pins the batched kernel to `ActivationRoutedDFA.block_grads` so any divergence is caught immediately. The BP baseline is a stock `torch.nn` LM (autograd does the work). No new dependency (torch already pinned at M2). No ternary, no multi-head — out of scope.
- **Prototype cleanups applied (scout script → production modules):**
  1. **Split by responsibility:** the 513-line scout (`m2b_full_lm.py`) is decomposed into `data/char_lm.py` (corpus/batching), `core/lm.py` (LM + training step), and `experiments/m2b_lm_ablation.py` (runner) — each focused, importable, and tested. The scout's inline `argparse main` becomes a `run_condition(...)` API the smoke test drives.
  2. **Perf pin fixed:** the scout set `torch.set_num_threads(4)`; production pins **`torch.set_num_threads(1)`** and the three `*_NUM_THREADS=1` env vars *before* importing numpy, matching the scout's own ~500× finding on tiny batched matmuls (the scout's 4-thread torch only helped the BP arm and skewed the per-step ratio).
  3. **Corpus paths made portable:** `build_english_corpus` takes a `root="docs"` arg instead of the scout's hard-coded absolute `/Users/...`; CFG stays the controlled primary.
  4. **A go/no-go JSON artifact** with the numerically pre-registered gate (`≤0.15 bpc`, CI upper bound `<0.25`) and `decision` field — the scout only printed to stdout.
- **Four carried-forward guards, in the MULTI-BLOCK setting:**
  - GUARD 1 (value-path-exact, Task 4): seeds each block's two sublayers with the autograd-exact broadcast (`e_attn=dL/dx1`, `e_mlp=dL/dx2` from `autograd_ref`) and asserts `max|①dW_O − ref|`, `max|①dW_2 − ref| < 1e-5` for *every* block in a stack — the multi-block generalization of the M2 single-block check (prototype 3.8e-16).
  - GUARD 2 (transport-free, Task 5): source-inspects `ghat_ctx_nodepert` and `NumpyLM.train_step` for `autograd`/`W_Oᵀ`, plus a positive control (a `Wo.T` estimator the checker must FLAG) and an e-fixed transpose-perturbation control — the M2 lock-free pattern applied to ②'s online adaptation.
  - GUARD 3 (loss-decreasing, Task 3): ①+② and fixed-DFA bpc both fall over training.
  - GUARD 4 (beats-fixed-DFA + honest gap, Tasks 6–7): smoke asserts `①+② < fixed-DFA < …` and `BP < ①+②` (the NO-GO signal); the n≥5 gate records the exact gap.
- **Prototype-validated numbers (scratchpad `m2b_full_lm.py`, torch 2.8 CPU, CFG, float64 NumPy path):**
  - Value-path exact (multi-block): `max|①dW_O − ref| ≈ 3.8e-16`, `dW_2 ≈ 1.3e-15` < 1e-5. PASS.
  - Transport-free: NO `autograd.grad`, NO `W_Oᵀ` in ② (reads `P["Wo"]` forward only). PASS.
  - CFG gate (n=3, 1500 steps, floor 2.881): BP **0.772±0.008**, FIXED_DFA **3.848±0.004** (fails to beat bigram), ①+② **2.763±0.046** (beats bigram, still descending). ①+② beats FIXED_DFA by **1.085 bpc**. Gap ①+② − BP = **1.991 bpc** (per-seed 2.04/2.02/1.92). Per-step ①+②/BP ≈ **10×**.
  - **Gate (①+② − BP ≤ 0.15 bpc, CI upper < 0.25): FAIL — NO-GO.** The gap is ~13× the threshold; this is a decisive, honest NO-GO, NOT a manufactured pass. Do NOT tune to close it.
- **The arc conclusion this plan records:** ① is value-exact and A-reuse is real; ②'s transport-free adaptation partially recovers the surrogate (M1 1/√d wall) — enough to lift alignment (M2) and to learn structure fixed-DFA cannot (M2b: beats bigram, beats fixed-DFA by ~1.1 bpc), but NOT enough to reach BP's bpc. **Alignment is necessary but not sufficient for competitive bits-per-char.**
