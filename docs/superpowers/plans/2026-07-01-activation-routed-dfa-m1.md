# Perturbation-Taught Feedback on a Deep MLP (GATE-0 + M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable alignment-measurement plumbing and test whether **② Perturbation-Taught Feedback** (transport-free node-perturbation adaptation of DFA's feedback matrices) beats fixed-random DFA on the `c/√L` depth-decay of gradient alignment — the cheapest falsification of the design's core bet.

**Architecture:** Pure NumPy (float64). A configurable-depth MLP produces the existing `ActivationState`, so the existing `Backprop` strategy is the **exact** alignment reference (no autograd/torch needed at M1 — that departure is deferred to the M2 transformer plan). Alignment is `cos(Δ_strategy, Δ_backprop)` per weight matrix. ② is a new `FeedbackStrategy` that adapts each feedback matrix `B_idx` toward node-perturbation gradient estimates `ĝ = (ΔL/ρ²)·ξ` using a `perturb_loss_fn` the runner injects via `StrategyState.metadata`, never reading a downstream weight.

**Tech Stack:** Python 3.8+, NumPy, pytest. No torch. Follows the existing `feedflipnets.core` `FeedbackStrategy` Protocol and `ActivationState`/`StrategyState` types.

**Scope note:** This is M1 only. M2 (Transformer block + ① Activation-Routed DFA, torch-CPU, lock-free taint tracer) is a separate plan gated on M1's outcome. The lock-free probe here implements the spec's **part (b)** e-fixed perturbation test; the torch **part (a)** taint tracer lands in M2.

**Spec:** `docs/superpowers/specs/2026-07-01-activation-routed-dfa-design.md`.

---

## File Structure

- `feedflipnets/core/deep_mlp.py` (new) — configurable-depth MLP: forward → `ActivationState`, `forward_from` for partial re-forward, softmax-CE per-sample loss + output error, and `make_perturb_loss_fn`. One responsibility: the M1 testbed model + its loss/perturbation surface.
- `feedflipnets/core/strategies.py` (modify) — add `PerturbationTaughtFeedback` (②) next to the existing `DFA`.
- `feedflipnets/eval/__init__.py` (new, empty package marker).
- `feedflipnets/eval/gradcheck.py` (new) — finite-difference gradient check (GATE-0).
- `feedflipnets/eval/alignment.py` (new) — per-matrix cosine, `θ=arccos`, min-over-layers, OLS depth-slope.
- `feedflipnets/eval/lockfree.py` (new) — the e-fixed perturbation lock-free probe (spec §5.4 part b).
- `experiments/m1_depth_sweep.py` (new) — depth sweep runner writing JSON/CSV to `data/report/m1/`.
- `tests/test_deep_mlp.py`, `tests/test_gradcheck.py`, `tests/test_alignment_probe.py`, `tests/test_lockfree_probe.py`, `tests/test_perturbation_feedback.py`, `tests/test_m1_smoke.py` (new).

---

## Task 1: Deep-MLP testbed

**Files:**
- Create: `feedflipnets/core/deep_mlp.py`
- Test: `tests/test_deep_mlp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deep_mlp.py
import numpy as np
from feedflipnets.core.deep_mlp import DeepMLP, output_error, softmax_ce_per_sample, make_perturb_loss_fn


def _data(n=16, d=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, c, size=n)
    return X, y


def test_forward_shapes_and_activationstate():
    model = DeepMLP(layer_dims=[8, 16, 16, 3], seed=1)
    X, y = _data()
    state, logits, acts = model.forward(X)
    assert logits.shape == (16, 3)
    # one weight per layer, one input per layer, one deriv per hidden layer
    assert len(state.weights) == 3
    assert len(state.layer_inputs) == 3
    assert len(state.layer_derivs) == 2
    assert acts[0].shape == (16, 8) and acts[-1].shape == (16, 16)


def test_perturb_loss_fn_zero_delta_matches_base():
    model = DeepMLP(layer_dims=[8, 16, 16, 3], seed=2)
    X, y = _data()
    fn, base, _state = make_perturb_loss_fn(model, X, y)
    hd = model.layer_dims[2]  # hidden activation index 2 has width 16
    zero = np.zeros((X.shape[0], model.layer_dims[2]))
    got = fn(2, zero)
    assert np.allclose(got, base, atol=1e-10)


def test_output_error_is_softmax_minus_onehot():
    model = DeepMLP(layer_dims=[8, 16, 3], seed=3)
    X, y = _data()
    _state, logits, _acts = model.forward(X)
    err = output_error(logits, y)
    assert err.shape == (16, 3)
    # rows sum to zero: (p - onehot) sums to (1 - 1) = 0
    assert np.allclose(err.sum(axis=1), 0.0, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_deep_mlp.py -q`
Expected: FAIL — `ModuleNotFoundError: feedflipnets.core.deep_mlp`

- [ ] **Step 3: Write minimal implementation**

```python
# feedflipnets/core/deep_mlp.py
"""Configurable-depth NumPy MLP testbed for M1 alignment experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from .types import ActivationState

Array = np.ndarray


def relu(x: Array) -> Array:
    return np.maximum(0.0, x)


def softmax(z: Array) -> Array:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y: Array, num_classes: int) -> Array:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def softmax_ce_per_sample(logits: Array, y: Array) -> Array:
    """Per-sample cross-entropy, shape (batch,)."""
    p = softmax(logits)
    yoh = one_hot(y, logits.shape[1])
    return -(yoh * np.log(p + 1e-12)).sum(axis=1)


def output_error(logits: Array, y: Array) -> Array:
    """dL/dz at the output = softmax - one_hot, shape (batch, C). NOT divided by batch."""
    return softmax(logits) - one_hot(y, logits.shape[1])


@dataclass
class DeepMLP:
    """Bias-free ReLU MLP. layer_dims = [d_in, h1, ..., h_{L-1}, C]."""

    layer_dims: List[int]
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights: List[Array] = [
            (rng.standard_normal((self.layer_dims[i], self.layer_dims[i + 1]))
             / np.sqrt(self.layer_dims[i])).astype(np.float64)
            for i in range(len(self.layer_dims) - 1)
        ]

    @property
    def num_layers(self) -> int:
        return len(self.weights)

    def forward(self, X: Array) -> Tuple[ActivationState, Array, List[Array]]:
        L = self.num_layers
        h = X.astype(np.float64)
        layer_inputs: List[Array] = []
        layer_derivs: List[Array] = []
        acts: List[Array] = [h]  # acts[j] = input activation feeding layer j (acts[0] = X)
        logits = h
        for idx in range(L):
            layer_inputs.append(h)
            z = h @ self.weights[idx]
            if idx < L - 1:
                layer_derivs.append((z > 0).astype(np.float64))
                h = relu(z)
                acts.append(h)
            else:
                logits = z
        state = ActivationState(
            layer_inputs=layer_inputs,
            layer_derivs=layer_derivs,
            weights=[w.copy() for w in self.weights],
        )
        return state, logits, acts

    def forward_from(self, start_idx: int, h_start: Array) -> Array:
        """Re-run layers start_idx..L-1 given the activation feeding layer start_idx."""
        L = self.num_layers
        h = h_start
        logits = h
        for idx in range(start_idx, L):
            z = h @ self.weights[idx]
            if idx < L - 1:
                h = relu(z)
            else:
                logits = z
        return logits


def make_perturb_loss_fn(
    model: DeepMLP, X: Array, y: Array
) -> Tuple[Callable[[int, Array], Array], Array, ActivationState]:
    """Return (fn, base_per_sample_loss, activation_state).

    fn(hidden_idx, delta_h) perturbs the post-activation acts[hidden_idx] by delta_h and
    returns the per-sample loss after re-running the rest of the network. Samples are
    independent in an MLP, so per-sample node perturbation is exact.
    """
    state, logits, acts = model.forward(X)
    base = softmax_ce_per_sample(logits, y)

    def fn(hidden_idx: int, delta_h: Array) -> Array:
        h_pert = acts[hidden_idx] + delta_h
        logits2 = model.forward_from(hidden_idx, h_pert)
        return softmax_ce_per_sample(logits2, y)

    return fn, base, state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_deep_mlp.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add feedflipnets/core/deep_mlp.py tests/test_deep_mlp.py
git commit -m "feat(m1): deep MLP testbed with per-sample loss and perturbation surface"
```

---

## Task 2: Finite-difference grad-check (GATE-0)

**Files:**
- Create: `feedflipnets/eval/__init__.py` (empty), `feedflipnets/eval/gradcheck.py`
- Test: `tests/test_gradcheck.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gradcheck.py
import numpy as np
from feedflipnets.core.deep_mlp import DeepMLP, output_error
from feedflipnets.core.strategies import Backprop
from feedflipnets.eval.gradcheck import max_rel_err_vs_finite_diff


def test_backprop_matches_finite_difference():
    model = DeepMLP(layer_dims=[6, 10, 4], seed=5)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 6))
    y = rng.integers(0, 4, size=12)
    err = max_rel_err_vs_finite_diff(Backprop(), model, X, y, eps=1e-6)
    assert err < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gradcheck.py -q`
Expected: FAIL — `ModuleNotFoundError: feedflipnets.eval.gradcheck`

- [ ] **Step 3: Write minimal implementation**

```python
# feedflipnets/eval/__init__.py
```

```python
# feedflipnets/eval/gradcheck.py
"""Finite-difference gradient check for feedback strategies (GATE-0)."""
from __future__ import annotations

import numpy as np

from ..core.deep_mlp import DeepMLP, output_error, softmax_ce_per_sample

Array = np.ndarray


def _strategy_grads(strategy, model: DeepMLP, X: Array, y: Array):
    state, logits, _acts = model.forward(X)
    err = output_error(logits, y)
    sstate = strategy.init(model.__dict__.get("layer_dims_desc") or _desc(model))
    grads, _ = strategy.backward(state, err, sstate)
    return grads


class _Desc:
    def __init__(self, dims):
        self.layer_dims = dims


def _desc(model: DeepMLP) -> _Desc:
    return _Desc(list(model.layer_dims))


def max_rel_err_vs_finite_diff(strategy, model: DeepMLP, X: Array, y: Array, eps: float = 1e-6) -> float:
    """Max relative error between a strategy's grads and central finite differences.

    The finite-difference reference is the TRUE loss gradient (exact BP). Use only with
    strategies expected to match BP (e.g. Backprop) as the GATE-0 correctness gate.
    """
    grads = _strategy_grads(strategy, model, X, y)
    n = X.shape[0]
    worst = 0.0
    for idx, W in enumerate(model.weights):
        key = f"W{idx}"
        analytic = grads[key]
        num = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])
        while not it.finished:
            i, j = it.multi_index
            orig = W[i, j]
            W[i, j] = orig + eps
            lp = softmax_ce_per_sample(model.forward(X)[1], y).sum() / n
            W[i, j] = orig - eps
            lm = softmax_ce_per_sample(model.forward(X)[1], y).sum() / n
            W[i, j] = orig
            num[i, j] = (lp - lm) / (2 * eps)
            it.iternext()
        denom = np.maximum(np.abs(analytic) + np.abs(num), 1e-8)
        worst = max(worst, float(np.max(np.abs(analytic - num) / denom)))
    return worst
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gradcheck.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add feedflipnets/eval/__init__.py feedflipnets/eval/gradcheck.py tests/test_gradcheck.py
git commit -m "feat(m1): finite-difference grad-check (GATE-0)"
```

---

## Task 3: Alignment probe

**Files:**
- Create: `feedflipnets/eval/alignment.py`
- Test: `tests/test_alignment_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_alignment_probe.py
import numpy as np
from feedflipnets.eval.alignment import per_matrix_cosine, theta_deg, min_theta_over_layers, depth_slope


def test_identical_grads_give_cosine_one():
    a = {"W0": np.ones((3, 4)), "W1": np.array([[1.0, -2.0]])}
    cos = per_matrix_cosine(a, a)
    assert np.allclose(list(cos.values()), 1.0, atol=1e-9)


def test_orthogonal_grads_give_cosine_zero():
    a = {"W0": np.array([[1.0, 0.0]])}
    b = {"W0": np.array([[0.0, 1.0]])}
    cos = per_matrix_cosine(a, b)
    assert abs(cos["W0"]) < 1e-9


def test_theta_and_slope():
    assert abs(theta_deg(1.0) - 0.0) < 1e-9
    assert abs(theta_deg(0.0) - 90.0) < 1e-9
    # theta increasing with depth -> positive slope of theta vs L
    slope, _ci = depth_slope([2, 4, 8], [10.0, 20.0, 40.0])
    assert slope > 0


def test_min_theta_over_layers_picks_worst():
    cos = {"W0": 0.9, "W1": 0.1}
    assert abs(min_theta_over_layers(cos) - theta_deg(0.1)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_alignment_probe.py -q`
Expected: FAIL — `ModuleNotFoundError: feedflipnets.eval.alignment`

- [ ] **Step 3: Write minimal implementation**

```python
# feedflipnets/eval/alignment.py
"""Alignment metrics: per-matrix cosine, theta, and depth-slope (spec §7)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray


def per_matrix_cosine(grads_a: Dict[str, Array], grads_b: Dict[str, Array]) -> Dict[str, float]:
    """Cosine between matching weight-matrix gradients, computed per matrix (never flattened together)."""
    out: Dict[str, float] = {}
    for key in grads_a:
        if key not in grads_b:
            continue
        a = grads_a[key].ravel()
        b = grads_b[key].ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        out[key] = float(np.dot(a, b) / denom)
    return out


def theta_deg(cosine: float) -> float:
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def min_theta_over_layers(cosines: Dict[str, float]) -> float:
    """Worst-aligned layer = largest angle."""
    return max(theta_deg(c) for c in cosines.values())


def depth_slope(depths: List[int], theta_mins: List[float]) -> Tuple[float, float]:
    """OLS slope of theta_min vs L (degrees per layer) with a 95% CI half-width."""
    x = np.asarray(depths, dtype=np.float64)
    yv = np.asarray(theta_mins, dtype=np.float64)
    n = len(x)
    xm, ym = x.mean(), yv.mean()
    sxx = float(((x - xm) ** 2).sum())
    slope = float(((x - xm) * (yv - ym)).sum() / (sxx + 1e-12))
    if n <= 2:
        return slope, float("inf")
    resid = yv - (ym + slope * (x - xm))
    s2 = float((resid ** 2).sum() / (n - 2))
    se = np.sqrt(s2 / (sxx + 1e-12))
    return slope, float(1.96 * se)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_alignment_probe.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add feedflipnets/eval/alignment.py tests/test_alignment_probe.py
git commit -m "feat(m1): alignment probe (per-matrix cosine, theta, depth-slope)"
```

---

## Task 4: Lock-free probe (spec §5.4 part b)

**Files:**
- Create: `feedflipnets/eval/lockfree.py`
- Test: `tests/test_lockfree_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lockfree_probe.py
import numpy as np
from feedflipnets.core.deep_mlp import DeepMLP, output_error
from feedflipnets.core.strategies import Backprop, DFA
from feedflipnets.eval.lockfree import e_fixed_perturbation_max_change


def _setup():
    model = DeepMLP(layer_dims=[6, 10, 10, 4], seed=7)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 6))
    y = rng.integers(0, 4, size=8)
    state, logits, _acts = model.forward(X)
    err = output_error(logits, y)
    return model, state, err


def test_dfa_is_lock_free():
    model, state, err = _setup()
    strat = DFA(rng=np.random.default_rng(0))
    change = e_fixed_perturbation_max_change(strat, model, state, err, downstream_idx=2, upstream_key="W0")
    assert change < 1e-12  # grad of W0 does not depend on downstream weight W2


def test_backprop_is_rejected():
    model, state, err = _setup()
    change = e_fixed_perturbation_max_change(Backprop(), model, state, err, downstream_idx=2, upstream_key="W0")
    assert change > 1e-6  # BP's W0 grad DOES depend on the downstream weight
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lockfree_probe.py -q`
Expected: FAIL — `ModuleNotFoundError: feedflipnets.eval.lockfree`

- [ ] **Step 3: Write minimal implementation**

```python
# feedflipnets/eval/lockfree.py
"""Lock-free invariant probe (spec §5.4 part b): e-fixed downstream-weight perturbation.

Holds the broadcast error e and layer-l caches FIXED, perturbs ONLY a downstream weight
as visible to backward, and asserts an upstream grad is unchanged. A lock-free strategy
never dereferences a downstream weight, so its upstream grad is invariant; backprop's is not.
The torch taint-tracer (part a) is added in the M2 plan.
"""
from __future__ import annotations

import copy

import numpy as np

from ..core.deep_mlp import DeepMLP

Array = np.ndarray


class _Desc:
    def __init__(self, dims):
        self.layer_dims = dims


def e_fixed_perturbation_max_change(
    strategy, model: DeepMLP, activations, error: Array, downstream_idx: int, upstream_key: str
) -> float:
    sstate = strategy.init(_Desc(list(model.layer_dims)))
    base_grads, _ = strategy.backward(activations, error, sstate)

    perturbed = copy.deepcopy(activations)
    rng = np.random.default_rng(123)
    perturbed.weights[downstream_idx] = perturbed.weights[downstream_idx] + 0.1 * rng.standard_normal(
        perturbed.weights[downstream_idx].shape
    )
    sstate2 = strategy.init(_Desc(list(model.layer_dims)))
    pert_grads, _ = strategy.backward(perturbed, error, sstate2)

    return float(np.max(np.abs(base_grads[upstream_key] - pert_grads[upstream_key])))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lockfree_probe.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add feedflipnets/eval/lockfree.py tests/test_lockfree_probe.py
git commit -m "feat(m1): e-fixed lock-free probe (rejects BP, passes DFA)"
```

---

## Task 5: PerturbationTaughtFeedback strategy (②)

**Files:**
- Modify: `feedflipnets/core/strategies.py` (add class + export)
- Test: `tests/test_perturbation_feedback.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_perturbation_feedback.py
import numpy as np
from feedflipnets.core.deep_mlp import DeepMLP, output_error, make_perturb_loss_fn
from feedflipnets.core.strategies import Backprop, DFA, PerturbationTaughtFeedback
from feedflipnets.eval.alignment import per_matrix_cosine
from feedflipnets.eval.lockfree import e_fixed_perturbation_max_change


class _Desc:
    def __init__(self, dims):
        self.layer_dims = dims


def _run(strategy, model, X, y, steps, lr=0.1):
    sstate = strategy.init(_Desc(list(model.layer_dims)))
    for _ in range(steps):
        state, logits, _acts = model.forward(X)
        err = output_error(logits, y)
        fn, _base, _st = make_perturb_loss_fn(model, X, y)
        sstate.metadata["perturb_loss_fn"] = fn
        grads, sstate = strategy.backward(state, err, sstate)
        for idx in range(model.num_layers):
            model.weights[idx] -= lr * grads[f"W{idx}"]
    return sstate


def test_perturbation_feedback_is_lock_free():
    model = DeepMLP(layer_dims=[6, 10, 10, 4], seed=2)
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((8, 6)), rng.integers(0, 4, size=8)
    state, logits, _ = model.forward(X)
    err = output_error(logits, y)
    strat = PerturbationTaughtFeedback(rng=np.random.default_rng(1))
    change = e_fixed_perturbation_max_change(strat, model, state, err, downstream_idx=2, upstream_key="W0")
    assert change < 1e-12


def test_feedback_alignment_improves_over_fixed_dfa():
    # After adaptation, B*e should align to the true dL/dh better than fixed DFA does,
    # so the ARGMAX weight-grad cosine to BP should exceed fixed DFA's on a deep-ish net.
    dims = [8, 24, 24, 24, 4]
    Xr = np.random.default_rng(9).standard_normal((32, 8))
    yr = np.random.default_rng(10).integers(0, 4, size=32)

    m_fixed = DeepMLP(layer_dims=dims, seed=3)
    _run(DFA(rng=np.random.default_rng(4)), m_fixed, Xr.copy(), yr, steps=0)
    m_adapt = DeepMLP(layer_dims=dims, seed=3)
    st = _run(PerturbationTaughtFeedback(rng=np.random.default_rng(4), samples_per_step=8, lr_B=0.2),
              m_adapt, m_adapt_X := Xr.copy(), yr, steps=200)

    # Measure worst-layer cosine to BP on a FRESH forward at the same weights for both feedbacks.
    def worst_cos(model, feedback_state):
        state, logits, _ = model.forward(Xr)
        err = output_error(logits, y=yr)
        bp, _ = Backprop().backward(state, err, Backprop().init(_Desc(dims)))
        strat = DFA(rng=np.random.default_rng(0))
        strat_state = strat.init(_Desc(dims))
        strat_state.feedback = feedback_state.feedback  # reuse adapted / fixed matrices
        dfa, _ = strat.backward(state, err, strat_state)
        return min(per_matrix_cosine(dfa, bp).values())

    fixed_state = DFA(rng=np.random.default_rng(4)).init(_Desc(dims))
    assert worst_cos(m_adapt, st) > worst_cos(m_adapt, fixed_state)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_perturbation_feedback.py -q`
Expected: FAIL — `ImportError: cannot import name 'PerturbationTaughtFeedback'`

- [ ] **Step 3: Write minimal implementation** (append to `feedflipnets/core/strategies.py`, before `__all__`, and add the name to `__all__`)

```python
@dataclass
class PerturbationTaughtFeedback:
    """② Transport-free feedback learning.

    Fast path is DFA. Each step, one feedback matrix (round-robin) is nudged so that
    ``error @ B_idx`` aligns with a node-perturbation estimate of dL/dh at the corresponding
    hidden activation. The estimate uses only the broadcast error and a scalar loss delta
    obtained from ``state.metadata['perturb_loss_fn']`` — it never reads a downstream weight.
    Update is direction-only (scale-invariant); antithetic ±xi cancels O(rho^2) curvature bias.
    """

    rng: np.random.Generator
    rho: float = 0.05
    lr_B: float = 0.1
    samples_per_step: int = 4

    def init(self, model: ModelDescription) -> StrategyState:
        dims = model.layer_dims
        out = dims[-1]
        mats: List[Array] = [
            (self.rng.standard_normal((out, hd)) / np.sqrt(out)).astype(np.float64)
            for hd in dims[1:-1]
        ]
        return StrategyState(feedback=mats, metadata={"step": 0})

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        weights = activations.weights
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs
        B = list(state.feedback)
        md = dict(state.metadata)

        grads: Gradients = {}
        batch = error.shape[0]
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ error / batch
        for idx in reversed(range(last_idx)):
            projected = error @ B[idx]
            delta_layer = projected * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta_layer / batch

        fn = md.get("perturb_loss_fn")
        if fn is not None and last_idx >= 1:
            step = int(md.get("step", 0))
            idx = step % last_idx            # feedback index 0..last_idx-1
            hidden_idx = idx + 1             # perturb post-activation acts[hidden_idx]
            width = B[idx].shape[1]
            ghat = np.zeros((batch, width), dtype=np.float64)
            for _ in range(self.samples_per_step):
                xi = self.rng.standard_normal((batch, width)) * self.rho
                loss_plus = fn(hidden_idx, xi)
                loss_minus = fn(hidden_idx, -xi)
                ghat += ((loss_plus - loss_minus) / (2.0 * self.rho ** 2))[:, None] * xi
            ghat /= self.samples_per_step

            projected = error @ B[idx]
            unit_g = ghat / (np.linalg.norm(ghat, axis=1, keepdims=True) + 1e-12)
            unit_p = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-12)
            dB = error.T @ (unit_g - unit_p) / batch
            B[idx] = B[idx] + self.lr_B * dB
            md["step"] = step + 1

        return grads, StrategyState(feedback=B, metadata=md)
```

Then update the export list:

```python
__all__ = [
    "FeedbackStrategy",
    "Backprop",
    "DFA",
    "TernaryDFA",
    "StructuredFeedback",
    "PerturbationTaughtFeedback",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_perturbation_feedback.py -q`
Expected: PASS (2 passed). If `test_feedback_alignment_improves_over_fixed_dfa` is flaky at the margin, raise `samples_per_step` to 16 and `steps` to 400 (variance ∝ width per spec §4) — do NOT loosen the assertion.

- [ ] **Step 5: Commit**

```bash
git add feedflipnets/core/strategies.py tests/test_perturbation_feedback.py
git commit -m "feat(m1): PerturbationTaughtFeedback strategy (transport-free feedback learning)"
```

---

## Task 6: M1 depth-sweep runner + negative control

**Files:**
- Create: `experiments/__init__.py` (empty package marker — repo convention; required for `python -m experiments.…` and the pytest import), `experiments/m1_depth_sweep.py`
- Test: `tests/test_m1_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m1_smoke.py
import numpy as np
from experiments.m1_depth_sweep import run_condition, fixed_dfa_slope_is_negative


def test_run_condition_deterministic():
    a = run_condition(strategy="dfa", depth=4, seed=0, steps=20, samples_per_step=2)
    b = run_condition(strategy="dfa", depth=4, seed=0, steps=20, samples_per_step=2)
    assert np.isclose(a["min_theta"], b["min_theta"])


def test_fixed_dfa_negative_control():
    # Fixed random DFA's worst-layer angle should grow with depth (theta slope > 0).
    assert fixed_dfa_slope_is_negative() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m1_smoke.py -q`
Expected: FAIL — `ModuleNotFoundError: experiments.m1_depth_sweep`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/m1_depth_sweep.py
"""M1: does Perturbation-Taught Feedback (②) bend the c/sqrt(L) alignment decay vs fixed DFA?"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from feedflipnets.core.deep_mlp import DeepMLP, output_error, make_perturb_loss_fn
from feedflipnets.core.strategies import Backprop, DFA, PerturbationTaughtFeedback
from feedflipnets.eval.alignment import per_matrix_cosine, min_theta_over_layers, depth_slope

DEPTHS = [2, 4, 8, 16]
SEEDS = [0, 1, 2, 3, 4]
WIDTH = 32
D_IN = 16
C = 4
N = 128


class _Desc:
    def __init__(self, dims):
        self.layer_dims = dims


def _dims(depth: int) -> List[int]:
    return [D_IN] + [WIDTH] * (depth - 1) + [C]


def _data(seed: int):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, D_IN)), rng.integers(0, C, size=N)


def run_condition(strategy: str, depth: int, seed: int, steps: int = 300,
                  samples_per_step: int = 8, lr: float = 0.1) -> Dict[str, float]:
    dims = _dims(depth)
    X, y = _data(seed)
    model = DeepMLP(layer_dims=dims, seed=seed + 100)
    if strategy == "dfa":
        strat = DFA(rng=np.random.default_rng(seed + 200))
    elif strategy == "perturb":
        strat = PerturbationTaughtFeedback(rng=np.random.default_rng(seed + 200),
                                           samples_per_step=samples_per_step, lr_B=0.2)
    else:
        raise ValueError(strategy)
    sstate = strat.init(_Desc(dims))
    for _ in range(steps):
        state, logits, _acts = model.forward(X)
        err = output_error(logits, y)
        fn, _base, _st = make_perturb_loss_fn(model, X, y)
        sstate.metadata["perturb_loss_fn"] = fn
        grads, sstate = strat.backward(state, err, sstate)
        for idx in range(model.num_layers):
            model.weights[idx] -= lr * grads[f"W{idx}"]

    state, logits, _acts = model.forward(X)
    err = output_error(logits, y)
    bp, _ = Backprop().backward(state, err, Backprop().init(_Desc(dims)))
    ref = DFA(rng=np.random.default_rng(0))
    ref_state = ref.init(_Desc(dims))
    ref_state.feedback = sstate.feedback
    strat_grads, _ = ref.backward(state, err, ref_state)
    cos = per_matrix_cosine(strat_grads, bp)
    return {"strategy": strategy, "depth": depth, "seed": seed,
            "min_theta": min_theta_over_layers(cos)}


def _mean_min_theta(strategy: str) -> List[float]:
    out = []
    for depth in DEPTHS:
        vals = [run_condition(strategy, depth, s)["min_theta"] for s in SEEDS]
        out.append(float(np.mean(vals)))
    return out


def fixed_dfa_slope_is_negative() -> bool:
    """Negative control uses a fast small config: theta slope vs depth should be POSITIVE
    (alignment DECAYS => angle grows). Returns True when the control reproduces."""
    thetas = []
    for depth in [2, 4, 8]:
        vals = [run_condition("dfa", depth, s, steps=0)["min_theta"] for s in SEEDS]
        thetas.append(float(np.mean(vals)))
    slope, _ = depth_slope([2, 4, 8], thetas)
    return slope > 0


def main() -> None:
    out_dir = Path("data/report/m1")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for strategy in ["dfa", "perturb"]:
        thetas = _mean_min_theta(strategy)
        slope, ci = depth_slope(DEPTHS, thetas)
        results[strategy] = {"depths": DEPTHS, "mean_min_theta": thetas,
                             "theta_slope_deg_per_layer": slope, "slope_ci95": ci}
    (out_dir / "m1_depth_sweep.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m1_smoke.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add experiments/__init__.py experiments/m1_depth_sweep.py tests/test_m1_smoke.py
git commit -m "feat(m1): depth-sweep runner with fixed-DFA negative control"
```

---

## Task 7: Run the M1 sweep and record the go/no-go artifact

**Files:**
- Create: `data/report/m1/m1_depth_sweep.json` (generated), `data/report/m1/README.md`

- [ ] **Step 1: Run the full sweep**

Run: `python -m experiments.m1_depth_sweep`
Expected: prints JSON with `dfa` and `perturb` blocks; `data/report/m1/m1_depth_sweep.json` written.

- [ ] **Step 2: Record the pre-registered M1 decision**

Create `data/report/m1/README.md` capturing the spec §6/M1 gate against the produced numbers:

```markdown
# M1 result — Perturbation-Taught Feedback vs fixed DFA

Pre-registered gate (spec §6/M1):
- Negative control: fixed-DFA `theta_slope_deg_per_layer` > 0 (alignment decays with depth). [PASS/FAIL]
- Success: `perturb` slope >= -2 deg/layer (flat, CI excludes the fixed-DFA decay band)
  AND at L=16 `perturb` beats `dfa` by >= 5 deg with non-overlapping 95% CIs. [PASS/FAIL]
- Budget precondition: samples_per_step x steps >= N(width). Record N and spend. [MET/UNMET]

Decision: [GO to M2 / revise estimator / drop ②]. Fill from m1_depth_sweep.json.
```

- [ ] **Step 3: Commit**

```bash
git add data/report/m1/m1_depth_sweep.json data/report/m1/README.md
git commit -m "chore(m1): record depth-sweep results and go/no-go decision"
```

---

## Self-Review

- **Spec coverage:** §5.1 `block_cache` intentionally deferred (M1 MLP needs no attention cache — noted in scope). §5.2 ② implemented (Task 5); ① deferred to M2. §5.3 torch deferred to M2 (M1 exact reference is hand-written float64 BP). §5.4 grad-check (Task 2), alignment probe (Task 3), lock-free part-b (Task 4); taint tracer deferred to M2. §6 GATE-0 (Tasks 2/4) + M1 negative control + gate (Tasks 6/7). §7 metrics (Task 3). §9 variance model reflected in Task 5 note + Task 7 budget precondition.
- **Placeholder scan:** none — every code step is complete and runnable.
- **Type consistency:** `_Desc(dims)` used everywhere a `ModelDescription`-like object is needed (has `.layer_dims`); grads keyed `W{idx}`; `StrategyState.metadata["perturb_loss_fn"]` is the single injection point; `make_perturb_loss_fn` returns `(fn, base, state)` consistently.
- **Known simplification (`ponytail`):** ② learns a fixed linear `e→g` map (spec §4 ceiling); this is sufficient for M1's depth-slope test, not for attention's moving target — that is ①'s job in M2.
