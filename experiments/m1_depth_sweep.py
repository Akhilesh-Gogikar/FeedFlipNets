"""M1: does Perturbation-Taught Feedback (②) bend the c/sqrt(L) alignment decay vs fixed DFA?"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from feedflipnets.core.deep_mlp import DeepMLP, make_perturb_loss_fn, output_error
from feedflipnets.core.strategies import DFA, Backprop, PerturbationTaughtFeedback
from feedflipnets.eval.alignment import (
    depth_slope,
    min_theta_over_layers,
    per_matrix_cosine,
)

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


def run_condition(
    strategy: str,
    depth: int,
    seed: int,
    steps: int = 300,
    samples_per_step: int = 8,
    lr: float = 0.1,
) -> Dict[str, float]:
    dims = _dims(depth)
    X, y = _data(seed)
    model = DeepMLP(layer_dims=dims, seed=seed + 100)
    if strategy == "dfa":
        strat = DFA(rng=np.random.default_rng(seed + 200))
    elif strategy == "perturb":
        strat = PerturbationTaughtFeedback(
            rng=np.random.default_rng(seed + 200),
            samples_per_step=samples_per_step,
            lr_B=0.2,
        )
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
    return {
        "strategy": strategy,
        "depth": depth,
        "seed": seed,
        "min_theta": min_theta_over_layers(cos),
    }


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


def _finite_or_none(obj):
    # ponytail: non-finite thetas mean training diverged at depth; emit strict-JSON null
    # (bare NaN is invalid JSON and breaks downstream readers) rather than hiding the result.
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite_or_none(v) for v in obj]
    return obj


def main() -> None:
    out_dir = Path("data/report/m1")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for strategy in ["dfa", "perturb"]:
        thetas = _mean_min_theta(strategy)
        slope, ci = depth_slope(DEPTHS, thetas)
        results[strategy] = {
            "depths": DEPTHS,
            "mean_min_theta": thetas,
            "theta_slope_deg_per_layer": slope,
            "slope_ci95": ci,
        }
    payload = json.dumps(_finite_or_none(results), indent=2)
    (out_dir / "m1_depth_sweep.json").write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
