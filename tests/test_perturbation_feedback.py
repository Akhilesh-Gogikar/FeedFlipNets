import numpy as np

from feedflipnets.core.deep_mlp import DeepMLP, make_perturb_loss_fn, output_error
from feedflipnets.core.strategies import DFA, Backprop, PerturbationTaughtFeedback
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
    change = e_fixed_perturbation_max_change(
        strat, model, state, err, downstream_idx=2, upstream_key="W0"
    )
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
    st = _run(
        PerturbationTaughtFeedback(rng=np.random.default_rng(4), samples_per_step=8, lr_B=0.2),
        m_adapt,
        Xr.copy(),
        yr,
        steps=200,
    )

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
