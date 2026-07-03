"""Accuracy-vs-transport frontier: how much must the update look like backprop to recover
backprop's accuracy? Sweep lambda: grad = (1-l)*DFA + l*BP. l=0 pure DFA, l=1 pure BP.
Convex (rises fast, plateaus near BP at small l) => gap cheaply closeable; linear/concave
(needs l->1) => the accuracy gap is fundamental. Deep MLP, hard teacher-labeled task.
Reuses M1's validated Backprop/DFA kernels. See data/report/speed/README.md.
"""
import numpy as np

from feedflipnets.core.deep_mlp import DeepMLP, output_error
from feedflipnets.core.strategies import DFA, Backprop

DIN, C, L, WIDTH = 32, 8, 6, 96
DIMS = [DIN] + [WIDTH] * (L - 1) + [C]
NTR, NTE = 2000, 1000


class _Desc:
    def __init__(self, dims):
        self.layer_dims = dims


def teacher_data(seed):
    rng = np.random.default_rng(seed)
    teacher = DeepMLP(layer_dims=[DIN, 128, 128, C], seed=999)
    X = rng.standard_normal((NTR + NTE, DIN))
    _, logits, _ = teacher.forward(X)
    y = logits.argmax(axis=1)
    return X[:NTR], y[:NTR], X[NTR:], y[NTR:]


def clip(grads, maxnorm=1.0):
    tot = np.sqrt(sum(float((g**2).sum()) for g in grads.values()))
    if tot <= maxnorm:
        return grads
    s = maxnorm / (tot + 1e-12)
    return {k: v * s for k, v in grads.items()}


def accuracy(model, X, y):
    _, logits, _ = model.forward(X)
    return float((logits.argmax(1) == y).mean())


def run(lam, seed, steps=1500, lr=0.2, bs=128):
    Xtr, ytr, Xte, yte = teacher_data(seed)
    model = DeepMLP(layer_dims=DIMS, seed=seed + 100)
    bp = Backprop()
    dfa = DFA(rng=np.random.default_rng(seed + 200))
    bp_s = bp.init(_Desc(DIMS))
    dfa_s = dfa.init(_Desc(DIMS))
    rng = np.random.default_rng(seed + 300)
    for _ in range(steps):
        idx = rng.choice(NTR, size=bs, replace=False)
        Xb, yb = Xtr[idx], ytr[idx]
        state, logits, _ = model.forward(Xb)
        err = output_error(logits, yb)
        g_bp, bp_s = bp.backward(state, err, bp_s)
        if lam >= 1.0:
            g = g_bp
        elif lam <= 0.0:
            g, dfa_s = dfa.backward(state, err, dfa_s)
        else:
            g_dfa, dfa_s = dfa.backward(state, err, dfa_s)
            g = {k: (1 - lam) * g_dfa[k] + lam * g_bp[k] for k in g_bp}
        g = clip(g)
        for i in range(model.num_layers):
            model.weights[i] -= lr * g[f"W{i}"]
    return accuracy(model, Xte, yte)


def main():
    lams = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    seeds = [0, 1, 2]
    print(f"task: teacher-labeled deep MLP, dims={DIMS}, {C} classes, {NTR}/{NTE} train/test")
    print("lambda  test_acc(mean over 3 seeds)   [0=pure DFA, 1=pure BP]")
    rows = {}
    for lam in lams:
        accs = [run(lam, s) for s in seeds]
        rows[lam] = float(np.mean(accs))
        print(f"  {lam:.2f}   {np.mean(accs):.3f}  (per-seed {[round(a, 3) for a in accs]})")
    bp_acc, dfa_acc = rows[1.0], rows[0.0]
    gap = bp_acc - dfa_acc
    print(f"\nBP acc={bp_acc:.3f}  DFA acc={dfa_acc:.3f}  gap={gap:.3f}")
    if gap > 1e-6:
        print(
            f"gap recovered at lambda=0.10: {(rows[0.1] - dfa_acc) / gap:.0%}   "
            f"at lambda=0.25: {(rows[0.25] - dfa_acc) / gap:.0%}"
        )


if __name__ == "__main__":
    main()
