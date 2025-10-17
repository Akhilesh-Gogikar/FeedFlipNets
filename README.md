FeedFlipNets
===========

Deterministic Direct Feedback Alignment (DFA) with ternary forward weights, implemented as a clear, reproducible baseline that runs on a laptop. The code accompanies the paper “FeedFlipNets: Deterministic DFA with Ternary Forwards—Stability and Convergence on Small, Standard Benchmarks” and reproduces its figures, tables, and results end‑to‑end.

Project status: complete — no further development planned.

- Paper PDF: `docs/paper/main.pdf`
- Citation: see `CITATION.cff`


What’s Inside (Paper‑Aligned)
-----------------------------

- Deterministic runs: fixed seeds, offline fixtures, CPU‑friendly NumPy.
- Method: ternary forward weights with float “shadow” parameters; forward uses `W = Q_τ(V)`, gradients update shadows `V` via DFA with fixed feedback matrices.
- Controls: flip schedule (`per_step`/`per_epoch`), threshold `τ`, deterministic vs. stochastic ternary.
- Structured feedback: orthogonal or Hadamard `B_l` options that stabilize deeper MLPs without sacrificing speed.
- Reproducible reporting: scripted sweeps yield JSON/CSV summaries and plots under `data/report/` and build the manuscript.


Results at a Glance
-------------------

- Vision (MNIST, real): test accuracy — BP 96.49% vs DFA (float) 95.14%.
- Text (AG News, real): test accuracy — BP 90.05% vs DFA (float) 89.98%.
- Text (20 Newsgroups, real): DFA (float) exceeds the matched BP baseline under this configuration, with the gap most visible at intermediate learning rates.
- Time‑series (UCR GunPoint): ternary and float DFA both reach 100%.
- Sparsity/throughput: ternary forwards yield typical zero ratios in the 0.5–0.75 range with competitive throughput, exposing accuracy–sparsity Pareto fronts and favorable memory footprints.

Theory & Guarantees
-------------------

- Alignment: we monitor a sign‑match statistic `p` that is typically > 1/2.
- Convergence: under ternary noise and `p > 1/2`, updates converge in finite time to a stationary neighborhood.
- Conditioning: orthogonal/Hadamard feedback raises the alignment floor and stabilizes deeper stacks.


Install
-------

```bash
git clone https://github.com/akigogikar/FeedFlipNets.git
cd FeedFlipNets
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

Requirements: Python 3.8+ (see `pyproject.toml` / `requirements-lock.txt`). Optional CNN baselines require `requirements-extras.txt`.


Quickstart
----------

- Smoke tests (fast, deterministic):

```bash
FEEDFLIP_DATA_OFFLINE=1 pytest -q tests/test_datasets_smoke.py tests/test_training_loops.py tests/integration/test_cli.py
```

- Train a preset from the CLI:

```bash
# Deterministic ternary flips
feedflip --preset mnist_mlp_dfa --feedback dfa --flip ternary --flip-schedule per_step --flip-threshold 0.05 --quant det

# Stochastic‑dithered ternary (seeded ⇒ reproducible)
feedflip --preset mnist_mlp_dfa --feedback dfa --flip ternary --flip-schedule per_step --flip-threshold 0.05 --quant stoch
```

Logs live in `runs/`. View TensorBoard with `tensorboard --logdir runs`.


Common Tasks
------------

```bash
# Reproduce curated benchmarks and compile the report
./reproduce_all.sh

# Make targets
make setup    # venv, deps, pre-commit
make test     # full test suite
make smoke    # quick deterministic sweep
make bench    # comprehensive benchmark + report
make report   # aggregate to data/report/
make paper    # build docs/paper/main.pdf
```


Selected Baselines
------------------

```bash
# MNIST MLP — BP + STE ternary (no DFA)
python -m cli.main --preset mnist_mlp_bp_ste

# MNIST MLP — DFA with normalized feedback (orthogonal)
python -m cli.main --preset mnist_mlp_dfa_orthogonal

# Fashion‑MNIST MLP — BP + STE ternary (no DFA)
python -m cli.main --preset fashion_mnist_mlp_bp_ste
```

CNN baselines (optional, PyTorch):

```bash
pip install -r requirements-extras.txt
python scripts/baselines/cnn_dfa_baselines.py --dataset fashion_mnist --method bp  --epochs 2 --batch-size 128 --run-dir runs/baselines/fmnist-bp
python scripts/baselines/cnn_dfa_baselines.py --dataset fashion_mnist --method dfa --epochs 2 --batch-size 128 --run-dir runs/baselines/fmnist-dfa
```


CLI and API
-----------

Presets live under `configs/presets/`. You can override any flag on the CLI or via config files.

```bash
# Short form using Make
make run PRESET=mnist_mlp_dfa EXTRA_ARGS='--feedback dfa --flip ternary --flip-schedule per_step'

# Or call the module directly
python -m cli.main --preset mnist_mlp_dfa --feedback dfa --flip ternary --flip-schedule per_step
```

Python API example:

```python
from pathlib import Path
import numpy as np
from feedflipnets.core.strategies import DFA
from feedflipnets.data.registry import get_dataset
from feedflipnets.training.trainer import FeedForwardModel, SGDOptimizer, Trainer

rng = np.random.default_rng(0)
spec = get_dataset("mnist", offline=True, cache_dir=Path(".cache"))
train_loader = spec.loader("train", batch_size=32)

model = FeedForwardModel([784, 256, 10], tau=0.05, quant="det", seed=0)
strategy = DFA(rng)
optimizer = SGDOptimizer(lr=0.05)

trainer = Trainer(model=model, strategy=strategy, optimizer=optimizer)
result = trainer.run(
    train_loader,
    epochs=5,
    seed=0,
    steps_per_epoch=spec.splits["train"] // 32,
    task_type="multiclass",
    num_classes=10,
    flip="ternary",
    flip_schedule="per_step",
    checkpoint_dir=Path("runs/example"),
)
print(result.metrics_path)
```


Reports & Reproducibility
-------------------------

- Offline by default with `FEEDFLIP_DATA_OFFLINE=1` and fixed seeds.
- Per‑run metrics live under `runs/`; aggregated summaries and plots go to `data/report/` (e.g., `benchmark_summary.{csv,md,json}`, `plots/`).
- Paper build: `scripts/build_pdf.sh` then open `docs/paper/main.pdf`.


Build the paper
---------------

```bash
scripts/build_pdf.sh           # latexmk/tectonic/pdflatex; auto‑generates Key Metrics from CSVs
open docs/paper/main.pdf       # on macOS
```

The Key Metrics table on the first page is built from `data/report/benchmark_summary.csv`. To tweak formatting, edit `scripts/generate_key_metrics.py`.


Configuration surface
---------------------

- `--feedback {backprop, dfa, ternary_dfa, structured}`
- `--flip {off, ternary}` and `--flip-schedule {per_step, per_epoch}`
- `--flip-threshold <float>` (τ)
- Optimizer: `--lr`, `--batch-size` (or via config)
- Seeds and dataset options; see presets in `configs/presets/`


Practical Tips
--------------

- Feedback structure: prefer orthogonal or Hadamard `B_l` for deeper stacks to preserve error magnitude and reduce tuning sensitivity.
- Flip schedule: start with `per_epoch` on text/tabular; tighten to `per_step` on vision/time‑series once training is stable.
- Thresholding: sweep `τ` to target zero ratios in the `0.5–0.7` range; push lower for accuracy, higher for compression.
- Optimization: consider gradient clipping around `1.0`; tune learning rates conservatively with ternary forwards to avoid limit cycles.
- Determinism: fix seeds for initialization and feedback to stabilize alignment onset and variance across runs.


Repository Layout
-----------------

- `cli/` — CLI entrypoint and commands
- `feedflipnets/` — core library (data, core modules, training)
- `configs/presets/` — runnable experiment presets
- `scripts/` — reporting/benchmark utilities
- `tests/` — unit, contract, integration tests
- `docs/paper/` — LaTeX manuscript and generated tables/figures
- `data/report/` — aggregated results and plots


Scope & Caveats
----------------

- Scope is intentionally limited to small MLPs on standard datasets; no SOTA claims on large‑scale CNNs/Transformers.
- Minimal CNN DFA baselines are included for reference and are expected to underperform BP without additional normalization/structure.


Troubleshooting
---------------

- Missing LaTeX? Install TeX Live/MacTeX or use the simpler `make paper` target.
- Slow downloads? Run with `FEEDFLIP_DATA_OFFLINE=1` to use fixtures.
- Fresh environment? Run `make setup` to install deps and pre‑commit hooks.


Citation & License
------------------

- Cite via `CITATION.cff` or the paper at `docs/paper/main.pdf`.
- License: see `LICENSE`.

BibTeX
```
@misc{gogikar_feedflipnets_2024,
  title   = {FeedFlipNets: Deterministic Offline Framework for Feedback Alignment},
  author  = {Gogikar, Aki},
  year    = {2024},
  version = {1.0.0-rc1},
  url     = {https://github.com/akigogikar/FeedFlipNets}
}
```

FAQ
---

- Do I need a GPU? No — CPU is fine for the supported experiments and smoke tests.


Keywords
--------

Deterministic Training, Direct Feedback Alignment, Ternary Networks, Quantized Training, Stability, Convergence, Reproducibility.
