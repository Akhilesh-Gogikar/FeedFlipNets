FeedFlipNets
===========

Deterministic Direct Feedback Alignment (DFA) with ternary forward weights on small, standard benchmarks. Reproducible on a laptop. Baseline‑oriented (not SOTA). Stochastic‑dithered ternary is provided with fixed seeds for unbiased, reproducible runs.

Project status: complete — no further development planned.


Why This Repo
-------------

- Deterministic: fixed seeds, offline fixtures, CPU‑friendly NumPy.
- Simple, strong baselines: backprop, DFA (float), ternary‑DFA, structured feedback (orthogonal/Hadamard).
- Clear controls: flip schedule (per_step/per_epoch), τ threshold, deterministic vs. stochastic ternary.
- Reproducible reports: JSON/CSV summaries and plots under `data/report/`.

Conceptually: we update float “shadow” weights `V` and expose ternary weights `W = Q_τ(V)` in the forward pass on a configurable “flip” schedule.


Install
-------

```bash
git clone https://github.com/akigogikar/FeedFlipNets.git
cd FeedFlipNets
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

Requirements: Python 3.8+ (see `pyproject.toml`). Optional CNN baselines require `requirements-extras.txt`.


Quickstart
----------

- Smoke tests (fast, deterministic):

```bash
FEEDFLIP_DATA_OFFLINE=1 pytest -q tests/test_datasets_smoke.py tests/test_training_loops.py tests/integration/test_cli.py
```

- Train a preset from the CLI:

```bash
# Deterministic ternary flips
python -m cli.main --preset mnist_mlp_dfa --feedback dfa --flip ternary --flip-schedule per_step --flip-threshold 0.05 --quant det

# Stochastic‑dithered ternary (seeded ⇒ reproducible)
python -m cli.main --preset mnist_mlp_dfa --feedback dfa --flip ternary --flip-schedule per_step --flip-threshold 0.05 --quant stoch
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


Export Trained Models
---------------------

Export MCU‑friendly formats from a checkpoint:

```bash
python scripts/export_ternary.py --ckpt runs/<run>/best.ckpt --out artifacts/export --name my_model --format all
# Produces: my_model_ternary_int8.npz, my_model_ternary_packed.bin + manifest.json, my_model_ternary.h
```

See `templates/mcu/README.md` for integration notes.


Reports & Reproducibility
-------------------------

- Offline by default with `FEEDFLIP_DATA_OFFLINE=1` and fixed seeds.
- Per‑run metrics live under `runs/`; aggregated summaries and plots go to `data/report/` (e.g., `benchmark_summary.{csv,md,json}`, `plots/`).
- Paper build: `scripts/build_pdf.sh` then open `docs/paper/main.pdf`.


Repository Layout
-----------------

- `cli/` — CLI entrypoint and commands
- `feedflipnets/` — core library (data, core modules, training)
- `configs/presets/` — runnable experiment presets
- `scripts/` — reporting/benchmark utilities
- `tests/` — unit, contract, integration tests
- `docs/paper/` — LaTeX manuscript and generated tables/figures
- `data/report/` — aggregated results and plots


Notes & Scope
-------------

- Baselines target small MLPs on standard datasets; we do not claim SOTA or large‑scale CNN/Transformer performance.
- DFA on CNNs is sensitive without backward‑path normalization/structure and is expected to underperform BP in our minimal baseline.


Citation & License
------------------

- Cite via `CITATION.cff` or the paper in `docs/paper/main.pdf`.
- License: see `LICENSE`.

Do I need a GPU? No — CPU is fine for the supported experiments and smoke tests.
