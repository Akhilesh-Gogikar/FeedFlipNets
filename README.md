FeedFlipNets
===========

Deterministic training with Direct Feedback Alignment (DFA) and ternary forward weights. Small, standard benchmarks. Reproducible on a laptop.


Why FeedFlipNets
----------------

- Determinism: fixed seeds, offline fixtures, CPU‑friendly NumPy loops.
- Stability: clear flip schedules (per_step, per_epoch) and structured feedback options.
- Convergence: a compact theoretical treatment and alignment curves in the paper.

Training updates float “shadow” weights V; the forward path exposes ternary weights W = Qτ(V) on a configurable schedule (“flip”).


Highlights
----------

- Strategies: backprop, DFA (float), ternary‑DFA, structured feedback (orthogonal/Hadamard).
- Flip controls: τ threshold, deterministic/stochastic ternarization, per_step/per_epoch schedules.
- Artifacts: JSON/CSV logs, plots, and compiled tables under `data/report/`.
- Presets: laptop‑safe MLPs for vision, text, and tabular datasets.
- Paper: LaTeX manuscript with auto‑generated Key Metrics table.


Install
-------

```bash
git clone https://github.com/akigogikar/FeedFlipNets.git
cd FeedFlipNets
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

Requirements: Python 3.8+ (see `pyproject.toml` / `requirements-lock.txt`).


Quickstart
----------

- Smoke tests (fast, deterministic):

```bash
FEEDFLIP_DATA_OFFLINE=1 pytest -q tests/test_datasets_smoke.py tests/test_training_loops.py tests/integration/test_cli.py
```

- Run a preset via CLI:

```bash
python -m cli.main --preset mnist_mlp_dfa --feedback dfa --flip ternary --flip-schedule per_step --flip-threshold 0.05
```

- One‑click reproduction of curated benchmarks and report:

```bash
./reproduce_all.sh
# Aggregations and plots: data/report/
```

Make targets
------------

```bash
make setup        # create venv, install deps, pre-commit hooks
make test         # run full test suite
make smoke        # run deterministic preset sweep
make bench ARGS=  # comprehensive benchmark + report
make report       # aggregate metrics to data/report/
make paper        # (simple) build of docs/paper/main.pdf
```


CLI and API
-----------

Presets live under `configs/presets/`. You can override any flag on the CLI or via config files.

```bash
# Short form using Make
make run PRESET=mnist_mlp_dfa EXTRA_ARGS='--feedback dfa --flip ternary --flip-schedule per_step'
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


Reproducibility & artifacts
---------------------------

- Offline by default with `FEEDFLIP_DATA_OFFLINE=1` and fixed seeds.
- Metrics and manifests are written under `runs/` per run.
- Aggregated summaries, tables, and plots land in `data/report/`:
  - `data/report/benchmark_summary.md` / `.csv`
  - `data/report/best_configs.*`
  - `data/report/plots/`


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


Repository layout
-----------------

- `cli/` — CLI entrypoint and commands
- `feedflipnets/` — core library (data, core modules, training)
- `configs/presets/` — runnable experiment presets
- `scripts/` — reporting and benchmark utilities
- `tests/` — unit, contract, integration, and performance tests
- `docs/paper/` — LaTeX manuscript and references
- `data/report/` — aggregated results and plots


Results snapshot
----------------

Exact numbers are maintained by the reporting pipeline. See:

- `data/report/benchmark_summary.md` for consolidated metrics
- `data/report/best_configs_table.*` for best settings by dataset
- `data/report/plots/` for LR/τ sweeps, throughput, and alignment curves


Troubleshooting
---------------

- Missing LaTeX? Install TeX Live/MacTeX or use the simpler `make paper` target.
- Slow downloads? Run with `FEEDFLIP_DATA_OFFLINE=1` to use fixtures.
- Fresh environment? Run `make setup` to install deps and pre‑commit hooks.


Citation & license
------------------

- Cite via `CITATION.cff` or the paper in `docs/paper/main.pdf`.
- License: see `LICENSE`.

**Do I need a GPU?**

No. CPU is fine for the supported experiments and smoke tests.


⸻

Roadmap
-------

* Optional binary feedback matrices (±1) with variance scaling.
* Structured feedback (e.g., orthogonal/Hadamard) for stability.
* Plug-in quantizers (LSQ, DoReFa) and per-layer thresholds.
* Exporters for on-device inference formats.

Update these bullets as plans evolve.


⸻

Keywords
--------

Deterministic Training, Direct Feedback Alignment, Ternary Networks, Quantized Training, Stability, Convergence, Reproducibility.


⸻

Contributing
------------

PRs welcome. Please run the smoke suite locally before submitting.

```bash
# Lint + tests
make format lint test smoke
```

Add `ruff`, `black`, `mypy`, or pre-commit hooks here if your workflow uses them.


⸻

Citation
--------

If this work helps your research or product, please cite:

```bibtex
@software{FeedFlipNets,
  author = {Gogikar, A.},
  title  = {FeedFlipNets: Deterministic DFA with Ternary Forwards—Stability and Convergence},
  year   = {2025},
  url    = {https://github.com/akigogikar/FeedFlipNets}
}
```

Swap in a paper or arXiv entry once available.


⸻

License
-------

Apache-2.0 © Aki Gogikar


⸻

Drop-in checklist for maintainers
---------------------------------

* Replace placeholders with fresh metrics or CLI shortcuts as the project matures.
* Confirm Python version and dependency pins.
* Add one tiny metrics table from a smoke run (before/after flip) to make the README “pop.”
* If you have CI, wire the badge to your workflow URL.
