# FeedFlipNets — Repo Map

Entrypoints
- CLI: `cli/main.py` — argparse; runs a resolved preset or experiment config.
- Pipeline: `feedflipnets/training/pipelines.py` — builds dataset, model, trainer; writes artifacts.

Core Components
- `feedflipnets/training/trainer.py` — deterministic loops with strategies and ternary flipping.
- `feedflipnets/core/strategies.py` — `Backprop`, `DFA`, `TernaryDFA`, `StructuredFeedback`.
- `feedflipnets/core/quant.py` — deterministic/stochastic ternary quantisers.

Data
- `feedflipnets/data/*` — MNIST, UCR, 20 Newsgroups, California Housing with offline fixtures.
- Offline control via `FEEDFLIP_DATA_OFFLINE=1` and `offline: true` in presets.

Reporting
- Metrics sinks: `feedflipnets/reporting/metrics.py` (JSONL/CSV), `summary.py` (AUC + summary).
- Artifacts manifest: `feedflipnets/reporting/artifacts.py`.

Bench & Aggregation
- Runner: `scripts/run_benchmark.py` — variants × datasets × seeds across offline/real modes.
- Aggregator: `scripts/compile_benchmark_report.py` — `runs/bench/*` → `data/report/*`.
- Snapshot (presets): `scripts/aggregate_modality_results.py` → `data/report/*`.

Convenience (added)
- `exp/runner.py` — Typer CLI wrapping the bench + report.
- `exp/experiments.yaml` — subsets: MWR, ABLATE, SWEEP.
- `exp/Dockerfile` — minimal container for scripted runs.

