#!/usr/bin/env bash
set -euo pipefail

# One-click, deterministic reproduction of core results (laptop-safe).
#
# Datasets: Fashion-MNIST (vision), AG News (text), California Housing (tabular)
# Seeds: 0..4 (95% CIs)
# Modes: offline + real
# Variants: a compact set from scripts/run_benchmark.py (includes BP/DFA baselines)

PY=${PY:-python}

echo "[repro] Running benchmarks for seeds 0..4"

$PY scripts/run_benchmark.py \
  --datasets fashion_mnist ag_news california_housing adult \
  --seeds 0 1 2 3 4

echo "[repro] Aggregating results into data/report"
$PY scripts/compile_benchmark_report.py

echo "[repro] Done. See data/report for tables and plots."
