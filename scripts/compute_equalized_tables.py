#!/usr/bin/env python3
"""Build paper-ready tables with matched seeds and t-interval CIs.

This utility loads run-level metrics from ``data/report/benchmark_runs.json``
and writes LaTeX tables into ``docs/paper/_generated`` with seeds equalised
across compared variants. The goal is to guarantee reviewer-grade statistics
without launching new training runs.

Currently generates:

* ``real_results_table.tex`` – main Table 1 (real datasets).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_JSON = ROOT / "data/report/benchmark_runs.json"
OUT_DIR = ROOT / "docs/paper/_generated"

T_CRIT: Mapping[int, float] = {
    1: 0.0,
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def _ci95(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    std = float(np.std(values, ddof=1))
    if std <= 0.0:
        return 0.0
    t = T_CRIT.get(n, 1.96)
    return float(t * std / math.sqrt(n))


def _load_runs() -> List[Mapping[str, object]]:
    if not RUNS_JSON.exists():
        raise FileNotFoundError(f"Missing run-level metrics: {RUNS_JSON}")
    return json.loads(RUNS_JSON.read_text())


def _build_index(
    rows: Iterable[Mapping[str, object]]
) -> Mapping[Tuple[str, str, str], MutableMapping[int, Mapping[str, float]]]:
    index: Dict[Tuple[str, str, str], MutableMapping[int, Mapping[str, float]]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        mode = str(row["mode"])
        variant = str(row["strategy_variant"])
        seed = int(row["seed"])
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        key = (dataset, mode, variant)
        slot = index.setdefault(key, {})
        slot[seed] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    return index


@dataclass
class VariantConfig:
    name: str
    label: str
    is_baseline: bool = False


@dataclass
class DatasetConfig:
    dataset: str
    mode: str
    metric: str
    pretty_metric: str
    units: str
    scale: float
    variants: Sequence[VariantConfig]


DISPLAY_NAMES: Mapping[str, str] = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion MNIST",
    "ag_news": "AG News",
    "adult": "Adult",
    "california_housing": "California Housing",
    "20newsgroups": "20NG",
    "ucr": "UCR GunPoint",
}


REAL_CONFIG: Sequence[DatasetConfig] = (
    DatasetConfig(
        dataset="mnist",
        mode="real",
        metric="accuracy",
        pretty_metric="Accuracy",
        units="%",
        scale=100.0,
        variants=(
            VariantConfig("backprop_float_lr15", "BP (float)", is_baseline=True),
            VariantConfig("dfa_float_lr15", "DFA (float)"),
            VariantConfig("dfa_ternary_epoch_tau005", "DFA ternary (pe)"),
        ),
    ),
    DatasetConfig(
        dataset="fashion_mnist",
        mode="real",
        metric="accuracy",
        pretty_metric="Accuracy",
        units="%",
        scale=100.0,
        variants=(
            VariantConfig("backprop_float_lr15", "BP (float)", is_baseline=True),
            VariantConfig("dfa_float_lr15", "DFA (float)"),
            VariantConfig("dfa_ternary_epoch_tau005", "DFA ternary (pe)"),
        ),
    ),
    DatasetConfig(
        dataset="ag_news",
        mode="real",
        metric="accuracy",
        pretty_metric="Accuracy",
        units="%",
        scale=100.0,
        variants=(
            VariantConfig("backprop_float", "BP (float)", is_baseline=True),
            VariantConfig("dfa_float", "DFA (float)"),
            VariantConfig("dfa_ternary_epoch_tau005", "DFA ternary (pe)"),
        ),
    ),
    DatasetConfig(
        dataset="adult",
        mode="real",
        metric="accuracy",
        pretty_metric="Accuracy",
        units="%",
        scale=100.0,
        variants=(
            VariantConfig("backprop_float", "BP (float)", is_baseline=True),
            VariantConfig("dfa_float", "DFA (float)"),
            VariantConfig("dfa_ternary_epoch_tau005", "DFA ternary (pe)"),
        ),
    ),
    DatasetConfig(
        dataset="california_housing",
        mode="real",
        metric="r2",
        pretty_metric="$R^2$",
        units="--",
        scale=1.0,
        variants=(
            VariantConfig("backprop_float", "BP (float)", is_baseline=True),
            VariantConfig("dfa_float", "DFA (float)"),
            VariantConfig("structured_hadamard_float", "DFA + Hadamard"),
        ),
    ),
    DatasetConfig(
        dataset="20newsgroups",
        mode="real",
        metric="accuracy",
        pretty_metric="Accuracy",
        units="%",
        scale=100.0,
        variants=(
            VariantConfig("backprop_float_lr15", "BP (float)", is_baseline=True),
            VariantConfig("dfa_float_lr15", "DFA (float)"),
            VariantConfig("dfa_ternary_epoch_tau005", "DFA ternary (pe)"),
        ),
    ),
    DatasetConfig(
        dataset="ucr",
        mode="real",
        metric="accuracy",
        pretty_metric="Accuracy",
        units="%",
        scale=100.0,
        variants=(
            VariantConfig("backprop_float", "BP (float)", is_baseline=True),
            VariantConfig("dfa_float", "DFA (float)"),
            VariantConfig("ternary_dfa_step", "DFA ternary (ps)"),
        ),
    ),
)


def _format_pm(mean: float, ci: float, precision: int = 2) -> str:
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(mean)} $\\pm$ {fmt.format(ci)}"


def _render_real_table(
    index: Mapping[Tuple[str, str, str], Mapping[int, Mapping[str, float]]]
) -> str:
    lines: List[str] = []
    lines.append("\\begin{tabularx}{\\linewidth}{l l l l l r}")
    lines.append("\\toprule")
    lines.append("Dataset & Metric & Variant & Mean $\\pm$ CI & $n$ & $\\Delta$ vs baseline \\\\")
    lines.append("\\midrule")

    for cfg in REAL_CONFIG:
        group_key = [(cfg.dataset, cfg.mode, v.name) for v in cfg.variants]
        seed_sets = []
        for key in group_key:
            seeds = set(index.get(key, {}).keys())
            if not seeds:
                raise ValueError(f"No seeds found for {key}")
            seed_sets.append(seeds)
        common = set.intersection(*seed_sets)
        if not common:
            raise ValueError(
                "No overlapping seeds across variants for "
                f"{cfg.dataset} ({cfg.mode}). Please adjust configuration or runs."
            )
        selected_seeds = sorted(common)

        # Baseline stats for deltas
        baseline_cfg = next(v for v in cfg.variants if v.is_baseline)
        baseline_key = (cfg.dataset, cfg.mode, baseline_cfg.name)
        baseline_metrics = index[baseline_key]
        baseline_values = [
            baseline_metrics[s][cfg.metric] for s in selected_seeds if s in baseline_metrics
        ]
        if len(baseline_values) < len(selected_seeds):
            missing = set(selected_seeds) - set(baseline_metrics.keys())
            raise ValueError(f"Baseline {baseline_key} missing seeds {missing}")
        baseline_scaled = [cfg.scale * v for v in baseline_values]
        baseline_mean = float(np.mean(baseline_scaled))

        first_row = True
        for variant in cfg.variants:
            key = (cfg.dataset, cfg.mode, variant.name)
            metrics = index[key]
            values = [metrics[s][cfg.metric] for s in selected_seeds if s in metrics]
            if len(values) != len(selected_seeds):
                missing = set(selected_seeds) - set(metrics.keys())
                raise ValueError(f"Variant {key} missing seeds {missing}")
            scaled = [cfg.scale * v for v in values]
            mean = float(np.mean(scaled))
            ci = _ci95(scaled)
            delta = mean - baseline_mean
            precision = 2 if cfg.scale == 1.0 else 2
            pm = _format_pm(mean, ci, precision=precision)
            delta_fmt = f"{delta:+.2f}"
            dataset_label = DISPLAY_NAMES.get(cfg.dataset, cfg.dataset.replace("_", " ").title())
            metric_label = (
                f"{cfg.pretty_metric} ({cfg.units})" if variant is cfg.variants[0] else ""
            )
            if first_row:
                row = (
                    f"{dataset_label} & {metric_label} & {variant.label} & {pm} "
                    f"& {len(selected_seeds)} & {delta_fmt} \\\\"
                )
                lines.append(row)
                first_row = False
            else:
                row = f" &  & {variant.label} & {pm} " f"& {len(selected_seeds)} & {delta_fmt} \\\\"
                lines.append(row)
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    return "\n".join(lines)


def main() -> None:
    rows = _load_runs()
    index = _build_index(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real_table = _render_real_table(index)
    (OUT_DIR / "real_results_table.tex").write_text(real_table)
    print(f"Wrote {OUT_DIR / 'real_results_table.tex'}")


if __name__ == "__main__":
    main()
