#!/usr/bin/env python3
"""Compile benchmark runs into aggregated artefacts for data/report.

This script scans ``runs/bench`` for manifests/metrics, computes
per-group statistics (mean/std over seeds), and writes a suite of files:

* ``data/report/benchmark_runs.json`` – raw run-level details.
* ``data/report/benchmark_summary.json`` – aggregated statistics.
* ``data/report/benchmark_summary.csv`` – tabular summary.
* ``data/report/benchmark_summary.md`` – human-readable report.

Usage:

    python scripts/compile_benchmark_report.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


ROOT = Path("runs/bench")
REPORT_DIR = Path("data/report")


def _read_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@dataclass
class RunRecord:
    dataset: str
    mode: str
    strategy: str
    strategy_variant: str
    flip: str
    flip_schedule: str
    seed: int
    run_dir: str
    metrics: Mapping[str, float]


def _iter_runs(root: Path) -> Iterable[RunRecord]:
    if not root.exists():
        return []
    for manifest_path in root.rglob("manifest.json"):
        run_dir = manifest_path.parent
        metrics_path = run_dir / "metrics_test.json"
        manifest = _read_json(manifest_path)
        metrics = _read_json(metrics_path)
        timing = _read_json(run_dir / "timing.json")
        config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
        train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
        model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
        data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
        dataset = str(data_cfg.get("name", "unknown")) if isinstance(data_cfg, dict) else "unknown"
        strategy = str(model_cfg.get("strategy", "unknown"))
        flip = str(train_cfg.get("flip", "unknown"))
        flip_schedule = str(train_cfg.get("flip_schedule", "off"))
        mode = "offline"
        env = manifest.get("environment", {}) if isinstance(manifest, dict) else {}
        if isinstance(env, dict) and env.get("offline") in {"0", 0, False}:
            mode = "real"
        parts = run_dir.relative_to(root).parts
        variant = parts[2] if len(parts) >= 3 else strategy
        seed_str = run_dir.name.replace("seed", "")
        try:
            seed = int(seed_str)
        except ValueError:
            seed = int(train_cfg.get("seed", 0)) if isinstance(train_cfg, dict) else 0
        numeric_metrics = {
            key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
        }
        if isinstance(timing, dict):
            test_timing = timing.get("test")
            if isinstance(test_timing, dict):
                total_sec = test_timing.get("total_sec")
                sample_count = numeric_metrics.get("sample_count")
                if (
                    isinstance(total_sec, (int, float))
                    and total_sec > 0
                    and isinstance(sample_count, (int, float))
                ):
                    numeric_metrics["test_throughput_samples_sec"] = float(sample_count) / float(
                        total_sec
                    )
        yield RunRecord(
            dataset=dataset,
            mode=mode,
            strategy=strategy,
            strategy_variant=variant,
            flip=flip,
            flip_schedule=flip_schedule,
            seed=seed,
            run_dir=str(run_dir),
            metrics=numeric_metrics,
        )


def _group_key(run: RunRecord) -> Tuple[str, str, str, str, str, str]:
    return (
        run.dataset,
        run.mode,
        run.strategy,
        run.strategy_variant,
        run.flip,
        run.flip_schedule,
    )


@dataclass
class SummaryRecord:
    dataset: str
    mode: str
    strategy: str
    strategy_variant: str
    flip: str
    flip_schedule: str
    metric: str
    mean: float
    std: float
    count: int


GroupKey = Tuple[str, str, str, str, str, str]


def _aggregate(runs: Sequence[RunRecord]) -> List[SummaryRecord]:
    grouped: MutableMapping[GroupKey, List[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[_group_key(run)].append(run)

    summary: List[SummaryRecord] = []
    for key, members in grouped.items():
        metrics_keys = set().union(*(run.metrics.keys() for run in members))
        for metric in sorted(metrics_keys):
            values = [run.metrics[metric] for run in members if metric in run.metrics]
            if not values:
                continue
            mu = mean(values)
            sigma = stdev(values) if len(values) > 1 else 0.0
            summary.append(
                SummaryRecord(
                    dataset=key[0],
                    mode=key[1],
                    strategy=key[2],
                    strategy_variant=key[3],
                    flip=key[4],
                    flip_schedule=key[5],
                    metric=metric,
                    mean=mu,
                    std=sigma,
                    count=len(values),
                )
            )
    return summary


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_csv(path: Path, rows: Sequence[SummaryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _format_pm(mu: float, sigma: float) -> str:
    if sigma == 0.0:
        return f"{mu:.4f}"
    return f"{mu:.4f} ± {sigma:.4f}"


def _best_by_dataset(summary: Sequence[SummaryRecord]) -> Dict[str, SummaryRecord]:
    best: Dict[str, SummaryRecord] = {}
    for record in summary:
        if record.metric not in {"accuracy", "macro_f1", "r2"}:
            continue
        key = (record.dataset, record.mode)
        current = best.get(key)
        if current is None:
            best[key] = record
            continue
        if record.metric == "r2":
            better = record.mean > current.mean
        else:
            better = record.mean > current.mean
        if better:
            best[key] = record
    return best


def _topline(summary: Sequence[SummaryRecord]):
    metric_labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "r2": "R²",
        "test_throughput_samples_sec": "Test Throughput (samples/s)",
        "ternary_zero_ratio": "Zero Ratio",
    }
    topline: MutableMapping[Tuple[str, str], Dict[str, SummaryRecord]] = defaultdict(dict)
    for record in summary:
        if record.metric not in metric_labels:
            continue
        key = (record.dataset, record.mode)
        current = topline[key].get(record.metric)
        if current is None or record.mean > current.mean:
            topline[key][record.metric] = record
    return metric_labels, topline


def _write_markdown(runs: Sequence[RunRecord], summary: Sequence[SummaryRecord]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORT_DIR / "benchmark_summary.md"
    grouped: MutableMapping[Tuple[str, str], List[SummaryRecord]] = defaultdict(list)
    for record in summary:
        grouped[(record.dataset, record.mode)].append(record)

    best = _best_by_dataset(summary)
    metric_labels, topline = _topline(summary)

    # Build variant→metrics view (for best-configs summary)
    by_variant: MutableMapping[Tuple[str, str, str], Dict[str, SummaryRecord]] = defaultdict(dict)
    for rec in summary:
        by_variant[(rec.dataset, rec.mode, rec.strategy_variant)][rec.metric] = rec

    def _primary_metric(dataset: str) -> str:
        return "r2" if dataset == "california_housing" else "accuracy"

    def _best_configs_lines() -> List[str]:
        rows: List[str] = []
        rows.append("## Best Configs\n")
        rows.append(
            "| Dataset | Mode | Primary | Best (μ±σ) | Variant | Flip | n | "
            "Baseline (μ±σ) | Δ | Effect Size |"
        )
        rows.append("|---|---|---|---|---|---|---:|---|---:|---:|")
        keys = sorted({(rec.dataset, rec.mode) for rec in summary})
        for dataset, mode in keys:
            primary = _primary_metric(dataset)
            variants = [v for (ds, md, v), _ in by_variant.items() if ds == dataset and md == mode]

            def _rec(variant: str, metric: str) -> SummaryRecord | None:
                return by_variant.get((dataset, mode, variant), {}).get(metric)

            # Best overall for primary metric
            best_rec: SummaryRecord | None = None
            for v in variants:
                r = _rec(v, primary)
                if r and (best_rec is None or r.mean > best_rec.mean):
                    best_rec = r
            # Best baseline among flip-off variants
            base_rec: SummaryRecord | None = None
            for v in variants:
                r = _rec(v, primary)
                if not r or r.flip != "off":
                    continue
                if base_rec is None or r.mean > base_rec.mean:
                    base_rec = r
            if best_rec is None:
                continue
            delta = best_rec.mean - base_rec.mean if base_rec is not None else float("nan")
            pooled = 0.0
            if base_rec is not None:
                pooled = ((best_rec.std or 0.0) ** 2 + (base_rec.std or 0.0) ** 2) / 2.0
                pooled = float(pooled) ** 0.5
            effect = (delta / pooled) if (base_rec is not None and pooled > 1e-12) else 0.0

            best_label = _format_pm(best_rec.mean, best_rec.std)
            base_label = _format_pm(base_rec.mean, base_rec.std) if base_rec else "—"
            flip_label = f"{best_rec.flip} ({best_rec.flip_schedule})"
            rows.append(
                "| "
                + " | ".join(
                    [
                        dataset,
                        mode,
                        primary,
                        best_label,
                        best_rec.strategy_variant,
                        flip_label,
                        str(best_rec.count),
                        base_label,
                        f"{delta:.4f}" if base_rec is not None else "—",
                        f"{effect:.3f}" if base_rec is not None else "—",
                    ]
                )
                + " |"
            )
        rows.append("")
        return rows

    lines: List[str] = []
    lines.append("# FeedFlipNets Benchmark Summary\n")
    lines.append("Aggregated over seeds with mean ± std.\n")
    lines.append("")

    if topline:
        lines.append("## Topline Highlights\n")
        lines.append("| Dataset | Mode | Metric | Mean ± Std | Strategy Variant | Flip | n |")
        lines.append("|---|---|---|---|---|---|---:|")
        for dataset, mode in sorted(topline.keys()):
            records = topline[(dataset, mode)]
            for metric in sorted(records.keys()):
                record = records[metric]
                label = metric_labels.get(metric, metric)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            dataset,
                            mode,
                            label,
                            _format_pm(record.mean, record.std),
                            record.strategy_variant,
                            f"{record.flip} ({record.flip_schedule})",
                            str(record.count),
                        ]
                    )
                    + " |"
                )
        lines.append("")

    # Best configs across modalities (primary metric)
    lines.extend(_best_configs_lines())

    for (dataset, mode), records in sorted(grouped.items()):
        lines.append(f"## {dataset} ({mode})\n")
        lines.append("| Strategy Variant | Flip | Metric | Mean ± Std | n |")
        lines.append("|---|---|---|---|---:|")
        records_sorted = sorted(records, key=lambda r: (r.strategy_variant, r.metric))
        for record in records_sorted:
            key = (dataset, mode)
            highlight = ""
            best_record = best.get(key)
            if (
                best_record
                and record.metric == best_record.metric
                and math.isclose(record.mean, best_record.mean)
            ):
                highlight = " **(best)**"
            lines.append(
                "| "
                + " | ".join(
                    [
                        record.strategy_variant,
                        f"{record.flip} ({record.flip_schedule})",
                        record.metric,
                        _format_pm(record.mean, record.std) + highlight,
                        str(record.count),
                    ]
                )
                + " |"
            )
        lines.append("")

    # Recommendations (appendix-ready prose)
    lines.append("## Recommendations\n")
    lines.append(
        "- Vision (MNIST, real): backprop_float with lr≈0.075 leads accuracy; "
        "if ternary forward is required, use DFA with per_epoch flips and τ≈0.05.\n"
    )
    lines.append(
        "- Text (20 Newsgroups, real): DFA float (lr≈0.05) remains most stable; "
        "ternary benefits from per_epoch flips with τ in [0.02, 0.05], accepting reduced accuracy "
        "for higher sparsity.\n"
    )
    lines.append("- Tabular (California Housing, real): DFA float with lr≈0.05 and grad_clip=1.0\n")
    lines.append("  improves robustness;\n")
    lines.append("  structured hadamard float is the throughput leader. Avoid per_step ternary.\n")
    lines.append(
        "- Time-series (UCR): accuracy saturates at 1.0 across methods; prefer ternary DFA\n"
    )
    lines.append("  for deployability and footprint.\n")
    lines.append("- Flip scheduling: prefer per_epoch over per_step on non-vision modalities.\n")
    lines.append("- Ternary threshold: start at τ=0.05 and adjust by modality (lower for text).\n")

    summary_path.write_text("\n".join(lines) + "\n")

    # Also export the Best Configs as a CSV one-pager for the paper
    best_rows_csv: List[Dict[str, object]] = []
    keys = sorted({(rec.dataset, rec.mode) for rec in summary})
    for dataset, mode in keys:
        primary = _primary_metric(dataset)
        variants = [v for (ds, md, v), _ in by_variant.items() if ds == dataset and md == mode]

        def _rec(variant: str, metric: str) -> SummaryRecord | None:
            return by_variant.get((dataset, mode, variant), {}).get(metric)

        best_rec = None
        for v in variants:
            r = _rec(v, primary)
            if r and (best_rec is None or r.mean > best_rec.mean):
                best_rec = r
        if best_rec is None:
            continue
        base_rec = None
        for v in variants:
            r = _rec(v, primary)
            if not r or r.flip != "off":
                continue
            if base_rec is None or r.mean > base_rec.mean:
                base_rec = r
        delta = best_rec.mean - base_rec.mean if base_rec is not None else float("nan")
        pooled = 0.0
        if base_rec is not None:
            pooled = ((best_rec.std or 0.0) ** 2 + (base_rec.std or 0.0) ** 2) / 2.0
            pooled = float(pooled) ** 0.5
        effect = (delta / pooled) if (base_rec is not None and pooled > 1e-12) else float("nan")
        best_rows_csv.append(
            {
                "dataset": dataset,
                "mode": mode,
                "primary": primary,
                "best_mean": float(best_rec.mean),
                "best_std": float(best_rec.std),
                "best_variant": best_rec.strategy_variant,
                "best_flip": best_rec.flip,
                "best_flip_schedule": best_rec.flip_schedule,
                "n": int(best_rec.count),
                "baseline_mean": (float(base_rec.mean) if base_rec else float("nan")),
                "baseline_std": (float(base_rec.std) if base_rec else float("nan")),
                "delta": float(delta) if math.isfinite(delta) else float("nan"),
                "effect_size": (float(effect) if math.isfinite(effect) else float("nan")),
            }
        )
    if best_rows_csv:
        out_csv = REPORT_DIR / "best_configs.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dataset",
                    "mode",
                    "primary",
                    "best_mean",
                    "best_std",
                    "best_variant",
                    "best_flip",
                    "best_flip_schedule",
                    "n",
                    "baseline_mean",
                    "baseline_std",
                    "delta",
                    "effect_size",
                ],
            )
            writer.writeheader()
            for row in best_rows_csv:
                writer.writerow(row)


def main() -> None:
    runs = list(_iter_runs(ROOT))
    if not runs:
        print("No runs found under runs/bench. Nothing to aggregate.")
        return

    summary = _aggregate(runs)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(REPORT_DIR / "benchmark_runs.json", [asdict(r) for r in runs])
    _write_json(REPORT_DIR / "benchmark_summary.json", [asdict(r) for r in summary])
    if summary:
        _write_csv(REPORT_DIR / "benchmark_summary.csv", summary)
    _write_markdown(runs, summary)
    print(f"Wrote aggregated artefacts to {REPORT_DIR}")


if __name__ == "__main__":
    main()
