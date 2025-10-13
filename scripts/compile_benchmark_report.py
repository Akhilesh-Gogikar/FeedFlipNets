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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

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
        # Optionally compute training throughput as well
        train_throughput = None
        try:
            train_timing = timing.get("train") if isinstance(timing, dict) else None
            if isinstance(train_timing, dict) and (train_timing.get("total_sec") or 0) > 0:
                mj = run_dir / "metrics_train.jsonl"
                if mj.exists():
                    last_line = ""
                    with mj.open("r", encoding="utf-8") as h:
                        for last_line in h:
                            pass
                    if last_line.strip():
                        rec = json.loads(last_line)
                        sc = rec.get("sample_count")
                        if isinstance(sc, (int, float)) and sc > 0:
                            train_throughput = float(sc) / float(train_timing.get("total_sec", 1.0))
        except Exception:
            train_throughput = None
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
        if train_throughput is not None:
            numeric_metrics["train_throughput_samples_sec"] = float(train_throughput)
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
    sem: float = 0.0
    ci95: float = 0.0


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
            # Use numpy for numerical robustness across Python versions
            mu = float(np.mean(values))
            sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            n = len(values)
            sem = (sigma / (n**0.5)) if n > 0 else 0.0
            ci95 = 1.96 * sem if n > 0 else 0.0
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
                    count=n,
                    sem=sem,
                    ci95=ci95,
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


def _format_pm(mu: float, err: float) -> str:
    if err == 0.0:
        return f"{mu:.4f}"
    return f"{mu:.4f} ± {err:.4f}"


def _best_by_dataset(summary: Sequence[SummaryRecord]) -> Dict[str, SummaryRecord]:
    best: Dict[str, SummaryRecord] = {}
    for record in summary:
        if record.metric not in {"accuracy", "macro_f1", "r2"}:
            continue
        key = (record.dataset, record.mode)
        current = best.get(key)
        # Skip non-finite means when considering bests
        if not math.isfinite(record.mean):
            continue
        if current is None or not math.isfinite(current.mean):
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
            "| Dataset | Mode | Primary | Best (μ±95% CI) | Variant | Flip | n | "
            "Baseline (μ±σ) | Δ | Effect Size |"
        )
        rows.append("|---|---|---|---|---|---|---:|---|---:|---:|")
        keys = sorted({(rec.dataset, rec.mode) for rec in summary})
        for dataset, mode in keys:
            primary = _primary_metric(dataset)
            variants = [v for (ds, md, v), _ in by_variant.items() if ds == dataset and md == mode]

            def _rec(variant: str, metric: str) -> SummaryRecord | None:
                rec = by_variant.get((dataset, mode, variant), {}).get(metric)
                if rec is None or not math.isfinite(rec.mean):
                    return None
                return rec

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

            best_label = _format_pm(best_rec.mean, best_rec.ci95)
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
        baseline_present = bool(base_rec is not None)
        note = ""
        if not baseline_present:
            note = "no_baseline"
        elif primary == "accuracy" and best_rec.mean >= 0.9995:
            note = "saturated"
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
                "baseline_present": baseline_present,
                "note": note,
            }
        )
    # Drop rows with non-finite best means to avoid NaNs in paper tables
    best_rows_csv = [
        r for r in best_rows_csv if math.isfinite(float(r.get("best_mean", float("nan"))))
    ]
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
                    "baseline_present",
                    "note",
                ],
            )
            writer.writeheader()
            for row in best_rows_csv:
                writer.writerow(row)

        # Best Configs — markdown-only table (paper-ready)
        md_path = REPORT_DIR / "best_configs_table.md"
        md_lines = [
            "| Dataset | Mode | Primary | Best (μ±σ) | Variant | Flip | n | "
            "Baseline (μ±σ) | Δ | Effect | Note |",
            "|---|---|---|---|---|---|---:|---|---:|---:|---|",
        ]
        for row in best_rows_csv:
            best_label = _format_pm(float(row["best_mean"]), float(row["best_std"]))
            if math.isfinite(float(row["baseline_mean"])):
                base_label = _format_pm(float(row["baseline_mean"]), float(row["baseline_std"]))
            else:
                base_label = "—"
            delta_str = f"{float(row['delta']):.4f}" if math.isfinite(float(row["delta"])) else "—"
            effect_str = (
                f"{float(row['effect_size']):.3f}"
                if math.isfinite(float(row["effect_size"]))
                else "—"
            )
            flip_label = f"{row['best_flip']} ({row['best_flip_schedule']})"
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["dataset"]),
                        str(row["mode"]),
                        str(row["primary"]),
                        best_label,
                        str(row["best_variant"]),
                        flip_label,
                        str(row["n"]),
                        base_label,
                        delta_str,
                        effect_str,
                        str(row["note"]),
                    ]
                )
                + " |"
            )
        md_path.write_text("\n".join(md_lines) + "\n")

        # Best Configs — LaTeX table (paper-ready)
        def _tex_escape(s: str) -> str:
            # Minimal escaping for LaTeX tabular
            if s is None:
                return ""
            s = str(s)
            s = s.replace("_", "\\_")
            # Normalize unicode em dash to LaTeX command
            s = s.replace("—", "\\textemdash{}")
            return s

        tex_path = REPORT_DIR / "best_configs_table.tex"
        # 11 columns: l l l c l l r c c c l
        # Use a minimal header without \hline to maximize engine compatibility
        header = "\\begin{tabular}{lllllllllll}\n"
        header += (
            "Dataset & Mode & Primary & Best (mean+/-std) & Variant & Flip & n & Baseline & "
            "Delta & Effect & Note \\\n"
        )
        rows_tex: List[str] = []
        for row in best_rows_csv:
            # Prefer UTF-8 ± using XeTeX to avoid math-mode \pm in tables
            best_label = f"{float(row['best_mean']):.4f} ± {float(row['best_std']):.4f}"
            if math.isfinite(float(row["baseline_mean"])):
                base_label = f"{float(row['baseline_mean']):.4f} ± {float(row['baseline_std']):.4f}"
            else:
                base_label = "\\textemdash{}"
            delta_str = (
                f"{float(row['delta']):.4f}"
                if math.isfinite(float(row["delta"]))
                else "\\textemdash{}"
            )
            effect_str = (
                f"{float(row['effect_size']):.3f}"
                if math.isfinite(float(row["effect_size"]))
                else "\\textemdash{}"
            )
            flip_label = f"{row['best_flip']} ({row['best_flip_schedule']})"
            rows_tex.append(
                " ".join(
                    [
                        _tex_escape(str(row["dataset"])),
                        "&",
                        _tex_escape(str(row["mode"])),
                        "&",
                        _tex_escape(str(row["primary"])),
                        "&",
                        best_label,
                        "&",
                        _tex_escape(str(row["best_variant"])),
                        "&",
                        _tex_escape(flip_label),
                        "&",
                        str(row["n"]),
                        "&",
                        _tex_escape(base_label),
                        "&",
                        _tex_escape(delta_str),
                        "&",
                        _tex_escape(effect_str),
                        "&",
                        _tex_escape(str(row["note"])) if row.get("note") else "",
                        "\\\\",
                    ]
                )
            )
        trailer = "\\end{tabular}\n"
        tex_path.write_text(header + "\n".join(rows_tex) + "\n" + trailer)


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
    # Auto-generate simple plots for LR and tau sweeps from summary
    plots_dir = REPORT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore
    if plt is not None:
        # Build lookup
        rec_map: Dict[Tuple[str, str, str, str], SummaryRecord] = {}
        for rec in summary:
            rec_map[(rec.dataset, rec.mode, rec.strategy_variant, rec.metric)] = rec
        keys = sorted({(rec.dataset, rec.mode) for rec in summary})
        lr_map = {"lr06": 0.03, "lr10": 0.05, "lr15": 0.075}
        for dataset, mode in keys:
            primary = "r2" if dataset == "california_housing" else "accuracy"
            X: List[float] = []
            y_back: List[float] = []
            e_back: List[float] = []
            y_dfa: List[float] = []
            e_dfa: List[float] = []
            for suffix, lr in lr_map.items():
                vb = f"backprop_float_{suffix}"
                vd = f"dfa_float_{suffix}"
                rb = rec_map.get((dataset, mode, vb, primary))
                rd = rec_map.get((dataset, mode, vd, primary))
                if rb is not None and rd is not None:
                    X.append(lr)
                    y_back.append(rb.mean)
                    e_back.append(rb.std)
                    y_dfa.append(rd.mean)
                    e_dfa.append(rd.std)
            if X:
                plt.figure(figsize=(4.8, 3.2))
                plt.errorbar(X, y_back, yerr=e_back, marker="o", label="backprop")
                plt.errorbar(X, y_dfa, yerr=e_dfa, marker="s", label="dfa")
                plt.xlabel("Learning rate")
                plt.ylabel(primary)
                plt.title(f"LR sweep — {dataset} ({mode})")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f"lr_sweep_{dataset}_{mode}.png", dpi=200)
                plt.close()
        for dataset, mode in keys:
            taus = [0.02, 0.05, 0.10]
            primary = "r2" if dataset == "california_housing" else "accuracy"
            X: List[float] = []
            Y: List[float] = []
            E: List[float] = []
            for tau in taus:
                # Our variants are named with tau*100, e.g., 0.02 -> 002
                v = f"dfa_ternary_epoch_tau{int(tau*100):03d}"
                r = rec_map.get((dataset, mode, v, primary))
                if r is not None:
                    X.append(tau)
                    Y.append(r.mean)
                    E.append(r.std)
            if X:
                plt.figure(figsize=(4.8, 3.2))
                plt.errorbar(X, Y, yerr=E, marker="o")
                plt.xlabel("tau (threshold)")
                plt.ylabel(primary)
                plt.title(f"Tau sweep — {dataset} ({mode})")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plots_dir / f"tau_sweep_{dataset}_{mode}.png", dpi=200)
                plt.close()
    print(f"Wrote aggregated artefacts to {REPORT_DIR}")


if __name__ == "__main__":
    main()
