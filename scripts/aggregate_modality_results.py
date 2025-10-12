#!/usr/bin/env python3
"""
Aggregate FeedFlipNets results across modalities into a single CSV/JSON/MD
under data/report/ for paper-ready tables.

This script expects runs to exist for these presets:
  - mnist_mlp_dfa
  - ucr_gunpoint_mlp_dfa
  - california_housing_mlp_dfa
  - 20newsgroups_bow_mlp_dfa

It reads each run's manifest and test metrics and writes:
  - data/report/modality_results.csv
  - data/report/modality_results.json
  - data/report/README.md (brief summary)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Row:
    preset: str
    modality: str
    dataset: str
    task_type: str
    run_dir: str
    epochs: Optional[int]
    batch_size: Optional[int]
    lr: Optional[float]
    flip: Optional[str]
    flip_schedule: Optional[str]
    seed: Optional[int]
    test_loss: Optional[float]
    accuracy: Optional[float]
    macro_f1: Optional[float]
    mae: Optional[float]
    rmse: Optional[float]
    r2: Optional[float]
    sample_count: Optional[float]
    samples_per_step: Optional[float]
    test_throughput_samples_sec: Optional[float]
    ternary_zero_ratio: Optional[float]


RUNS: Dict[str, str] = {
    "mnist_mlp_dfa": "runs/mnist-mlp-dfa",
    "ucr_gunpoint_mlp_dfa": "runs/ucr-gunpoint-mlp-dfa",
    "california_housing_mlp_dfa": "runs/california-housing-mlp-dfa",
    "20newsgroups_bow_mlp_dfa": "runs/20newsgroups-bow-mlp-dfa",
}


def modality_for(dataset: str) -> str:
    mapping = {
        "mnist": "vision",
        "ucr": "time_series",
        "california_housing": "tabular",
        "20newsgroups": "text",
    }
    return mapping.get(dataset, "unknown")


def task_type_for(dataset: str) -> str:
    if dataset == "california_housing":
        return "regression"
    return "multiclass"


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def collect() -> List[Row]:
    rows: List[Row] = []
    for preset, run_dir in RUNS.items():
        rdir = Path(run_dir)
        manifest = read_json(rdir / "manifest.json")
        test_metrics = read_json(rdir / "metrics_test.json")
        cfg = manifest.get("config", {}) if isinstance(manifest, dict) else {}
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        dataset = data_cfg.get("name") if isinstance(data_cfg, dict) else None
        dataset = str(dataset) if dataset is not None else "unknown"
        train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
        row = Row(
            preset=preset,
            modality=modality_for(dataset),
            dataset=dataset,
            task_type=task_type_for(dataset),
            run_dir=str(rdir),
            epochs=(
                int(train_cfg.get("epochs"))
                if isinstance(train_cfg, dict) and "epochs" in train_cfg
                else None
            ),
            batch_size=(
                int(train_cfg.get("batch_size"))
                if isinstance(train_cfg, dict) and "batch_size" in train_cfg
                else None
            ),
            lr=(
                float(train_cfg.get("lr"))
                if isinstance(train_cfg, dict) and "lr" in train_cfg
                else None
            ),
            flip=(
                str(train_cfg.get("flip"))
                if isinstance(train_cfg, dict) and "flip" in train_cfg
                else None
            ),
            flip_schedule=(
                str(train_cfg.get("flip_schedule"))
                if isinstance(train_cfg, dict) and "flip_schedule" in train_cfg
                else None
            ),
            seed=(
                int(train_cfg.get("seed"))
                if isinstance(train_cfg, dict) and "seed" in train_cfg
                else None
            ),
            test_loss=(
                float(test_metrics.get("loss"))
                if isinstance(test_metrics, dict) and "loss" in test_metrics
                else None
            ),
            accuracy=(
                float(test_metrics.get("accuracy"))
                if isinstance(test_metrics, dict) and "accuracy" in test_metrics
                else None
            ),
            macro_f1=(
                float(test_metrics.get("macro_f1"))
                if isinstance(test_metrics, dict) and "macro_f1" in test_metrics
                else None
            ),
            mae=(
                float(test_metrics.get("mae"))
                if isinstance(test_metrics, dict) and "mae" in test_metrics
                else None
            ),
            rmse=(
                float(test_metrics.get("rmse"))
                if isinstance(test_metrics, dict) and "rmse" in test_metrics
                else None
            ),
            r2=(
                float(test_metrics.get("r2"))
                if isinstance(test_metrics, dict) and "r2" in test_metrics
                else None
            ),
            sample_count=(
                float(test_metrics.get("sample_count"))
                if isinstance(test_metrics, dict) and "sample_count" in test_metrics
                else None
            ),
            samples_per_step=(
                float(test_metrics.get("samples_per_step"))
                if isinstance(test_metrics, dict) and "samples_per_step" in test_metrics
                else None
            ),
            test_throughput_samples_sec=(
                float(test_metrics.get("test_throughput_samples_sec"))
                if isinstance(test_metrics, dict) and "test_throughput_samples_sec" in test_metrics
                else None
            ),
            ternary_zero_ratio=(
                float(test_metrics.get("ternary_zero_ratio"))
                if isinstance(test_metrics, dict) and "ternary_zero_ratio" in test_metrics
                else None
            ),
        )
        rows.append(row)
    return rows


def write_csv(rows: List[Row], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json_rows(rows: List[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in rows]
    out_path.write_text(json.dumps(payload, indent=2))


def write_md(rows: List[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# FeedFlipNets Modality Results (offline fixtures)\n")
    lines.append("All runs were executed offline with deterministic seeds.\n")
    lines.append("Test-set metrics shown.\n")
    lines.append("")
    # Compact table
    lines.append(
        "| Preset | Modality | Dataset | Flip | Schedule | Epochs | Acc | Macro-F1 | "
        "MAE | RMSE | R2 | Samples/Step | Test Throughput (samples/s) | Zero Ratio | Run Dir |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        acc = f"{r.accuracy:.4f}" if r.accuracy is not None else ""
        f1 = f"{r.macro_f1:.4f}" if r.macro_f1 is not None else ""
        mae = f"{r.mae:.4f}" if r.mae is not None else ""
        rmse = f"{r.rmse:.4f}" if r.rmse is not None else ""
        r2 = f"{r.r2:.4f}" if r.r2 is not None else ""
        samples_step = f"{r.samples_per_step:.2f}" if r.samples_per_step is not None else ""
        throughput = (
            f"{r.test_throughput_samples_sec:.2f}"
            if r.test_throughput_samples_sec is not None
            else ""
        )
        zero_ratio = f"{r.ternary_zero_ratio:.3f}" if r.ternary_zero_ratio is not None else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    r.preset,
                    r.modality,
                    r.dataset,
                    (r.flip or ""),
                    (r.flip_schedule or ""),
                    str(r.epochs or ""),
                    acc,
                    f1,
                    mae,
                    rmse,
                    r2,
                    samples_step,
                    throughput,
                    zero_ratio,
                    r.run_dir,
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = collect()
    report_dir = Path("data/report")
    write_csv(rows, report_dir / "modality_results.csv")
    write_json_rows(rows, report_dir / "modality_results.json")
    write_md(rows, report_dir / "README.md")


if __name__ == "__main__":
    main()
