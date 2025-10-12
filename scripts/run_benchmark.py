#!/usr/bin/env python3
"""Comprehensive FeedFlipNets benchmark runner.

This orchestrates multi-modality sweeps across datasets, feedback
strategies, flip schedules, and seeds. Results are written to
``runs/bench/<dataset>/<mode>/<variant>/seed<seed>``.

Example usage (defaults to full sweep with seeds 0,1,2):

    python scripts/run_benchmark.py --seeds 0 1 2

Selectively run pieces, e.g. only MNIST offline:

    python scripts/run_benchmark.py --datasets mnist --modes offline

After completion, run ``python scripts/compile_benchmark_report.py`` to
consolidate the results into ``data/report`` artefacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Sequence

from feedflipnets.training import pipelines


@dataclass(frozen=True)
class ModeConfig:
    name: str
    offline: bool
    epoch_multiplier: float = 1.2
    data_overrides: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    preset: str
    modes: Sequence[ModeConfig]
    data_overrides: Mapping[str, object] = field(default_factory=dict)
    train_overrides: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantConfig:
    identifier: str
    strategy: str
    flip: str
    flip_schedule: str
    model_overrides: Mapping[str, object] = field(default_factory=dict)
    train_overrides: Mapping[str, object] = field(default_factory=dict)


MODE_OFFLINE = ModeConfig(name="offline", offline=True, epoch_multiplier=1.0)
MODE_REAL = ModeConfig(name="real", offline=False, epoch_multiplier=1.2)

VARIANTS: Sequence[VariantConfig] = (
    VariantConfig("backprop_float", strategy="backprop", flip="off", flip_schedule="off"),
    VariantConfig(
        "backprop_ternary_step",
        strategy="backprop",
        flip="ternary",
        flip_schedule="per_step",
    ),
    VariantConfig("dfa_float", strategy="dfa", flip="off", flip_schedule="off"),
    VariantConfig("dfa_ternary_step", strategy="dfa", flip="ternary", flip_schedule="per_step"),
    VariantConfig("dfa_ternary_epoch", strategy="dfa", flip="ternary", flip_schedule="per_epoch"),
    VariantConfig(
        "ternary_dfa_step",
        strategy="ternary_dfa",
        flip="ternary",
        flip_schedule="per_step",
    ),
    VariantConfig(
        "structured_orth_float",
        strategy="structured",
        flip="off",
        flip_schedule="off",
        model_overrides={
            "structure_type": "orthogonal",
            "feedback_refresh": "per_step",
        },
    ),
    VariantConfig(
        "structured_orth_ternary",
        strategy="structured",
        flip="ternary",
        flip_schedule="per_step",
        model_overrides={
            "structure_type": "orthogonal",
            "feedback_refresh": "per_step",
        },
    ),
    VariantConfig(
        "structured_hadamard_float",
        strategy="structured",
        flip="off",
        flip_schedule="off",
        model_overrides={
            "structure_type": "hadamard",
            "feedback_refresh": "per_step",
        },
    ),
    VariantConfig(
        "structured_hadamard_ternary",
        strategy="structured",
        flip="ternary",
        flip_schedule="per_step",
        model_overrides={
            "structure_type": "hadamard",
            "feedback_refresh": "per_step",
        },
    ),
)

DATASETS: Sequence[DatasetConfig] = (
    DatasetConfig(name="mnist", preset="mnist_mlp_dfa", modes=(MODE_OFFLINE, MODE_REAL)),
    DatasetConfig(name="ucr", preset="ucr_gunpoint_mlp_dfa", modes=(MODE_OFFLINE, MODE_REAL)),
    DatasetConfig(
        name="california_housing",
        preset="california_housing_mlp_dfa",
        modes=(MODE_OFFLINE, MODE_REAL),
    ),
    DatasetConfig(
        name="20newsgroups",
        preset="20newsgroups_bow_mlp_dfa",
        modes=(MODE_OFFLINE, MODE_REAL),
    ),
)


def _prepare_config(
    dataset: DatasetConfig,
    mode: ModeConfig,
    variant: VariantConfig,
    seed: int,
) -> Dict[str, object]:
    base = pipelines.load_preset(dataset.preset)
    config = json.loads(json.dumps(base))

    config["offline"] = bool(mode.offline)

    data_cfg = config.setdefault("data", {})
    options = data_cfg.setdefault("options", {})
    if dataset.data_overrides:
        options.update(dataset.data_overrides)
    if mode.data_overrides:
        options.update(mode.data_overrides)
    options.setdefault("seed", seed)
    data_cfg["options"] = options

    model_cfg = config.setdefault("model", {})
    model_cfg["strategy"] = variant.strategy
    if variant.model_overrides:
        model_cfg.update(variant.model_overrides)

    train_cfg = config.setdefault("train", {})
    if dataset.train_overrides:
        train_cfg.update(dataset.train_overrides)
    if variant.train_overrides:
        train_cfg.update(variant.train_overrides)

    base_epochs = int(train_cfg.get("epochs", 1))
    scaled_epochs = max(1, math.ceil(base_epochs * mode.epoch_multiplier))
    train_cfg["epochs"] = scaled_epochs
    train_cfg["seed"] = seed

    train_cfg["flip"] = variant.flip
    if variant.flip == "off":
        train_cfg["flip_schedule"] = "off"
    else:
        train_cfg["flip_schedule"] = variant.flip_schedule

    train_cfg.setdefault("eval_every", 1)

    config["train"] = train_cfg
    config["model"] = model_cfg
    config["data"] = data_cfg
    return config


def _run_single(
    dataset: DatasetConfig,
    mode: ModeConfig,
    variant: VariantConfig,
    seed: int,
    *,
    skip_existing: bool,
    base_run_dir: Path,
) -> None:
    run_dir = base_run_dir / dataset.name / mode.name / variant.identifier / f"seed{seed}"
    manifest = run_dir / "manifest.json"
    if skip_existing and manifest.exists():
        print(f"[skip] {run_dir} (manifest exists)")
        return

    config = _prepare_config(dataset, mode, variant, seed)
    config.setdefault("train", {})["run_dir"] = str(run_dir)

    os.environ["FEEDFLIP_DATA_OFFLINE"] = "1" if mode.offline else "0"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[run] dataset={dataset.name} mode={mode.name} variant={variant.identifier} "
        f"seed={seed} -> {run_dir}"
    )

    result = pipelines.run_pipeline(config)
    if isinstance(result, list):
        for item in result:
            print(item)
    else:
        print(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", help="Subset of dataset names to run")
    parser.add_argument("--modes", nargs="*", choices=["offline", "real"], help="Subset of modes")
    parser.add_argument("--variants", nargs="*", help="Subset of variant identifiers")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2], help="Seeds to run")
    parser.add_argument(
        "--base-run-dir",
        default="runs/bench",
        help="Base directory where benchmark runs are written",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run benchmarks even if manifests already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_datasets = set(args.datasets) if args.datasets else None
    requested_modes = set(args.modes) if args.modes else None
    requested_variants = set(args.variants) if args.variants else None
    seeds: Sequence[int] = args.seeds if args.seeds else [0, 1, 2]
    skip_existing = not args.no_skip_existing
    base_run_dir = Path(args.base_run_dir)

    dataset_map = {cfg.name: cfg for cfg in DATASETS}
    variant_map = {cfg.identifier: cfg for cfg in VARIANTS}

    if requested_datasets:
        missing = requested_datasets - dataset_map.keys()
        if missing:
            sys.exit(f"Unknown dataset(s): {', '.join(sorted(missing))}")

    if requested_variants:
        missing = requested_variants - variant_map.keys()
        if missing:
            sys.exit(f"Unknown variant(s): {', '.join(sorted(missing))}")

    for dataset in DATASETS:
        if requested_datasets and dataset.name not in requested_datasets:
            continue
        for mode in dataset.modes:
            if requested_modes and mode.name not in requested_modes:
                continue
            for variant in VARIANTS:
                if requested_variants and variant.identifier not in requested_variants:
                    continue
                for seed in seeds:
                    _run_single(
                        dataset,
                        mode,
                        variant,
                        seed,
                        skip_existing=skip_existing,
                        base_run_dir=base_run_dir,
                    )


if __name__ == "__main__":
    main()
