#!/usr/bin/env python3
"""Run optional CNN baselines and ingest them into runs/bench.

This small harness calls `scripts/baselines/cnn_dfa_baselines.py` and writes
bench-compatible manifests/metrics so they appear in the aggregated report.

Example:
  # Fashion‑MNIST bp/dfa with 2 seeds
  python scripts/run_cnn_baselines.py --datasets fashion_mnist \
      --methods bp dfa --seeds 0 1 --epochs 2 --batch-size 128 --limit 5000

  # CIFAR‑10 bp only
  python scripts/run_cnn_baselines.py --datasets cifar10 --methods bp --seeds 0

Requires extras: `pip install -r requirements-extras.txt`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def _call_cnn_script(
    dataset: str,
    method: str,
    epochs: int,
    batch_size: int,
    lr: float,
    limit: int | None,
    seed: int,
    tmp_dir: Path,
) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        str(Path("scripts/baselines/cnn_dfa_baselines.py")),
        "--dataset",
        dataset,
        "--method",
        method,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--run-dir",
        str(tmp_dir),
    ]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Find the last JSON line containing the saved path
    saved_path: Path | None = None
    for line in proc.stdout.strip().splitlines()[::-1]:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict) and "saved" in rec:
            saved_path = Path(rec["saved"]).resolve()
            break
    if not saved_path or not saved_path.exists():
        raise RuntimeError("CNN baseline did not produce a metrics JSON file")
    return saved_path


def _ingest(
    metrics_path: Path,
    *,
    dataset: str,
    method: str,
    seed: int,
    base_run_dir: Path,
) -> None:
    # Read epoch metrics
    payload = json.loads(metrics_path.read_text())
    test_hist = payload.get("test", []) if isinstance(payload, dict) else []
    if not test_hist:
        raise RuntimeError(f"No test history in {metrics_path}")
    last = test_hist[-1]
    accuracy = float(last.get("acc", 0.0))
    loss = float(last.get("loss", 0.0))

    variant = f"cnn_{method}"
    run_dir = base_run_dir / dataset / "real" / variant / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal manifest/config compatible with compiler expectations
    manifest = {
        "config": {
            "data": {"name": dataset, "options": {"seed": seed}},
            "model": {"strategy": variant},
            "train": {
                "epochs": int(payload.get("train", [{}])[-1].get("epoch", 0)) if payload.get("train") else 0,
                "seed": seed,
                "flip": "off",
                "flip_schedule": "off",
            },
        },
        "environment": {"offline": 0},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Write a minimal test metrics JSON; compiler will pick up accuracy
    (run_dir / "metrics_test.json").write_text(json.dumps({"accuracy": accuracy, "loss": loss}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=["fashion_mnist", "cifar10"], default=["fashion_mnist"])
    p.add_argument("--methods", nargs="+", choices=["bp", "dfa"], default=["bp", "dfa"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--tmp-out", type=Path, default=Path("runs/baselines"))
    p.add_argument("--base-run-dir", type=Path, default=Path("runs/bench"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        for method in args.methods:
            for seed in (args.seeds or [0, 1, 2]):
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "status": "running",
                        }
                    )
                )
                metrics_path = _call_cnn_script(
                    dataset,
                    method,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    limit=args.limit,
                    seed=seed,
                    tmp_dir=args.tmp_out,
                )
                _ingest(
                    metrics_path,
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    base_run_dir=args.base_run_dir,
                )
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "status": "ingested",
                        }
                    )
                )


if __name__ == "__main__":
    main()

