#!/usr/bin/env python3
"""Generate a LaTeX deployability table from recorded runs.

For each run directory (e.g., runs/mnist-mlp-dfa), we extract:
- Parameter count (from checkpoint shapes)
- Forward bit-width (assumed ternary => 2 bits)
- Sparsity per layer (fraction of zeros after ternary quantization)
- Peak RAM during inference (weights at forward bit-width + max activation for batch=1)
- CPU latency/inference (from test timings divided by total tested samples)

Outputs docs/paper/_generated/deployability_table.tex.

Usage:
  python scripts/gen_deployability_table.py \
      --runs runs/mnist-mlp-dfa runs/20newsgroups-bow-mlp-dfa \
      --out docs/paper/_generated/deployability_table.tex

Notes:
- Energy/inference is left as N/A with a footnote about MLPerf Tiny/MLMark.
- Results depend on the local CPU; include device string for transparency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import platform
from typing import Dict, List, Tuple

import numpy as np


def _load_ckpt_shapes_and_arrays(path: pathlib.Path) -> Tuple[List[Tuple[int, int]], Dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    shapes: List[Tuple[int, int]] = []
    for k in sorted(arrays):
        if not k.startswith("W"):
            continue
        w = arrays[k]
        if w.ndim != 2:
            raise ValueError(f"unexpected tensor rank for {k}: shape={w.shape}")
        shapes.append((int(w.shape[0]), int(w.shape[1])))
    return shapes, arrays


def _sparsity_per_layer(arrays: Dict[str, np.ndarray]) -> List[float]:
    ratios: List[float] = []
    for k in sorted(arrays):
        if not k.startswith("W"):
            continue
        w = arrays[k]
        zeros = float(np.count_nonzero(np.isclose(w, 0.0)))
        total = float(w.size)
        ratios.append(0.0 if total == 0 else zeros / total)
    return ratios


def _timing_latency(run_dir: pathlib.Path) -> float | None:
    """Return average test latency per sample in milliseconds, if available."""
    timing_path = run_dir / "timing.json"
    test_metrics_path = run_dir / "metrics_test.jsonl"
    if not timing_path.exists() or not test_metrics_path.exists():
        return None
    try:
        timing = json.loads(timing_path.read_text())
        test_total = float(timing.get("test", {}).get("total_sec", 0.0))
        # Count how many test epochs we logged
        test_epochs = 0
        samples_per_epoch = None
        with test_metrics_path.open("r") as fh:
            for line in fh:
                test_epochs += 1
                if samples_per_epoch is None:
                    rec = json.loads(line)
                    samples_per_epoch = int(rec.get("sample_count", 0))
        if not test_epochs or not samples_per_epoch:
            return None
        total_samples = test_epochs * samples_per_epoch
        if total_samples <= 0 or test_total <= 0:
            return None
        latency_sec = test_total / float(total_samples)
        return latency_sec * 1e3
    except Exception:
        return None


def _peak_ram_kb(shapes: List[Tuple[int, int]], forward_bits: int = 2) -> int:
    """Compute peak RAM during inference for batch=1 (rough estimate).

    - Weights at `forward_bits` per parameter.
    - Activations as float32 (4 bytes) for the largest layer dimension.
    """
    # Weights memory
    params = sum(int(m * n) for m, n in shapes)
    weights_bytes = int(math.ceil(params * forward_bits / 8.0))
    # Activations (batch=1): max layer width in the chain (including input/output)
    dims: List[int] = [shapes[0][0]] + [n for _, n in shapes]
    max_width = max(dims) if dims else 0
    activ_bytes = max_width * 4  # float32 inference activations
    return int((weights_bytes + activ_bytes) / 1024)


def _model_desc(shapes: List[Tuple[int, int]]) -> str:
    dims = [shapes[0][0]] + [n for _, n in shapes]
    return "-".join(str(x) for x in dims)


def _device_str() -> str:
    parts = [platform.system(), platform.machine()]
    proc = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    if proc:
        parts.append(proc)
    return " ".join(p for p in parts if p)


def format_sparsity_list(vals: List[float]) -> str:
    # Render as [0.67, 0.45, ...] with two decimals
    return "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories under runs/")
    ap.add_argument(
        "--out",
        default="docs/paper/_generated/deployability_table.tex",
        help="Output LaTeX path",
    )
    args = ap.parse_args()

    rows: List[str] = []
    device = _device_str()
    for run in args.runs:
        run_dir = pathlib.Path(run)
        if not run_dir.exists():
            print(f"warning: run not found: {run}")
            continue
        # Prefer last.ckpt (final) if present, else best.ckpt
        ckpt = run_dir / "last.ckpt"
        if not ckpt.exists():
            ckpt = run_dir / "best.ckpt"
        shapes, arrays = _load_ckpt_shapes_and_arrays(ckpt)
        params = sum(int(m * n) for m, n in shapes)
        forward_bits = 2  # ternary forward path
        sparsity = _sparsity_per_layer(arrays)
        peak_kb = _peak_ram_kb(shapes, forward_bits=forward_bits)
        latency_ms = _timing_latency(run_dir)
        # Derive a short name
        name = run_dir.name.replace("-", " ")
        model = _model_desc(shapes)
        latency_str = f"{latency_ms:.3f}" if latency_ms is not None else "N/A"
        energy_str = "N/A\footnotesize{ (MLPerf Tiny/MLMark)}"
        # Params in thousands
        params_k = f"{params/1e3:.1f}k"
        row = (
            f"{name} ({model}) & {params_k} & {forward_bits} & "
            f"{format_sparsity_list(sparsity)} & {peak_kb} & {latency_str} & {energy_str} & {device} \\\\"
        )
        rows.append(row)

    header = r"""
\begin{table}[H]
\centering
\small
\begin{tabular}{l r c l r r l l}
\toprule
Model & Params & Bits & Sparsity (per layer) & Peak RAM (KB) & CPU Lat. (ms) & Energy/inf. & Device \\
\midrule
"""
    footer = r"""
\bottomrule
\end{tabular}
\caption{Deployability metrics on a single CPU device. Forward path uses ternary (2-bit) weights. Peak RAM estimates include quantized weights and batch=1 activations. Energy/inference left as N/A; plan to follow MLPerf Tiny/EEMBC MLMark methodology.}
\end{table}
"""

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = header + "\n".join(rows) + "\n" + footer
    out_path.write_text(content)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
