#!/usr/bin/env python3
"""
Generate a simple deployability table for the paper.

Metrics:
- Params (count)
- Forward bit-width (weights)
- Sparsity by layer (quantized at tau)
- Peak RAM during inference (packed ternary weights + max activations)
- CPU latency per inference (1-sample, NumPy) on host device

Outputs LaTeX to docs/paper/_generated/deployability_table.tex
and a Markdown copy to data/report/deployability.md

Scope: multiple representative presets (MNIST, 20NG, AG News, UCR GunPoint).
"""
from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Import modules directly from files to avoid executing package __init__ (which pulls heavy deps)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Local, minimal implementations to avoid importing the full package
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=np.float32)


def quantize_ternary_det(weights: np.ndarray, tau: float) -> np.ndarray:
    out = np.zeros_like(weights, dtype=float)
    out[weights > tau] = 1.0
    out[weights < -tau] = -1.0
    return out


class FeedForwardModel:
    def __init__(self, layer_dims: List[int], tau: float, quant: str = "det", seed: int = 0):
        self.layer_dims = list(layer_dims)
        self.tau = float(tau)
        self.quant = str(quant)
        self.seed = int(seed)
        self.reset(seed)

    def reset(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        dims = self.layer_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            W = rng.standard_normal((in_dim, out_dim), dtype=np.float32) * 0.05
            self.weights.append(W)

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, None]:
        x = inputs
        for idx, W in enumerate(self.weights):
            z = x @ W
            if idx < len(self.weights) - 1:
                x = relu(z)
            else:
                x = z
        return x, None


OUT_TEX = ROOT / "docs/paper/_generated/deployability_table.tex"
OUT_MD = ROOT / "data/report/deployability.md"
OUT_MCU = ROOT / "data/report/mcu_estimates.md"


@dataclass
class DeviceInfo:
    name: str
    os: str
    arch: str


def detect_device() -> DeviceInfo:
    # Try macOS CPU brand string first
    name = ""
    try:
        import subprocess

        name = (
            subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                check=False,
                text=True,
            ).stdout.strip()
            or ""
        )
    except Exception:
        name = ""
    if not name:
        name = platform.processor() if hasattr(sys, "platform") else "CPU"

    os_name = f"{platform.system()} {platform.release()}"
    arch = platform.machine() or platform.processor() or "unknown"
    return DeviceInfo(name=name, os=os_name, arch=arch)


# --- Preset dimension helpers (avoid importing heavy modules) ---


@dataclass
class Profile:
    label: str
    dims: List[int]
    tau: float


def _parse_hidden_from_yaml(path: Path, default: List[int]) -> List[int]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return default
    import re

    m = re.search(r"hidden:\s*\[(.*?)\]", text)
    if not m:
        return default
    try:
        parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
        return [int(p) for p in parts]
    except Exception:
        return default


def dims_mnist() -> Profile:
    hidden = _parse_hidden_from_yaml(
        ROOT / "configs/presets/mnist_mlp_dfa.yaml", default=[256, 256]
    )
    return Profile(label="MNIST MLP (784-256-256-10)", dims=[784, *hidden, 10], tau=0.05)


def dims_20ng() -> Profile:
    # Canonical 20 Newsgroups has 20 classes; n_features from preset (default 2048)
    path = ROOT / "configs/presets/20newsgroups_bow_mlp_dfa.yaml"
    n_features = 2048
    hidden = [512, 256]
    try:
        import re

        txt = path.read_text(encoding="utf-8")
        m = re.search(r"n_features:\s*(\d+)", txt)
        if m:
            n_features = int(m.group(1))
        hidden = _parse_hidden_from_yaml(path, default=hidden)
    except Exception:
        pass
    return Profile(
        label=f"20NG BoW MLP ({n_features}-{'-'.join(map(str, hidden))}-20)",
        dims=[n_features, *hidden, 20],
        tau=0.05,
    )


def dims_ag_news() -> Profile:
    # bow_dim = min(max(2048, svd_dim or 2048), tfidf_max_features)
    path = ROOT / "configs/presets/ag_news_tfidf_mlp_dfa.yaml"
    svd_dim = 2048
    tfidf_max = 30000
    hidden = [512, 128]
    try:
        import re

        txt = path.read_text(encoding="utf-8")
        m = re.search(r"svd_dim:\s*(\d+)", txt)
        if m:
            svd_dim = int(m.group(1))
        m2 = re.search(r"tfidf_max_features:\s*(\d+)", txt)
        if m2:
            tfidf_max = int(m2.group(1))
        hidden = _parse_hidden_from_yaml(path, default=hidden)
    except Exception:
        pass
    d_in = min(max(2048, svd_dim or 2048), tfidf_max)
    return Profile(
        label=f"AG News TFIDF MLP ({d_in}-{'-'.join(map(str, hidden))}-4)",
        dims=[d_in, *hidden, 4],
        tau=0.05,
    )


def dims_ucr_gunpoint() -> Profile:
    # Canonical UCR GunPoint: length=150, binary classification (2 classes)
    hidden = _parse_hidden_from_yaml(
        ROOT / "configs/presets/ucr_gunpoint_mlp_dfa.yaml", default=[128, 128]
    )
    return Profile(
        label=f"UCR GunPoint MLP (150-{'-'.join(map(str, hidden))}-2)",
        dims=[150, *hidden, 2],
        tau=0.05,
    )


def layer_param_counts(dims: List[int]) -> List[int]:
    counts: List[int] = []
    for i in range(len(dims) - 1):
        counts.append(int(dims[i] * dims[i + 1]))
    return counts


def per_layer_sparsity(weights: List[np.ndarray], tau: float) -> List[float]:
    spars: List[float] = []
    for W in weights:
        Wq = quantize_ternary_det(W.astype(np.float32), float(tau))
        zero = float(np.count_nonzero(np.isclose(Wq, 0.0)))
        spars.append(zero / float(Wq.size) if Wq.size else 0.0)
    return spars


def packed_ternary_bytes(n_params: int) -> int:
    # 2 bits per param -> bytes
    bits = n_params * 2
    return (bits + 7) // 8


def peak_activation_bytes(dims: List[int]) -> int:
    # Float32 activations, take max across layers including input
    max_acts = max(int(d) for d in dims)
    return max_acts * 4


def time_forward_latency(
    dims: List[int], runs: int = 1000, warmup: int = 50
) -> Tuple[float, float]:
    """
    Return (median_ms, p95_ms) latency for 1-sample forward on host CPU.
    Uses NumPy matmuls in FeedForwardModel.
    """
    # Build model with small tau; quantization not used in forward timing
    model = FeedForwardModel(layer_dims=dims, tau=0.05, quant="det", seed=0)
    x = np.random.default_rng(0).standard_normal((1, dims[0])).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        model.forward(x)

    latencies: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        model.forward(x)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)  # ms

    lat = np.array(latencies)
    return float(np.median(lat)), float(np.percentile(lat, 95))


def format_sparsities(spars: List[float]) -> str:
    # e.g., [0.62, 0.68, 0.71] -> 0.62 | 0.68 | 0.71
    return " | ".join(f"{s:.2f}" for s in spars)


def main() -> int:
    profiles = [dims_mnist(), dims_20ng(), dims_ag_news(), dims_ucr_gunpoint()]
    device = detect_device()

    # Prepare LaTeX table
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TEX.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by scripts/generate_deployability_table.py\n")
        f.write("\\begin{table}[H]\\centering\\small\n")
        f.write(
            "\\begin{tabular}{l r r l r r l}\\toprule\n"
            "Model "
            "& Params "
            "& Bit-width "
            "& Sparsity by layer "
            "& Peak RAM (KB) "
            "& CPU Latency (ms) "
            "& Energy/inf. \\\\ \\midrule\n"
        )
        for prof in profiles:
            model = FeedForwardModel(layer_dims=prof.dims, tau=prof.tau, quant="det", seed=0)
            spars = per_layer_sparsity(model.weights, prof.tau)
            n_params = sum(layer_param_counts(prof.dims))
            w_bytes = packed_ternary_bytes(n_params)
            act_bytes = peak_activation_bytes(prof.dims)
            peak_kb = (w_bytes + act_bytes) / 1024.0
            med_ms, p95_ms = time_forward_latency(prof.dims, runs=1000, warmup=50)
            f.write(
                f"{prof.label} on {device.name} "
                f"& {n_params} "
                f"& 2 (ternary) "
                f"& {format_sparsities(spars)} "
                f"& {peak_kb:.1f} "
                f"& {med_ms:.3f} (p95 {p95_ms:.3f}) "
                f"& N/A (MLPerf Tiny/EEMBC MLMark) \\\\ \n"
            )
        f.write("\\bottomrule\n")
        f.write(
            "\\end{tabular}"
            "\\caption{Deployability profile. Bit-width counts weights only; peak RAM assumes packed ternary weights (2 bits/param) + max float32 activations; latency measured with 1-sample NumPy forward on the host CPU and reported in milliseconds (ms). "  # noqa: E501
            "Values below 1 ms appear with three decimals (e.g., 0.012 ms = 12\\,\\textmu s). Energy/inference is N/A; for hardware measurements, follow MLPerf Tiny and EEMBC MLMark conventions (device profile, median/p95 latencies and energies, toolchain).}"  # noqa: E501
            "\\end{table}\n"
        )

    # Markdown summary for README/data/report
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# Deployability (host profile)\n\n")
        f.write(f"Device: {device.name} ({device.os}, {device.arch})\n\n")
        f.write(
            "| Model | Params | Bit-width | Sparsity by layer | Peak RAM (KB) | CPU Latency (ms) | Energy/inf. |\n"
        )
        f.write("|---|---:|:---:|:---|---:|---:|:---|\n")
        for prof in profiles:
            model = FeedForwardModel(layer_dims=prof.dims, tau=prof.tau, quant="det", seed=0)
            spars = per_layer_sparsity(model.weights, prof.tau)
            n_params = sum(layer_param_counts(prof.dims))
            w_bytes = packed_ternary_bytes(n_params)
            act_bytes = peak_activation_bytes(prof.dims)
            peak_kb = (w_bytes + act_bytes) / 1024.0
            med_ms, p95_ms = time_forward_latency(prof.dims, runs=300, warmup=40)
            md_spars = ", ".join(f"{s:.2f}" for s in spars)
            f.write(
                f"| {prof.label} | {n_params} | 2 (ternary) | {md_spars} | {peak_kb:.1f} | {med_ms:.3f} (p95 {p95_ms:.3f}) | N/A (MLPerf Tiny/EEMBC) |\n"  # noqa: E501
            )
        f.write(
            "\nNotes: weights bit-width excludes activations; "
            "peak RAM uses packed ternary for weights and float32 activations; "
            "latency uses NumPy forward (BLAS).\n"  # noqa: E501
            "latency uses NumPy forward (BLAS).\n"  # noqa: E501
        )  # noqa: E501

    # MCU estimates (ops-based)
    mcu_clock_mhz = float(os.environ.get("FFN_MCU_CLOCK_MHZ", "100"))
    cycles_per_mac = float(os.environ.get("FFN_MCU_CYCLES_PER_MAC", "1.0"))
    OUT_MCU.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MCU.open("w", encoding="utf-8") as f:
        f.write("# MCU Estimates (ops-based)\n\n")
        f.write(
            f"Assumptions: cycles_per_mac={cycles_per_mac}, clock={mcu_clock_mhz} MHz; nonzero-weight MACs only.\n\n"
        )
        f.write("| Model | Nonzero MACs | Cycles (est.) | Latency (ms, est.) |\n")
        f.write("|---|---:|---:|---:|\n")
        for prof in profiles:
            model = FeedForwardModel(layer_dims=prof.dims, tau=prof.tau, quant="det", seed=0)
            spars = per_layer_sparsity(model.weights, prof.tau)
            nonzero_macs = 0.0
            for (din, dout), sp in zip(zip(prof.dims[:-1], prof.dims[1:]), spars):
                total = float(din * dout)
                nonzero_macs += total * (1.0 - float(sp))
            cycles = nonzero_macs * cycles_per_mac
            latency_ms = cycles / (mcu_clock_mhz * 1e6) * 1e3
            f.write(f"| {prof.label} | {int(nonzero_macs)} | {int(cycles)} | {latency_ms:.3f} |\n")

    print(f"Wrote {OUT_TEX}, {OUT_MD} and {OUT_MCU}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
