#!/usr/bin/env python3
"""
Export trained FeedFlipNets weights to MCU-friendly formats.

Inputs
- --ckpt: path to a checkpoint saved by the Trainer (best.ckpt/last.ckpt)
- --tau:  ternary threshold used for quantization (default: 0.05)
- --out:  output directory (default: artifacts/export)
- --format: npz | bin | c | all (default: all)
- --name: optional model name label (default: inferred from ckpt dir)

Outputs
- NPZ: <name>_ternary_int8.npz with int8 {-1,0,+1} layer weights and dims
- BIN: <name>_ternary_packed.bin with 2-bit packed weights and
       <name>_manifest.json describing shapes, offsets, and code map
- C header: <name>_ternary.h with dims and int8 {-1,0,+1} arrays per layer

Notes
- Pack mapping (2-bit): 0 -> 0, 1 -> +1, 2 -> -1, 3 -> unused
- Weight matrix layout follows training: (in_dim, out_dim), row-major
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _infer_dims_from_ckpt(keys: List[str], arrays: Dict[str, np.ndarray]) -> List[int]:
    layers = sorted([k for k in keys if k.startswith("W")], key=lambda k: int(k[1:]))
    if not layers:
        raise ValueError("Checkpoint has no weight arrays W0, W1, ...")
    dims: List[int] = [int(arrays[layers[0]].shape[0])]
    for k in layers:
        w = arrays[k]
        if w.ndim != 2:
            raise ValueError(f"Weight {k} must be 2D, got shape {w.shape}")
        dims.append(int(w.shape[1]))
    return dims


def _quantize_ternary_det(W: np.ndarray, tau: float) -> np.ndarray:
    out = np.zeros_like(W, dtype=np.int8)
    out[W > tau] = 1
    out[W < -tau] = -1
    return out


def _pack_2bit(codes: np.ndarray) -> bytes:
    """Pack an array of 0/1/2 codes into 2-bit packed bytes (row-major)."""
    flat = codes.astype(np.uint8).ravel()
    # pack 4 values per byte: v0 in bits [1:0], v1 in [3:2], v2 in [5:4], v3 in [7:6]
    n = flat.size
    out = np.zeros((n + 3) // 4, dtype=np.uint8)
    for i in range(n):
        byte_index = i // 4
        shift = (i % 4) * 2
        out[byte_index] |= (flat[i] & 0x03) << shift
    return out.tobytes()


def _codes_from_int8(tern: np.ndarray) -> np.ndarray:
    # Map int8 {-1,0,+1} -> {2,0,1}
    codes = np.zeros_like(tern, dtype=np.uint8)
    codes[tern == 0] = 0
    codes[tern == 1] = 1
    codes[tern == -1] = 2
    return codes


def export_npz(name: str, outdir: Path, dims: List[int], layers: Dict[str, np.ndarray], tau: float) -> Path:
    path = outdir / f"{name}_ternary_int8.npz"
    save: Dict[str, np.ndarray] = {"dims": np.asarray(dims, dtype=np.int32), "tau": np.asarray([tau], dtype=np.float32)}
    for k, W in layers.items():
        save[k] = W.astype(np.int8)
    np.savez_compressed(path, **save)
    return path


def export_bin(name: str, outdir: Path, dims: List[int], layers: Dict[str, np.ndarray], tau: float) -> Tuple[Path, Path]:
    bin_path = outdir / f"{name}_ternary_packed.bin"
    manifest_path = outdir / f"{name}_manifest.json"
    offset = 0
    entries = []
    with bin_path.open("wb") as fh:
        for idx in range(len(dims) - 1):
            key = f"W{idx}"
            tern = layers[key]
            codes = _codes_from_int8(tern)
            blob = _pack_2bit(codes)
            fh.write(blob)
            entries.append(
                {
                    "name": key,
                    "shape": list(map(int, tern.shape)),
                    "offset": int(offset),
                    "nbytes": int(len(blob)),
                }
            )
            offset += len(blob)

    manifest = {
        "name": name,
        "dims": dims,
        "tau": tau,
        "mapping": {"0": 0, "1": +1, "2": -1, "3": None},
        "layout": "row-major",
        "packed": {"bits": 2, "per_byte": 4},
        "layers": entries,
        "binary": bin_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return bin_path, manifest_path


def export_c_header(name: str, outdir: Path, dims: List[int], layers: Dict[str, np.ndarray], tau: float) -> Path:
    safe = name.upper().replace("-", "_").replace(" ", "_")
    path = outdir / f"{name}_ternary.h"
    with path.open("w", encoding="utf-8") as f:
        guard = f"FEEDFLIPNETS_{safe}_TERNARY_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* dims: {dims}, tau: {tau} */\n")
        f.write(f"static const int32_t ffn_dims[] = {{ {', '.join(map(str, dims))} }};\n")
        f.write(f"static const float ffn_tau = {tau:.8f}f;\n\n")
        for idx in range(len(dims) - 1):
            key = f"W{idx}"
            tern = layers[key].astype(np.int8)
            flat = tern.ravel()
            values = ", ".join(str(int(v)) for v in flat)
            f.write(
                f"static const int8_t ffn_{key}[] = {{ {values} }}; /* shape=({tern.shape[0]},{tern.shape[1]}) */\n"
            )
        f.write("\n#endif /* end of header */\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Export ternary weights for deployment")
    ap.add_argument("--ckpt", required=True, help="Path to Trainer checkpoint (.ckpt)")
    ap.add_argument("--tau", type=float, default=0.05, help="Ternary threshold for quantization")
    ap.add_argument("--out", type=Path, default=Path("artifacts/export"), help="Output directory")
    ap.add_argument("--format", choices=["all", "npz", "bin", "c"], default="all")
    ap.add_argument("--name", default=None, help="Optional label for outputs (default derived from ckpt)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model_name = args.name or ckpt.parent.name or "model"

    # Load checkpoint arrays (float), infer dims, quantize
    with np.load(ckpt, allow_pickle=False) as data:
        keys = list(data.keys())
        dims = _infer_dims_from_ckpt(keys, data)
        layers: Dict[str, np.ndarray] = {}
        for i in range(len(dims) - 1):
            k = f"W{i}"
            layers[k] = _quantize_ternary_det(np.asarray(data[k], dtype=np.float32), float(args.tau))

    made: List[Path] = []
    if args.format in ("all", "npz"):
        made.append(export_npz(model_name, outdir, dims, layers, float(args.tau)))
    if args.format in ("all", "bin"):
        bp, mp = export_bin(model_name, outdir, dims, layers, float(args.tau))
        made.extend([bp, mp])
    if args.format in ("all", "c"):
        made.append(export_c_header(model_name, outdir, dims, layers, float(args.tau)))

    print("Exported:")
    for p in made:
        print(f" - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

