#!/usr/bin/env python3
"""Generate the CNN BP-vs-DFA gap table from GPU-campaign artifacts.

Reads the metrics/timing produced by ``scripts/baselines/cnn_dfa_baselines.py``
during the RunPod GPU campaign (see runs/gpu_campaign/) and emits a LaTeX
table plus a machine-readable provenance JSON. Faithful reporting: the DFA
run's training loss diverges even though accuracy rises, and this is recorded.

Usage:
  python scripts/gen_cnn_gap_table.py \
      --campaign-dir runs/gpu_campaign \
      --out docs/paper/_generated/cnn_gap_table.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _final(metrics: dict, split: str) -> dict:
    return metrics[split][-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", default="runs/gpu_campaign")
    ap.add_argument("--out", default="docs/paper/_generated/cnn_gap_table.tex")
    args = ap.parse_args()

    cdir = Path(args.campaign_dir)
    bp = json.loads((cdir / "cnn_bp_full_metrics.json").read_text())
    dfa = json.loads((cdir / "cnn_dfa_full_metrics.json").read_text())
    timing = [
        json.loads(row)
        for row in (cdir / "cnn_timing.jsonl").read_text().splitlines()
        if row.strip()
    ]
    tmap = {t["tag"]: t for t in timing}

    bp_test = _final(bp, "test")["acc"] * 100
    dfa_test = _final(dfa, "test")["acc"] * 100
    bp_train = _final(bp, "train")["acc"] * 100
    dfa_train = _final(dfa, "train")["acc"] * 100
    dfa_test_loss = _final(dfa, "test")["loss"]
    bp_wall = tmap["gpu_bp_full"]["wall_s"]
    dfa_wall = tmap["gpu_dfa_full"]["wall_s"]
    gpu_t = tmap["time_gpu"]["wall_s"]
    cpu_t = tmap["time_cpu"]["wall_s"]

    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Method & Train acc (\%) & Test acc (\%) & GPU wall (s) \\",
        r"\hline",
        rf"BP (float) & {bp_train:.2f} & {bp_test:.2f} & {bp_wall:.1f} \\",
        rf"DFA (float) & {dfa_train:.2f} & {dfa_test:.2f} & {dfa_wall:.1f} \\",
        r"\hline",
        rf"$\Delta$ (DFA$-$BP) & {dfa_train - bp_train:+.2f} & "
        rf"{dfa_test - bp_test:+.2f} & {dfa_wall - bp_wall:+.1f} \\",
        r"\hline",
        r"\end{tabular}",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    prov = {
        "source": "RunPod GPU campaign (device-fixed DFA), FashionMNIST CNN, 8 epochs, batch 128, lr 0.2, full data",
        "bp_test_acc_pct": round(bp_test, 2),
        "dfa_test_acc_pct": round(dfa_test, 2),
        "gap_test_pct": round(dfa_test - bp_test, 2),
        "bp_gpu_wall_s": bp_wall,
        "dfa_gpu_wall_s": dfa_wall,
        "bp_gpu_vs_cpu_speedup": round(cpu_t / gpu_t, 2),
        "dfa_test_loss_final": dfa_test_loss,
        "dfa_loss_diverges": dfa_test_loss > 1e3,
        "timing": timing,
    }
    prov_path = out.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"wrote {prov_path}")
    print(f"BP {bp_test:.2f}%  DFA {dfa_test:.2f}%  gap {dfa_test-bp_test:+.2f} pts")
    print(f"GPU vs CPU speedup (same BP workload): {cpu_t/gpu_t:.2f}x")
    print(f"DFA final test loss: {dfa_test_loss:.3e} (diverges: {dfa_test_loss>1e3})")


if __name__ == "__main__":
    main()
