#!/usr/bin/env python3
"""
Generate the 'Key Metrics' rows for docs/paper/main.tex from aggregated CSVs.

Outputs LaTeX rows to docs/paper/_generated/key_metrics_rows.tex

Sources consulted:
- data/report/benchmark_summary.csv (primary)

Formatting rules:
- MNIST, AG News values shown as percentages with two decimals.
- 20NG values shown as decimals with three places (mean ± std).
- UCR shown as percentage (best ternary-DFA real run).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/report/benchmark_summary.csv"
OUTDIR = ROOT / "docs/paper/_generated"
OUT_ROWS = OUTDIR / "key_metrics_rows.tex"
OUT_TABLE = OUTDIR / "key_metrics_table.tex"


@dataclass
class Stat:
    mean: float
    std: float
    count: int
    variant: str


def _best_accuracy(
    rows: list[dict],
    dataset: str,
    mode: str,
    strategy: Optional[str] = None,
    variant_prefix: Optional[str] = None,
    flip: Optional[str] = None,
) -> Optional[Stat]:
    best: Optional[Stat] = None
    for r in rows:
        try:
            if r["dataset"] != dataset or r["mode"] != mode or r["metric"] != "accuracy":
                continue
            if strategy and r["strategy"] != strategy:
                continue
            if variant_prefix and not r["strategy_variant"].startswith(variant_prefix):
                continue
            if flip and r["flip"] != flip:
                continue
            mean = float(r["mean"])
            std = float(r.get("std", 0.0) or 0.0)
            cnt = int(float(r.get("count", 0) or 0))
            if (best is None) or (mean > best.mean):
                best = Stat(mean=mean, std=std, count=cnt, variant=r["strategy_variant"])
        except Exception:
            # Skip malformed rows
            continue
    return best


def pct(x: float) -> str:
    return f"{x*100:.2f}\\%"


def dec3(x: float) -> str:
    s = f"{x:.3f}"
    # Match table’s compact style like .321
    if s.startswith("0"):
        s = s[1:]
    return s


def main() -> int:
    if not CSV.exists():
        raise SystemExit(f"Missing CSV: {CSV}")

    with CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # MNIST (real): compare best BP float vs best DFA float (flip off)
    mnist_bp = _best_accuracy(
        rows, "mnist", "real", strategy="backprop", variant_prefix="backprop_float", flip="off"
    )
    mnist_dfa = _best_accuracy(
        rows, "mnist", "real", strategy="dfa", variant_prefix="dfa_float", flip="off"
    )

    # 20NG (real): best BP float vs best DFA float (flip off)
    ng_bp = _best_accuracy(
        rows,
        "20newsgroups",
        "real",
        strategy="backprop",
        variant_prefix="backprop_float",
        flip="off",
    )
    ng_dfa = _best_accuracy(
        rows, "20newsgroups", "real", strategy="dfa", variant_prefix="dfa_float", flip="off"
    )

    # AG News (real): compare BP float vs DFA float (flip off)
    ag_bp = _best_accuracy(
        rows, "ag_news", "real", strategy="backprop", variant_prefix="backprop_float", flip="off"
    )
    ag_dfa = _best_accuracy(
        rows, "ag_news", "real", strategy="dfa", variant_prefix="dfa_float", flip="off"
    )

    # UCR (real): best ternary DFA
    ucr_tern = _best_accuracy(rows, "ucr", "real", strategy="ternary_dfa")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    if mnist_bp and mnist_dfa:
        left = f"{pct(mnist_bp.mean)}"
        if mnist_bp.count and mnist_bp.count > 1:
            left = f"{pct(mnist_bp.mean)} $\\pm$ {pct(mnist_bp.std)}"
        right = f"{pct(mnist_dfa.mean)}"
        if mnist_dfa.count and mnist_dfa.count > 1:
            right = f"{pct(mnist_dfa.mean)} $\\pm$ {pct(mnist_dfa.std)}"
        lines.append(f"Vision (MNIST, real) & Test Acc. (BP vs DFA float) & {left} vs {right} \\\\")
    # 20NG
    if ng_bp and ng_dfa:
        left = f"{dec3(ng_bp.mean)}"
        if ng_bp.count and ng_bp.count > 1:
            left = f"{dec3(ng_bp.mean)} $\\pm$ {dec3(ng_bp.std)}"
        right = f"{dec3(ng_dfa.mean)}"
        if ng_dfa.count and ng_dfa.count > 1:
            right = f"{dec3(ng_dfa.mean)} $\\pm$ {dec3(ng_dfa.std)}"
        lines.append(f"Text (20NG, real) & Test Acc. (BP vs DFA float) & {left} vs {right} \\\\")
    # AG News
    if ag_bp and ag_dfa:
        left = f"{pct(ag_bp.mean)}"
        if ag_bp.count and ag_bp.count > 1:
            left = f"{pct(ag_bp.mean)} $\\pm$ {pct(ag_bp.std)}"
        right = f"{pct(ag_dfa.mean)}"
        if ag_dfa.count and ag_dfa.count > 1:
            right = f"{pct(ag_dfa.mean)} $\\pm$ {pct(ag_dfa.std)}"
        lines.append(f"AG News (real) & Test Acc. (BP vs DFA float) & {left} vs {right} \\\\")
    # UCR
    if ucr_tern:
        lines.append(
            f"Time-series (UCR) & Best Test Acc. (Ternary DFA, real) & {pct(ucr_tern.mean)} \\\\"
        )

    # Static rows
    lines.append(r"Sparsity (Ternary) & Zero ratio (typical) & \(\approx\) 0.5--0.75 \\")
    lines.append(r"Determinism & Seeds, fixtures & Yes \\")

    # rows file
    OUT_ROWS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # full table file
    table = [
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\begingroup\\setlength{\\tabcolsep}{3pt}",
        # Wrap the value column using tabularx's X-type (Y alias defined in main.tex)
        "\\begin{tabularx}{\\linewidth}{@{} l l Y @{}}",
        "\\toprule",
        "Headline & Metric & Value \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabularx}\\par\\endgroup",
        "\\caption*{Key Metrics (condensed). Representative values from our benchmark suite.}",
        "\\end{table}",
        "",
    ]
    OUT_TABLE.write_text("\n".join(table), encoding="utf-8")
    print(f"Wrote {OUT_ROWS.relative_to(ROOT)} and {OUT_TABLE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
