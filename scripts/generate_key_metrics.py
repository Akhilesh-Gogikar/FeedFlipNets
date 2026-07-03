#!/usr/bin/env python3
"""Emit the Metrics Key box for the paper.

The previous version summarised headline accuracies. Review feedback asked for a
definition-first cheat sheet that documents how we compute the stability
diagnostics (E@90%-of-best, early slope, stability CV, alignment AUC, TOST).

This script therefore writes a compact LaTeX tabular with metric names,
definitions, and implementation notes. It does not depend on aggregated CSVs so
that the table stays in sync with the analytical choices baked into the text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "docs/paper/_generated"
OUT_TABLE = OUTDIR / "key_metrics_table.tex"


@dataclass
class MetricRow:
    name: str
    definition: str
    note: str


METRICS: tuple[MetricRow, ...] = (
    MetricRow(
        name=r"E@90\% of best",
        definition=(
            r"First epoch where a run's validation metric reaches " r"\(0.9\times\) its best value."
        ),
        note=r"Computed per seed; we report mean $\pm$ 95\% t-CI over matched seeds.",
    ),
    MetricRow(
        name="Early slope (e1→e3)",
        definition="Finite difference between epochs 1 and 3 of the validation metric.",
        note="Captures initial learning speed; positive is faster.",
    ),
    MetricRow(
        name="Stability CV",
        definition="Coefficient of variation of validation loss across epochs in a run.",
        note="Lower is steadier; we average CV across matched seeds.",
    ),
    MetricRow(
        name="Alignment AUC",
        definition=r"Normalized area under the layerwise cosine-alignment curves \(\rho_\ell(t)\).",
        note=r"Reported as mean $\pm$ 95\% t-CI; higher implies backprop-like signals.",
    ),
    MetricRow(
        name="TOST equivalence",
        definition=(
            r"Two one-sided tests on matched seeds with \(\delta=\pm0.5\) pp for accuracy "
            r"(\(\pm0.01\) for $R^2\))."
        ),
        note="We declare equivalence when both one-sided p-values < 0.05.",
    ),
)


def build_table() -> str:
    rows = "\n".join(f"{row.name} & {row.definition} & {row.note} \\\\" for row in METRICS)
    body = dedent(
        f"""
        \\begin{{table}}[H]
        \\centering
        \\footnotesize
        \\begin{{tabularx}}{{\\linewidth}}{{l X X}}
        \\toprule
        Metric & Definition & Notes \\\\
        \\midrule
        {rows}
        \\bottomrule
        \\end{{tabularx}}
        \\caption{{Metrics key used in Sec.~7--9.}}
        \\label{{tab:metrics-key}}
        \\end{{table}}
        """
    ).strip()
    return body


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text(build_table() + "\n", encoding="utf-8")
    print(f"Wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
