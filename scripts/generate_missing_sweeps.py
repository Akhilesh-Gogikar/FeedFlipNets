#!/usr/bin/env python3
"""
Create placeholder sweep plots for datasets/modes where sweep data is missing,
so LaTeX figures have complete panels and consistent layout.

Outputs placed under data/report/plots/:
 - lr_sweep_california_housing_real.png
 - lr_sweep_ucr_gunpoint_real.png
 - tau_sweep_ucr_gunpoint_real.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path("data/report/plots").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _placeholder(path: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.axis("off")
    ax.text(
        0.5,
        0.6,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.4,
        "No sweep data available",
        ha="center",
        va="center",
        fontsize=10,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    _placeholder(
        OUT_DIR / "lr_sweep_california_housing_real.png",
        "LR sweep — california_housing (real)",
        "Learning rate",
        "Primary",
    )
    _placeholder(
        OUT_DIR / "lr_sweep_ucr_gunpoint_real.png",
        "LR sweep — ucr_gunpoint (real)",
        "Learning rate",
        "Primary",
    )
    _placeholder(
        OUT_DIR / "tau_sweep_ucr_gunpoint_real.png",
        "Tau sweep — ucr_gunpoint (real)",
        "tau (threshold)",
        "Primary",
    )


if __name__ == "__main__":
    main()
