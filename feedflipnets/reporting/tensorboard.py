"""TensorBoard adapter for scalar and alignment calibration logging.

This module is optional. If TensorBoard is not available, the adapter
gracefully degrades to a no-op while still saving calibration figures
as PNGs under the run directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _make_writer(log_dir: Path):
    """Try to construct a TensorBoard writer, else return None.

    Prefers PyTorch's SummaryWriter if available, then tensorboardX.
    """
    try:  # PyTorch TensorBoard
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        try:  # tensorboardX
            from tensorboardX import SummaryWriter  # type: ignore

            return SummaryWriter(log_dir=str(log_dir))
        except Exception:
            return None


class TensorBoardAdapter:
    """Write metrics and alignment calibration to TensorBoard.

    - Scalars: logs every float metric under ``{split}/{name}``.
    - Alignment calibration: plots binned p(⟨ΔV_DFA, ΔV_BP⟩>0) vs ρ bins.
    - Also saves a PNG figure into ``run_dir`` for permanence.
    """

    def __init__(self, run_dir: str | Path, *, split: str = "train") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        # Write TB logs next to run_dir under a "tb" subfolder to avoid clutter
        self.tb_dir = self.run_dir / "tb"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self._writer = _make_writer(self.tb_dir)

    # ----- Standard logging hooks -----
    def on_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        if self._writer is None:
            return
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                tag = f"{self.split}/{name}"
                try:
                    self._writer.add_scalar(tag, float(value), global_step=epoch)
                except Exception:
                    # Best-effort: skip metrics that TB can't serialize
                    pass

    # Alias for sinks compatibility
    __call__ = on_epoch

    # ----- Alignment calibration hook -----
    def on_alignment(self, epoch: int, alignment: Mapping[str, object]) -> None:
        # Expected structure from Trainer: {"per_layer": {"rho": [[...], ...], "p": [[...], ...]}}
        try:
            per_layer = alignment.get("per_layer", {})  # type: ignore[assignment]
            rho_layers: Sequence[Sequence[float]] = per_layer.get("rho", [])  # type: ignore[assignment]
            p_layers: Sequence[Sequence[float]] = per_layer.get("p", [])  # type: ignore[assignment]
        except Exception:
            return
        # Flatten across layers
        all_rhos = np.array([r for layer in rho_layers for r in layer], dtype=np.float32)
        all_ps = np.array([p for layer in p_layers for p in layer], dtype=np.float32)
        if all_rhos.size == 0 or all_ps.size == 0:
            return

        # Bin rhos into fixed-width bins and compute mean p in each bin
        bins = np.linspace(-1.0, 1.0, 21)
        inds = np.digitize(all_rhos, bins, right=True)
        p_means = []
        centers = []
        for b in range(1, len(bins)):
            mask = inds == b
            if np.any(mask):
                p_means.append(float(np.mean(all_ps[mask])))
            else:
                p_means.append(np.nan)
            centers.append(float((bins[b - 1] + bins[b]) * 0.5))

        # Save a calibration figure and log to TensorBoard if available
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(centers, p_means, marker="o", linestyle="-", label="empirical p")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("rho")
            ax.set_ylabel("p = Pr[dot>0]")
            ax.set_title("Alignment calibration: p vs rho")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            # Persist PNG to the run directory
            out_path = self.run_dir / f"calibration_p_vs_rho_epoch{int(epoch)}.png"
            fig.savefig(out_path, bbox_inches="tight")
            if self._writer is not None:
                try:
                    self._writer.add_figure(f"{self.split}/p_vs_rho", fig, global_step=epoch)
                except Exception:
                    pass
            plt.close(fig)
        except Exception:
            # If plotting fails, silently skip to avoid breaking training
            return

