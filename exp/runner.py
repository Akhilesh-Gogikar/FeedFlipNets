#!/usr/bin/env python3
"""Typer CLI wrapper for benchmark orchestration and reporting.

This module delegates to the existing scripts:
  - scripts/run_benchmark.py
  - scripts/compile_benchmark_report.py

Usage:
  python -m exp.runner run --subset MWR
  python -m exp.runner report --out runs/report.md
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Optional

try:
    import typer
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "Typer is required for exp.runner. Install with `pip install typer`."
    ) from exc


app = typer.Typer(add_completion=False)


def _sh(args: List[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(args, check=True, cwd=str(cwd) if cwd else None)


def _load_plan(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load experiment plans") from exc
    return yaml.safe_load(path.read_text()) or {}


@app.command()
def run(
    plan: Path = typer.Option(Path("exp/experiments.yaml"), help="Path to experiments.yaml"),
    subset: List[str] = typer.Option(
        ["MWR"], help="Subset key(s) from experiments.yaml; can be repeated"
    ),
    data_root: Optional[Path] = typer.Option(
        None, help="Optional datasets root (unused by offline fixtures)"
    ),
) -> None:
    """Run one or more benchmark subsets defined in experiments.yaml."""

    cfg = _load_plan(plan)
    subsets = subset or ["MWR"]
    for name in subsets:
        entry = (cfg.get("subsets") or {}).get(name, {})
        if not entry:
            raise typer.BadParameter(f"Subset {name!r} not found in {plan}")

        datasets = [str(x) for x in entry.get("datasets") or []]
        modes = [str(x) for x in entry.get("modes") or []]
        variants = [str(x) for x in entry.get("variants") or []]
        seeds = [str(int(x)) for x in entry.get("seeds") or []]

        args: List[str] = [
            "python",
            "-m",
            "scripts.run_benchmark",
        ]
        if datasets:
            args.extend(["--datasets", *datasets])
        if modes:
            args.extend(["--modes", *modes])
        if variants:
            args.extend(["--variants", *variants])
        if seeds:
            args.extend(["--seeds", *seeds])

        typer.echo(
            "[exp.runner] Running subset "
            f"{name}: datasets={datasets} modes={modes} variants={variants} seeds={seeds}"
        )
        _sh(args)

    # Always compile a report after all subsets
    _sh(["python", "-m", "scripts.compile_benchmark_report"])


@app.command()
def report(
    runs_dir: Path = typer.Option(Path("runs/bench"), help="Runs root to scan"),
    out: Path = typer.Option(Path("runs/report.md"), help="Output Markdown report path"),
) -> None:
    """Compile the benchmark report and write a convenient copy to runs/report.md."""

    # Reuse the existing aggregator to refresh data/report/*
    _sh(["python", "-m", "scripts.compile_benchmark_report"])

    src = Path("data/report/benchmark_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        out.write_text(src.read_text())
    else:
        out.write_text("No report available. Run exp.runner run first.\n")


if __name__ == "__main__":  # pragma: no cover
    app()
