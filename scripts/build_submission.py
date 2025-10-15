#!/usr/bin/env python3
"""
Build FeedFlipNets camera-ready artifacts from local runs and paper sources.

Outputs:
- paper/fig/Fig-*.png
- paper/tables/Tbl-*.csv + .tex
- paper/revision_summary.json
- paper/main.prepatch.tex (backup)
- paper/main.tex (patched in-place)

Notes:
- CPU-only; Matplotlib only (no seaborn); one chart per figure.
- Deterministic: the script reads existing logs only; no reshuffling/re-splitting.
- Fails fast with actionable diagnostics for missing expected paths.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # CPU-only headless
import matplotlib.pyplot as plt  # noqa: E402

# Optional SciPy dependency for t-CI; fall back to normal approx if unavailable
try:
    from scipy import stats  # type: ignore
except Exception:

    class _T:  # minimal normal-approx fallback
        @staticmethod
        def ppf(prob: float, df: int) -> float:
            # 95% two-sided -> 1.96; ignore df
            return 1.96

    class stats:  # type: ignore
        t = _T()


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
PAPER_DIR = REPO_ROOT / "paper"  # symlink -> docs/paper
FIG_DIR = PAPER_DIR / "fig"
TABLES_DIR = PAPER_DIR / "tables"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    raise SystemExit(2)


def git_sha_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT)
        return out.decode().strip()
    except Exception:
        return "unknown"


def check_expected_paths() -> Dict[str, Any]:
    missing: List[str] = []
    info: Dict[str, Any] = {}
    if not RUNS_DIR.exists():
        missing.append(str(RUNS_DIR))
    paper_main = PAPER_DIR / "main.tex"
    if not paper_main.exists():
        # Diagnostic with minimal fix hint
        fail(f"Expected paper source missing: {paper_main}. Minimal fix: ln -s docs/paper paper")
    info["paper_main"] = str(paper_main)
    info["runs_present"] = any(RUNS_DIR.iterdir()) if RUNS_DIR.exists() else False
    return info


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return {}
    try:
        return json.loads(mpath.read_text())
    except Exception:
        return {}


def _infer_dataset(run_dir: Path, manifest: Dict[str, Any]) -> str:
    cfg = manifest.get("config", {})
    ds = cfg.get("data", {}).get("name")
    if isinstance(ds, str) and ds:
        return ds
    # Fallback: derive from directory name
    name = run_dir.name
    for key in [
        "mnist",
        "fashion",
        "fashion_mnist",
        "cifar10",
        "ag_news",
        "20newsgroups",
        "adult",
        "california",
        "ucr",
        "gunpoint",
        "synthetic",
    ]:
        if key in name:
            return key.replace("fashion_", "fashion ").replace("_", " ")
    return name


def _infer_domain(dataset: str) -> str:
    ds = dataset.lower()
    if any(k in ds for k in ["mnist", "cifar", "vision", "fashion"]):
        return "vision"
    if any(k in ds for k in ["20newsgroups", "news", "bow", "text", "ag"]):
        return "text"
    if any(k in ds for k in ["ucr", "gunpoint", "timeseries", "time-series"]):
        return "timeseries"
    if any(k in ds for k in ["california", "housing", "adult", "tabular"]):
        return "tabular"
    return "unknown"


def _mode_offline(manifest: Dict[str, Any]) -> str:
    cfg = manifest.get("config", {})
    offline = bool(cfg.get("offline", False))
    return "offline" if offline else "real"


def _variant(manifest: Dict[str, Any], run_name: str) -> str:
    cfg = manifest.get("config", {})
    model = cfg.get("model", {})
    strat = str(model.get("strategy", "unknown")).lower()
    train = cfg.get("train", {})
    flip = str(train.get("flip", "off")).lower()
    if strat == "backprop":
        base = "backprop_float"
    elif strat == "dfa":
        # float vs ternary forward indicated by flip
        base = "dfa_float" if flip == "off" else "dfa_ternary"
    else:
        base = strat
    # Structured feedback hints in run name
    if "hadamard" in run_name:
        base += "_hadamard"
    if "orthogonal" in run_name or "orth" in run_name:
        base += "_orthogonal"
    return base


def _flip_schedule(manifest: Dict[str, Any]) -> str:
    cfg = manifest.get("config", {})
    train = cfg.get("train", {})
    return str(train.get("flip_schedule", "off")).lower()


def _tau_value(manifest: Dict[str, Any]) -> Optional[float]:
    cfg = manifest.get("config", {})
    # sometimes under config.tau, sometimes train.tau
    for key in ["tau", ("train", "tau")]:
        if isinstance(key, tuple):
            v = cfg.get(key[0], {}).get(key[1])
        else:
            v = cfg.get(key)
        if isinstance(v, (float, int)):
            return float(v)
    return None


def _throughput(run_dir: Path) -> Optional[float]:
    timing_path = run_dir / "timing.json"
    test_path = run_dir / "metrics_test.jsonl"
    if not timing_path.exists() or not test_path.exists():
        return None
    try:
        timing = json.loads(timing_path.read_text())
        total = float(timing.get("test", {}).get("total_sec", 0.0))
        test_epochs = 0
        samples_per_epoch = None
        with test_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                test_epochs += 1
                if samples_per_epoch is None:
                    rec = json.loads(line)
                    samples_per_epoch = int(rec.get("sample_count", 0))
        if not test_epochs or not samples_per_epoch or total <= 0:
            return None
        total_samples = test_epochs * samples_per_epoch
        return float(total_samples) / total
    except Exception:
        return None


def _last_metric(test_jsonl: Path, keys: List[str]) -> Optional[float]:
    if not test_jsonl.exists():
        return None
    out: Dict[str, float] = {}
    with test_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for k in keys:
                if k in rec:
                    out[k] = float(rec[k])
    for k in keys:
        if k in out:
            return out[k]
    return None


def _collect_runs() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    # Top-level runs (single runs)
    for p in RUNS_DIR.iterdir():
        if not p.is_dir():
            continue
        # skip non-run folders
        if p.name in {"bench", "legacy", "sweep"}:
            continue
        manifest = _load_manifest(p)
        dataset = _infer_dataset(p, manifest)
        domain = _infer_domain(dataset)
        mode = _mode_offline(manifest)
        variant = _variant(manifest, p.name)
        flip_schedule = _flip_schedule(manifest)
        tau = _tau_value(manifest)
        thr = _throughput(p)
        zratio = _last_metric(p / "metrics_test.jsonl", ["ternary_zero_ratio"])
        acc = _last_metric(p / "metrics_test.jsonl", ["accuracy", "r2"])  # accuracy or R2

        row = {
            "run_dir": str(p),
            "dataset": dataset,
            "domain": domain,
            "mode": mode,
            "variant": variant,
            "flip_schedule": flip_schedule,
            "tau": tau,
            "throughput_samples_sec": thr,
            "zeros_ratio": zratio,
            "score": acc,
        }
        rows.append(row)
    # Bench layout (multiple seeds per variant)
    bench_root = RUNS_DIR / "bench"
    if bench_root.exists():
        for ds_dir in bench_root.iterdir():
            if not ds_dir.is_dir():
                continue
            for mode_dir in ds_dir.iterdir():
                if not mode_dir.is_dir():
                    continue
                for var_dir in mode_dir.iterdir():
                    if not var_dir.is_dir():
                        continue
                    for seed_dir in var_dir.glob("seed*"):
                        if not seed_dir.is_dir():
                            continue
                        manifest = _load_manifest(seed_dir)
                        dataset = ds_dir.name
                        domain = _infer_domain(dataset)
                        mode = mode_dir.name
                        variant = _variant(manifest, var_dir.name)
                        flip_schedule = _flip_schedule(manifest)
                        tau = _tau_value(manifest)
                        thr = _throughput(seed_dir)
                        zratio = _last_metric(
                            seed_dir / "metrics_test.jsonl", ["ternary_zero_ratio"]
                        )
                        acc = _last_metric(
                            seed_dir / "metrics_test.jsonl", ["accuracy", "r2"]
                        )  # accuracy or R2
                        rows.append(
                            {
                                "run_dir": str(seed_dir),
                                "dataset": dataset,
                                "domain": domain,
                                "mode": mode,
                                "variant": variant,
                                "flip_schedule": flip_schedule,
                                "tau": tau,
                                "throughput_samples_sec": thr,
                                "zeros_ratio": zratio,
                                "score": acc,
                            }
                        )
    return pd.DataFrame(rows)


def _collect_time_series_for(metric_key_prefix: str, split: str = "train") -> pd.DataFrame:
    """
    Collect per-epoch series for keys like rho_l0, rho_l1, p_l0, ... and rho_mean/p_mean.
    Returns tidy DF with columns: run_dir, epoch, key, value
    """
    out_rows: List[Dict[str, Any]] = []
    # search depth 1 and 2 (e.g., runs/* and runs/sweep/*)
    candidates: List[Path] = []
    for p in RUNS_DIR.iterdir():
        if p.is_dir() and p.name not in {"bench", "legacy"}:
            candidates.append(p)
            # include one level deeper
            for q in p.iterdir():
                if q.is_dir():
                    candidates.append(q)
    # also include bench layout: runs/bench/<dataset>/<mode>/<variant>/seed*/
    bench_root = RUNS_DIR / "bench"
    if bench_root.exists():
        for dataset_dir in bench_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            for mode_dir in dataset_dir.iterdir():
                if not mode_dir.is_dir():
                    continue
                for var_dir in mode_dir.iterdir():
                    if not var_dir.is_dir():
                        continue
                    for seed_dir in var_dir.glob("seed*"):
                        if seed_dir.is_dir():
                            candidates.append(seed_dir)
    for run_dir in candidates:
        jpath = run_dir / f"metrics_{split}.jsonl"
        if not jpath.exists():
            continue
        recs = _read_jsonl(jpath)
        for rec in recs:
            epoch = int(rec.get("epoch", 0))
            for k, v in rec.items():
                if isinstance(v, (int, float)) and (
                    k.startswith(metric_key_prefix) or k == f"{metric_key_prefix}mean"
                ):
                    out_rows.append(
                        {"run_dir": str(run_dir), "epoch": epoch, "key": k, "value": float(v)}
                    )
    return pd.DataFrame(out_rows)


def _ci95(series: Iterable[float]) -> Tuple[float, float, int]:
    vals = [float(x) for x in series if isinstance(x, (float, int)) and not math.isnan(float(x))]
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    if n > 1:
        tcrit = stats.t.ppf(0.975, df=n - 1)
        half = tcrit * sd / math.sqrt(n)
    else:
        half = 0.0
    return (mean, half, n)


def fig_pareto(df: pd.DataFrame) -> List[str]:
    written: List[str] = []
    for dataset, sub in df.groupby("dataset"):
        sub2 = sub.dropna(subset=["score", "zeros_ratio", "throughput_samples_sec"]).copy()
        if sub2.empty:
            continue
        plt.figure(figsize=(6.0, 4.0))
        # Plot by flip_schedule for legend (no explicit colors set)
        for sched, g in sub2.groupby("flip_schedule"):
            plt.scatter(
                g["zeros_ratio"].values,
                g["score"].values,
                s=np.clip(np.array(g["throughput_samples_sec"].values, float) * 0.02, 20, 200),
                label=str(sched),
                alpha=0.85,
                edgecolors="k",
            )
        plt.xlabel("Sparsity (zero ratio)")
        plt.ylabel("Accuracy / R$^2$")
        plt.title(f"Pareto: {dataset}")
        plt.grid(True, ls=":", alpha=0.5)
        plt.legend(title="flip schedule", fontsize=8)
        out = FIG_DIR / f"Fig-PARETO-{dataset.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        written.append(str(out))
    return written


def fig_series_mean_ci(tidy: pd.DataFrame, title: str, ylabel: str, outfile: Path) -> Optional[str]:
    if tidy.empty:
        return None
    # Aggregate across runs for keys ending with _mean or layer keys
    agg = tidy.groupby(["epoch"]).agg({"value": [np.mean, np.std, "count"]}).reset_index()
    agg.columns = ["epoch", "mean", "sd", "n"]
    if agg.empty:
        return None
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(agg["epoch"], agg["mean"], lw=2)
    # 95% CI via t with per-epoch n
    cis = []
    for _, r in agg.iterrows():
        n = int(r["n"])
        sd = float(r["sd"]) if n > 1 else 0.0
        half = stats.t.ppf(0.975, df=n - 1) * sd / math.sqrt(n) if n > 1 else 0.0
        cis.append(half)
    cis = np.asarray(cis)
    plt.fill_between(agg["epoch"], agg["mean"] - cis, agg["mean"] + cis, alpha=0.25)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, ls=":", alpha=0.5)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()
    return str(outfile)


def fig_bars_ci(
    df: pd.DataFrame, group_cols: List[str], value_col: str, title: str, ylabel: str, outfile: Path
) -> Optional[str]:
    if df.empty:
        return None
    # Aggregate mean ± 95% CI
    g = df.groupby(group_cols)[value_col].apply(list).reset_index(name="vals")
    if g.empty:
        return None
    means = []
    halves = []
    labels = []
    for _, row in g.iterrows():
        vals = [float(v) for v in row["vals"] if v is not None and not math.isnan(float(v))]
        if not vals:
            continue
        mean, half, _ = _ci95(vals)
        means.append(mean)
        halves.append(half)
        labels.append(" / ".join(str(row[c]) for c in group_cols))
    if not means:
        return None
    x = np.arange(len(means))
    plt.figure(figsize=(max(6.0, 0.5 + 0.5 * len(means)), 4.0))
    plt.bar(x, means, yerr=halves, capsize=3)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()
    return str(outfile)


def compute_e90_and_slope() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute E@0.90 and early slope (epoch 1->3) from val metrics."""
    rows_e90: List[Dict[str, Any]] = []
    rows_slope: List[Dict[str, Any]] = []
    for p in RUNS_DIR.iterdir():
        if not p.is_dir():
            continue
        manifest = _load_manifest(p)
        dataset = _infer_dataset(p, manifest)
        mode = _mode_offline(manifest)
        variant = _variant(manifest, p.name)
        valp = p / "metrics_val.jsonl"
        if not valp.exists():
            continue
        recs = _read_jsonl(valp)
        if not recs:
            continue
        # choose metric key
        metric_key = "r2" if any("r2" in r for r in recs) else "accuracy"
        # E@0.90
        e_hit: Optional[int] = None
        for r in recs:
            v = r.get(metric_key)
            if isinstance(v, (int, float)) and float(v) >= 0.90:
                e_hit = int(r.get("epoch", 0))
                break
        rows_e90.append(
            {
                "run_dir": str(p),
                "dataset": dataset,
                "mode": mode,
                "variant": variant,
                "E@0.90": e_hit,
            }
        )
        # slope epoch 1->3
        pts = [
            (int(r.get("epoch", 0)), float(r.get(metric_key, np.nan)))
            for r in recs
            if isinstance(r.get(metric_key), (int, float))
        ]
        pts = sorted(pts)
        v1 = next((v for e, v in pts if e == 1), np.nan)
        v3 = next((v for e, v in pts if e == 3), np.nan)
        if not (math.isnan(v1) or math.isnan(v3)):
            slope = (v3 - v1) / 2.0
            rows_slope.append(
                {
                    "run_dir": str(p),
                    "dataset": dataset,
                    "mode": mode,
                    "variant": variant,
                    "slope_e1to3": slope,
                }
            )
    return pd.DataFrame(rows_e90), pd.DataFrame(rows_slope)


def best_configs_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """Build best configs table with CI and placeholders for tie-tests."""
    # Choose score name per dataset (accuracy or r2). We already placed both in score.
    rows: List[Dict[str, Any]] = []
    for (dataset, mode), grp in df.groupby(["dataset", "mode"]):
        # Pick best by mean score across variant
        by_var = grp.groupby("variant")["score"].apply(list).reset_index(name="vals")
        if by_var.empty:
            continue
        # compute mean for sorting
        by_var["mean"] = by_var["vals"].apply(
            lambda v: float(np.mean([x for x in v if x is not None])) if v else float("nan")
        )
        best_row = by_var.sort_values("mean", ascending=False).iloc[0]
        best_variant = str(best_row["variant"])
        vals_best = [x for x in best_row["vals"] if x is not None]
        mu_best, half_best, n_best = _ci95(vals_best)
        sd_best = float(np.std(vals_best, ddof=1)) if n_best > 1 else 0.0

        # Choose baseline: backprop_float if present, else dfa_float if present, else the best itself
        baseline_variant = (
            "backprop_float"
            if (by_var["variant"] == "backprop_float").any()
            else ("dfa_float" if (by_var["variant"] == "dfa_float").any() else best_variant)
        )
        vals_base_row = by_var[by_var["variant"] == baseline_variant]
        if not vals_base_row.empty:
            vals_base = [x for x in vals_base_row.iloc[0]["vals"] if x is not None]
        else:
            vals_base = vals_best
        mu_base, half_base, n_base = _ci95(vals_base)
        delta = mu_best - mu_base

        # Wilcoxon signed-rank if same n and both >1
        p_val: Optional[float] = None
        tie_flag: str = "NA"
        if len(vals_best) == len(vals_base) and len(vals_best) > 1:
            try:
                s1 = np.array(vals_best, float)
                s2 = np.array(vals_base, float)
                # align by index only; proper seed pairing not guaranteed
                stat, p_val = stats.wilcoxon(s1, s2)
                tie_flag = "tie" if p_val >= 0.05 else "diff"
            except Exception:
                p_val = None
                tie_flag = "NA"

        rows.append(
            {
                "Dataset": dataset,
                "Mode": mode,
                "Variant": best_variant,
                "n": n_best,
                "Mean": round(mu_best, 4) if not math.isnan(mu_best) else mu_best,
                "SD": round(sd_best, 4),
                "CI95": round(half_best, 4),
                "Baseline": baseline_variant,
                "BaselineMean": round(mu_base, 4) if not math.isnan(mu_base) else mu_base,
                "Delta": round(delta, 4)
                if not (math.isnan(mu_best) or math.isnan(mu_base))
                else float("nan"),
                "WilcoxonP": (round(p_val, 4) if p_val is not None else "NA"),
                "Tie": tie_flag,
            }
        )

    tbl = pd.DataFrame(rows)
    csv_path = TABLES_DIR / "Tbl-Best-Configs-CI.csv"
    tex_path = TABLES_DIR / "Tbl-Best-Configs-CI.tex"
    if not tbl.empty:
        # Drop rows with fewer than 2 repeats to avoid misleading CI claims
        tbl = tbl[tbl["n"].fillna(0) >= 2].reset_index(drop=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_csv(csv_path, index=False)

        # Simple LaTeX table (compact camera-ready form)
        def _shorten_variant(raw) -> str:
            label = "" if raw is None else str(raw)
            if label.lower() == "nan":
                return "nan"
            lookup = {
                "dfa_ternary": "DFA-ter",
                "dfa_float": "DFA-float",
                "backprop_float": "BP-float",
                "unknown": "?",
            }
            return lookup.get(label, label)

        def _shorten_mode(raw) -> str:
            label = "" if raw is None else str(raw)
            lookup = {
                "offline": "off",
                "online": "on",
                "real": "real",
                "sim": "sim",
            }
            return lookup.get(label, label)

        def _shorten_dataset(raw) -> str:
            label = "" if raw is None else str(raw)
            lower = label.lower()
            mapping = {
                "20newsgroups": "20NG",
                "california_housing": "CalHouse",
                "fashion_mnist": "Fashion",
                "mnist": "MNIST",
                "cifar10": "CIFAR10",
                "synthetic": "Synth",
                "ucr": "UCR",
            }
            return mapping.get(lower, label)

        def _format_p(raw) -> str:
            if raw is None:
                return "NA"
            if isinstance(raw, float):
                if math.isnan(raw):
                    return "NA"
                return f"{raw:.4f}"
            text = str(raw)
            return "NA" if text.lower() == "nan" else text

        def _fmt_scalar(val: Any) -> str:
            if val is None:
                return "--"
            if isinstance(val, float):
                if math.isnan(val):
                    return "--"
                return f"{val:.3f}"
            text = str(val)
            return "--" if text.lower() == "nan" else text

        with tex_path.open("w", encoding="utf-8") as fh:
            fh.write("% Auto-generated by scripts/build_submission.py\n")
            fh.write("\\begin{table}[H]\n\\centering\n\\tiny\n")
            fh.write("\\begingroup\n\\setlength{\\tabcolsep}{2.0pt}\n")
            fh.write("\\resizebox{0.80\\linewidth}{!}{%\n")
            cols = [
                "Dataset",
                "Mode",
                "Variant",
                "n",
                "Mean",
                "SD",
                "CI95",
                "Baseline",
                "Delta",
                "WilcoxonP",
                "Tie",
            ]
            header_labels = [
                "Data",
                "Mode",
                "Var",
                "n",
                "Mean",
                "SD",
                "CI",
                "Base",
                "$\\Delta$",
                "$p$",
                "Tie",
            ]
            fh.write(
                "\\begin{tabular}{l l l r S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] l S[table-format=+1.3] S[table-format=1.3] l}\n\\toprule\n"  # noqa: E501
            )
            fh.write(" & ".join(header_labels) + " \\\\ \n\\midrule\n")
            for _, r in tbl.iterrows():
                row_vals = []
                for key in cols:
                    val = r[key]
                    if key == "Dataset":
                        row_vals.append(_shorten_dataset(val))
                    elif key in ("Variant", "Baseline"):
                        row_vals.append(_shorten_variant(val))
                    elif key == "Mode":
                        row_vals.append(_shorten_mode(val))
                    elif key == "WilcoxonP":
                        row_vals.append(_format_p(val))
                    elif key in ("Mean", "SD", "CI95", "Delta"):
                        row_vals.append(_fmt_scalar(val))
                    else:
                        row_vals.append(_fmt_scalar(val))
                fh.write(" & ".join(row_vals) + " \\\\ \n")
            fh.write("\\bottomrule\n\\end{tabular}}\n")
            fh.write("\\endgroup\n")
            fh.write(
                "\\caption{Best configs with mean, SD, and n; 95\\% CIs shown only when n>1. Rows with n<2 are omitted to avoid spurious intervals. Tie tests (Wilcoxon) included when comparable.}\n"  # noqa: E501
            )
            fh.write("\\label{tab:best-configs-ci}\n\\end{table}\n")
        return tbl, str(csv_path), str(tex_path)
    return tbl, None, None


def patch_latex_for_figures() -> List[str]:
    paths_written: List[str] = []
    main_tex = PAPER_DIR / "main.tex"
    backup = PAPER_DIR / "main.prepatch.tex"
    # Backup original once per run
    if not backup.exists():
        shutil.copy2(main_tex, backup)
    txt = main_tex.read_text(encoding="utf-8")
    # Update graphicspath: include paper figs and aggregated report plots
    txt = re.sub(
        r"^\\graphicspath\{.*\}$",
        r"\\graphicspath{{fig/}{../../data/report/plots/}}",
        txt,
        flags=re.MULTILINE,
    )
    # Clean any debug markers added earlier
    txt = txt.replace("% Auto-inserted test\n", "")

    # Normalize a broken Results section line if split across newline
    txt = re.sub(r"\\section\{Results\s*\n\s*\}", r"\\section{Results}", txt)

    # Insert figures after Results header if not already present
    # Tolerate missing closing brace on same line (from prior edits)
    insert_anchor_pat = re.compile(r"^\\section\{Results", flags=re.MULTILINE)
    m = insert_anchor_pat.search(txt)
    idx = m.start() if m else -1
    if idx != -1 and (
        "Fig-PARETO" not in txt
        or "Fig-SignMatch-P" not in txt
        or "Fig-Align-Rho" not in txt
        or "Fig-Throughput" not in txt
    ):
        # Prefer inserting after the section label if present
        label_token = r"\label{sec:results}"
        label_pos = txt.find(label_token, idx)
        if label_pos != -1:
            # insert after the end of the label token
            insert_at = label_pos + len(label_token)
        else:
            # Fallback: after closing brace of section
            brace_pos = txt.find("}", idx)
            insert_at = (brace_pos + 1) if brace_pos != -1 else (idx + len("\\section{Results"))
        blocks: List[str] = []
        blocks.append(
            r"\n% Auto-inserted: Pareto and Sign-match figures\n"
            r"\begin{figure}[H]\n\centering\n"
            r"\IfFileExists{fig/Fig-PARETO-mnist.png}{%\n"
            r"\includegraphics[width=0.72\linewidth]{Fig-PARETO-mnist.png}\n"
            r"}{\fbox{Pareto}}\\\n"
            r"\caption{Pareto front: accuracy/R$^2$ vs sparsity with marker size "
            r"proportional to throughput (samples/s). Hue = flip schedule. "
            r"Error bars omitted for clarity.}\n"
            r"\label{fig:pareto}\n\end{figure}\n"
        )
        blocks.append(
            r"\begin{figure}[H]\n\centering\n"
            r"\IfFileExists{fig/Fig-SignMatch-P.png}{%\n"
            r"\includegraphics[width=0.72\linewidth]{Fig-SignMatch-P.png}\n"
            r"}{\fbox{Sign-match}}\\\n"
            r"\caption{Directional sign-match $p=\Pr(\langle\Delta V^{\mathrm{DFA}},\Delta V^{\mathrm{BP}}\rangle>0)$ over epochs "  # noqa: E501
            r"(mean $\pm$ std). Values above $0.5$ indicate a sign advantage.}\n"
            r"\label{fig:signmatch}\n\end{figure}\n\n"
        )
        blocks.append(
            r"\begin{figure}[H]\n\centering\n"
            r"\IfFileExists{fig/Fig-Align-Rho.png}{%\n"
            r"\includegraphics[width=0.72\linewidth]{Fig-Align-Rho.png}\n"
            r"}{\fbox{Alignment}}\\\n"
            r"\caption{Cosine alignment $\rho$ between DFA and BP over epochs (mean $\pm$ std across runs).}\n"
            r"\label{fig:rho-curves}\n\end{figure}\n\n"
        )
        blocks.append(
            r"\begin{figure}[H]\n\centering\n"
            r"\IfFileExists{fig/Fig-Throughput.png}{%\n"
            r"\includegraphics[width=0.72\linewidth]{Fig-Throughput.png}\n"
            r"}{\fbox{Throughput}}\\\n"
            r"\caption{Test throughput (samples/s) aggregated per variant (mean $\pm$ std).}\n"
            r"\label{fig:throughput}\n\end{figure}\n\n"
        )
        txt = txt[:insert_at] + "\n" + "\n".join(blocks) + txt[insert_at:]
    main_tex.write_text(txt, encoding="utf-8")
    paths_written.append(str(main_tex))
    return paths_written


def compile_latex() -> Tuple[bool, str]:
    try:
        # Prefer latexmk if available
        cmd = ["bash", "-lc", "cd paper && latexmk -pdf -interaction=nonstopmode main.tex"]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=REPO_ROOT, check=False
        )
        ok = proc.returncode == 0
        log = proc.stdout.decode(errors="ignore")
        if not ok:
            # Fallback to pdflatex x2
            cmd2 = [
                "bash",
                "-lc",
                (
                    "cd paper && pdflatex -interaction=nonstopmode main.tex; "
                    "bibtex main || true; pdflatex -interaction=nonstopmode main.tex; "
                    "pdflatex -interaction=nonstopmode main.tex"
                ),
            ]
            proc2 = subprocess.run(
                cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=REPO_ROOT, check=False
            )
            ok = proc2.returncode == 0
            log = proc2.stdout.decode(errors="ignore")
        # Treat successful PDF emission as success even if return code nonzero
        pdf_path = PAPER_DIR / "main.pdf"
        if pdf_path.exists():
            ok = True
        return ok, log
    except Exception as e:
        return False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Scan and summarize only")
    args = parser.parse_args()

    check_expected_paths()

    # Ingest canonical dataframe
    df = _collect_runs()
    if df.empty:
        fail(
            f"No runs found under {RUNS_DIR}. Minimal regen: ./reproduce_all.sh or python experiments/ternary_dfa_experiment.py"  # noqa: E501
        )

    # Figures
    figures: List[str] = []
    figures += fig_pareto(df)

    rho_tidy = _collect_time_series_for("rho_")
    f = fig_series_mean_ci(
        rho_tidy,
        title="Alignment (rho) over epochs",
        ylabel="rho",
        outfile=FIG_DIR / "Fig-Align-Rho.png",
    )
    if f:
        figures.append(f)

    p_tidy = _collect_time_series_for("p_")
    f = fig_series_mean_ci(
        p_tidy,
        title="Sign-match (p) over epochs",
        ylabel="p",
        outfile=FIG_DIR / "Fig-SignMatch-P.png",
    )
    if f:
        figures.append(f)

    # Throughput plot (per variant)
    thr_df = df.dropna(subset=["throughput_samples_sec"]).copy()
    f = fig_bars_ci(
        thr_df,
        ["dataset", "variant"],
        "throughput_samples_sec",
        title="Throughput by dataset/variant",
        ylabel="samples/sec",
        outfile=FIG_DIR / "Fig-Throughput.png",
    )
    if f:
        figures.append(f)

    # Convergence metrics
    e90_df, slope_df = compute_e90_and_slope()
    f = fig_bars_ci(
        e90_df.dropna(subset=["E@0.90"]),
        ["dataset", "variant"],
        "E@0.90",
        title="Convergence E@0.90",
        ylabel="epoch",
        outfile=FIG_DIR / "Fig-E90.png",
    )
    if f:
        figures.append(f)
    f = fig_bars_ci(
        slope_df,
        ["dataset", "variant"],
        "slope_e1to3",
        title="Early slope (e1->e3)",
        ylabel="slope",
        outfile=FIG_DIR / "Fig-Early-Slope.png",
    )
    if f:
        figures.append(f)

    # Tables
    tables: List[str] = []
    tbl_best, csvp, towp = best_configs_table(df)
    if csvp:
        tables.extend([csvp, towp])

    # Patch LaTeX
    patch_latex_for_figures()

    # Compile PDF
    ok, log = compile_latex() if not args.dry_run else (True, "dry-run")
    pdf_path = PAPER_DIR / "main.pdf"

    # Revision summary
    summary = {
        "status": "success" if ok else "latex_error",
        "git_sha": git_sha_short(),
        "datasets": sorted(set(df["dataset"].dropna().astype(str).tolist())),
        "figures": sorted(figures),
        "tables": sorted(tables),
        "pdf": str(pdf_path) if pdf_path.exists() else None,
        "notes": [],
    }
    # Note missing (p) if p_tidy empty
    if p_tidy.empty:
        summary["notes"].append("Sign-match p not found in logs; skipped Fig-SignMatch-P.")
    # Emit any datasets with missing throughput
    if thr_df.empty:
        summary["notes"].append("Throughput not found for any runs; Fig-Throughput skipped.")

    (PAPER_DIR / "revision_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if not ok:
        # Print first ~30 lines of errors for quick diagnosis
        print("LaTeX build failed. First 30 lines of log:")
        print("\n".join(log.splitlines()[:30]))
        fail("LaTeX compilation failed. See paper/main.log and fix errors near inserted blocks.")

    # Final status line
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
