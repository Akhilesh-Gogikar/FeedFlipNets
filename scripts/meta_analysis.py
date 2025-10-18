import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

RUNS_DIR = Path("runs")
PLOTS_DIR = Path("data/report/plots")
REPORT_CSV = Path("data/report/meta_convergence_summary.csv")


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate truncated lines
                continue
    return rows


def first_epoch_crossing(
    df: pd.DataFrame, metric: str, threshold: float, mode: str = "max"
) -> float:
    """
    Return first epoch where a metric crosses a threshold.
    Falls back to the last epoch if it never crosses.

    mode = "max" for accuracy-like (>= thresh), "min" for loss-like (<= thresh).
    """
    if df.empty or metric not in df:
        return float("nan")
    if mode == "max":
        crossed = df[df[metric] >= threshold]
    else:
        crossed = df[df[metric] <= threshold]
    if not crossed.empty:
        return float(crossed.iloc[0]["epoch"])  # first crossing
    return float(df["epoch"].max())


def early_slope(df: pd.DataFrame, metric: str, epochs: int = 3) -> float:
    """
    Improvement slope over first `epochs`: (last - first) / (epochs - 1).
    """
    if df.empty or metric not in df:
        return float("nan")
    sub = df.sort_values("epoch").head(epochs)
    if len(sub) < 2:
        return float("nan")
    return (float(sub.iloc[-1][metric]) - float(sub.iloc[0][metric])) / (len(sub) - 1)


def analyze_run(run_dir: Path) -> Dict:
    name = run_dir.name
    train_path = run_dir / "metrics.jsonl"
    val_path = run_dir / "metrics_val.jsonl"
    # keep lines <=88 for flake8; avoid black collapsing by using helper variables
    train_exists = train_path.exists()
    val_exists = val_path.exists()
    train = pd.DataFrame(load_jsonl(train_path)) if train_exists else pd.DataFrame()
    val = pd.DataFrame(load_jsonl(val_path)) if val_exists else pd.DataFrame()

    # Infer task type
    is_classif = "accuracy" in train.columns or "accuracy" in val.columns
    is_reg = "r2" in train.columns or "r2" in val.columns

    rec: Dict[str, float] = {"run": name}
    if is_classif:
        acc_df = val if "accuracy" in val.columns else train
        if not acc_df.empty:
            best_acc = float(acc_df["accuracy"].max())
            threshold = 0.9 * best_acc
        else:
            threshold = float("nan")
        epochs_to_rel = first_epoch_crossing(acc_df, "accuracy", threshold, mode="max")
        rec["epochs_to_90pct_best_acc"] = epochs_to_rel
        rec["early_acc_slope_e3"] = early_slope(acc_df, "accuracy", epochs=3)
    if is_reg:
        r2_df = val if "r2" in val.columns else train
        # threshold: 90% of max R2 achieved in this run
        if not r2_df.empty:
            target = 0.9 * float(r2_df["r2"].max())
        else:
            target = float("nan")
        to_90pct = first_epoch_crossing(r2_df, "r2", target, mode="max")
        rec["epochs_to_90pct_r2"] = to_90pct
        rec["early_r2_slope_e3"] = early_slope(r2_df, "r2", epochs=3)

    # Generic stability proxies
    if not val.empty:
        has_loss = "loss" in val.columns
        rec["val_loss_var"] = float(val["loss"].var()) if has_loss else float("nan")
    if not train.empty:
        rec["train_loss_var_e5"] = (
            float(train.sort_values("epoch").head(5)["loss"].var())
            if "loss" in train.columns
            else float("nan")
        )

    return rec


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        # Skip non-run directories
        if (
            not (run_dir / "metrics.jsonl").exists()
            and not (run_dir / "metrics_train.jsonl").exists()
        ):
            continue
        try:
            rec = analyze_run(run_dir)
            records.append(rec)
        except Exception:
            continue

    if not records:
        print("No run metrics found.")
        return

    df = pd.DataFrame.from_records(records)
    df.to_csv(REPORT_CSV, index=False)

    # Plot: epochs to 0.90 accuracy (classification only)
    cls = (
        df.dropna(subset=["epochs_to_90pct_best_acc"])
        if "epochs_to_90pct_best_acc" in df.columns
        else pd.DataFrame()
    )
    if not cls.empty:
        plt.figure(figsize=(6, 3.0))
        cls_sorted = cls.sort_values("epochs_to_90pct_best_acc")
        y = cls_sorted["run"]
        vals = cls_sorted["epochs_to_90pct_best_acc"]
        plt.barh(y, vals, color="#4e79a7")
        plt.xlabel("Epochs to 90% of best accuracy")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "meta_epochs_to_90pct_acc.png", dpi=200)
        plt.close()

    # Plot: early slopes (acc or r2)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.0))
    if "early_acc_slope_e3" in df.columns and df["early_acc_slope_e3"].notna().any():
        a = df.dropna(subset=["early_acc_slope_e3"]).sort_values(
            "early_acc_slope_e3", ascending=False
        )
        ax[0].barh(a["run"], a["early_acc_slope_e3"], color="#59a14f")
        ax[0].set_title("Early acc slope (e1→e3)")
    else:
        ax[0].axis("off")

    if "early_r2_slope_e3" in df.columns and df["early_r2_slope_e3"].notna().any():
        r = df.dropna(subset=["early_r2_slope_e3"]).sort_values(
            "early_r2_slope_e3", ascending=False
        )
        ax[1].barh(r["run"], r["early_r2_slope_e3"], color="#e15759")
        ax[1].set_title("Early R2 slope (e1→e3)")
    else:
        ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "meta_early_slopes.png", dpi=200)
    plt.close()

    print(f"Wrote {REPORT_CSV} and plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
