import csv
from pathlib import Path
from typing import Dict, Tuple

CSV_PATH = Path("data/report/meta_convergence_summary.csv")
OUT_TEX = Path("data/report/schedule_delta.tex")


def load_meta(csv_path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            run = r["run"]
            rows[run] = {
                "e90": float(r["epochs_to_0.90_acc"]) if r["epochs_to_0.90_acc"] else float("nan"),
                "slope": float(r["early_acc_slope_e3"])
                if r["early_acc_slope_e3"]
                else float("nan"),
            }
    return rows


def pair(
    rows: Dict[str, Dict[str, float]], base: str, epoch: str
) -> Tuple[float, float, float, float]:
    a = rows.get(base, {"e90": float("nan"), "slope": float("nan")})
    b = rows.get(epoch, {"e90": float("nan"), "slope": float("nan")})
    return a["e90"], a["slope"], b["e90"], b["slope"]


def fmt(x: float, nd: int = 3) -> str:
    if x != x:  # NaN
        return ""
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return f"{x:.{nd}f}"


def main() -> None:
    rows = load_meta(CSV_PATH)
    mn_e, mn_s, mne_e, mne_s = pair(rows, "mnist-mlp-dfa", "mnist-mlp-dfa-per-epoch")
    ng_e, ng_s, nge_e, nge_s = pair(
        rows, "20newsgroups-bow-mlp-dfa", "20newsgroups-bow-mlp-dfa-per-epoch"
    )
    with OUT_TEX.open("w") as f:
        f.write("\\begin{tabular}{l r r r r r r}\\n")
        f.write("Dataset & E@0.90(ps) & S(ps) & E@0.90(pe) & S(pe) & dE & dS \\\\n")
        f.write("\\hline\n")
        de = (mne_e - mn_e) if mn_e == mn_e and mne_e == mne_e else float("nan")
        ds = (mne_s - mn_s) if mn_s == mn_s and mne_s == mne_s else float("nan")
        f.write(
            f"MNIST & {fmt(mn_e)} & {fmt(mn_s)} & {fmt(mne_e)} & {fmt(mne_s)} & {fmt(de)} & {fmt(ds)} \\\n"
        )
        de = (nge_e - ng_e) if ng_e == ng_e and nge_e == nge_e else float("nan")
        ds = (nge_s - ng_s) if ng_s == ng_s and nge_s == nge_s else float("nan")
        f.write(
            f"20NG & {fmt(ng_e)} & {fmt(ng_s)} & {fmt(nge_e)} & {fmt(nge_s)} & {fmt(de)} & {fmt(ds)} \\\n"
        )
        f.write("\\end{tabular}\n")


if __name__ == "__main__":
    main()
