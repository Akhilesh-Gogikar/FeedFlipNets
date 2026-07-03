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
            rows[r["run"]] = {
                "e90": float(r.get("epochs_to_90pct_best_acc") or "nan"),
                "slope": float(r.get("early_acc_slope_e3") or "nan"),
            }
    return rows


def pair(
    rows: Dict[str, Dict[str, float]], base: str, epoch: str
) -> Tuple[float, float, float, float]:
    a = rows.get(base, {"e90": float("nan"), "slope": float("nan")})
    b = rows.get(epoch, {"e90": float("nan"), "slope": float("nan")})
    return a["e90"], a["slope"], b["e90"], b["slope"]


def fmt(x: float, nd: int = 3) -> str:
    if x != x:
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

    de_mn = (mne_e - mn_e) if mn_e == mn_e and mne_e == mne_e else float("nan")
    ds_mn = (mne_s - mn_s) if mn_s == mn_s and mne_s == mne_s else float("nan")
    de_ng = (nge_e - ng_e) if ng_e == ng_e and nge_e == nge_e else float("nan")
    ds_ng = (nge_s - ng_s) if ng_s == ng_s and nge_s == nge_s else float("nan")

    lines = [
        "\\begin{tabular}{l r r r r r r}",
        "Dataset & E@90\\% best (ps) & S(ps) & E@90\\% best (pe) & S(pe) & dE & dS \\",
        "\\hline",
        f"MNIST & {fmt(mn_e)} & {fmt(mn_s)} & {fmt(mne_e)} & {fmt(mne_s)} & {fmt(de_mn)} & {fmt(ds_mn)} \\",  # noqa: E501
        f"20NG & {fmt(ng_e)} & {fmt(ng_s)} & {fmt(nge_e)} & {fmt(nge_s)} & {fmt(de_ng)} & {fmt(ds_ng)} \\",  # noqa: E501
        "\\end{tabular}",
    ]

    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
