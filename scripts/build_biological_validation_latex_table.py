"""Build the plain RQ4 study-by-method LaTeX table from canonical evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/published/v0.2.4/target-discovery-v1/cytometry.csv"
OUTPUT = ROOT / "manuscript/oedm2026/tables/table_rq4_biological_by_study.tex"
METHODS = [
    "StrataSCAN",
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "kNN-DBSCAN",
    "kNN+Leiden",
]
LABELS = {
    "kNN-DBSCAN": r"$k$NN-DBSCAN",
    "kNN+Leiden": r"$k$NN+Leiden",
}


def study(dataset_id: str) -> str:
    return "samusik" if dataset_id.startswith("samusik_") else dataset_id


def number(value: float) -> str:
    return r"---" if pd.isna(value) else f"{value:.3f}"


def main() -> None:
    frame = pd.read_csv(SOURCE)
    frame["study"] = frame["dataset_id"].map(study)
    frame["macro_target_f1"] = pd.to_numeric(frame["macro_target_f1"], errors="coerce")
    frame["target_discovery_rate"] = pd.to_numeric(
        frame["target_discovery_rate"], errors="coerce"
    )
    frame.loc[~frame["status"].eq("ok"), ["macro_target_f1", "target_discovery_rate"]] = pd.NA

    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & Levine & Mosmann\textsuperscript{a} & Nilsson & Samusik\textsuperscript{b} & Median F1 & Discovery & Complete \\",
        r"\midrule",
    ]
    for method in METHODS:
        part = frame.loc[frame["method"].eq(method)]
        cells: dict[str, str] = {}
        for current in ("levine", "mosmann", "nilsson"):
            values = part.loc[part["study"].eq(current), "macro_target_f1"].dropna()
            cells[current] = number(values.iloc[0]) if len(values) else r"---"
        samusik = part.loc[part["study"].eq("samusik") & part["status"].eq("ok"), "macro_target_f1"].dropna()
        if len(samusik):
            cells["samusik"] = f"{samusik.mean():.3f} $\\pm$ {samusik.std(ddof=1):.3f}"
            if len(samusik) != 10:
                cells["samusik"] += f" ({len(samusik)}/10)"
        else:
            cells["samusik"] = r"---"
        successful = part.loc[part["status"].eq("ok")]
        median = number(successful["macro_target_f1"].median())
        discovery = number(successful["target_discovery_rate"].mean())
        complete = f"{len(successful)}/13"
        label = LABELS.get(method, method)
        # Bold the best value in each reported column; completion ties are retained.
        if method == "StrataSCAN":
            for current in ("levine", "mosmann", "nilsson"):
                cells[current] = rf"\textbf{{{cells[current]}}}"
            discovery = rf"\textbf{{{discovery}}}"
        elif method == "kNN+Leiden":
            cells["samusik"] = rf"\textbf{{{cells['samusik']}}}"
            median = rf"\textbf{{{median}}}"
        if len(successful) == 13:
            complete = rf"\textbf{{{complete}}}"
        lines.append(
            f"{label} & {cells['levine']} & {cells['mosmann']} & {cells['nilsson']} & "
            f"{cells['samusik']} & {median} & {discovery} & {complete} \\\\" 
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
