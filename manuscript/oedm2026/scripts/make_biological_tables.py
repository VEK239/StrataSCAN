from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = [
    "StrataSCAN", "DBSCAN", "HDBSCAN", "OPTICS", "SNN-DBSCAN",
    "VDBSCAN-2007", "AMD-DBSCAN", "kNN-DBSCAN", "kNN+Leiden",
]
DATASETS = ["levine", "nilsson"]
METRICS = ["target_f1", "noise_f1", "pairwise_f1"]


def load_filtered(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = {
        "dataset", "method", "ok_fraction", "target_f1", "noise_f1",
        "pairwise_f1", "granularity_ratio", "runtime_seconds",
        "process_peak_rss_mb",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame[frame["dataset"].isin(DATASETS)].copy()
    if set(frame["dataset"]) != set(DATASETS):
        raise ValueError("Both Levine and Nilsson must be present")
    frame["method"] = frame["method"].replace({"kNN + Leiden": "kNN+Leiden"})
    frame["dataset"] = pd.Categorical(frame["dataset"], DATASETS, ordered=True)
    frame["method"] = pd.Categorical(frame["method"], METHODS, ordered=True)
    frame = frame.sort_values(["dataset", "method"]).reset_index(drop=True)
    if len(frame) != len(DATASETS) * len(METHODS):
        raise ValueError(f"Expected 18 method-dataset rows, got {len(frame)}")
    failed = frame["ok_fraction"].fillna(0).eq(0)
    for column in METRICS + ["granularity_ratio", "runtime_seconds", "process_peak_rss_mb"]:
        frame.loc[failed, column] = np.nan
    frame["status"] = np.where(failed, "unavailable", "completed")
    return frame


def metric_cell(row: pd.Series) -> str:
    if row["status"] != "completed":
        return r"--"
    return f"{row['target_f1']:.3f} / {row['noise_f1']:.3f} / {row['pairwise_f1']:.3f}"


def write_quality_table(frame: pd.DataFrame, output: Path) -> None:
    pivot = {(str(r.dataset), str(r.method)): r for _, r in frame.iterrows()}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Performance on the Levine and Nilsson cytometry benchmarks. Each cell reports target-wise F1 / noise F1 / pairwise F1. A dash denotes that the method did not complete under the benchmark resource policy.}",
        r"\label{tab:biology-levine-nilsson}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Levine & Nilsson \\",
        r"\midrule",
    ]
    for method in METHODS:
        name = r"\textbf{StrataSCAN}" if method == "StrataSCAN" else method
        cells = [metric_cell(pivot[(dataset, method)]) for dataset in DATASETS]
        lines.append(f"{name} & {cells[0]} & {cells[1]} " + "\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_resource_table(frame: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Resource and granularity diagnostics for the completed Levine and Nilsson runs. Granularity is the predicted-to-reference cluster-count ratio.}",
        r"\label{tab:biology-levine-nilsson-resources}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Method & Granularity ratio & Runtime (s) & Peak RSS (MB) \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        sub = frame[frame["dataset"].astype(str).eq(dataset)]
        for _, row in sub.iterrows():
            name = r"\textbf{StrataSCAN}" if str(row.method) == "StrataSCAN" else str(row.method)
            if row.status == "completed":
                values = f"{row.granularity_ratio:.2f} & {row.runtime_seconds:.2f} & {row.process_peak_rss_mb:.1f}"
            else:
                values = r"-- & -- & --"
            lines.append(f"{dataset.capitalize()} & {name} & {values} " + "\\\\")
        if dataset != DATASETS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path(__file__).resolve().parents[1] / "tables" / "performance_tables_2026-08-06" / "05_biological_by_dataset.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "tables" / "biology_levine_nilsson",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_filtered(args.source)
    frame.to_csv(args.output_dir / "biological_levine_nilsson_results.csv", index=False)
    write_quality_table(frame, args.output_dir / "table_biological_levine_nilsson.tex")
    write_resource_table(frame, args.output_dir / "table_biological_levine_nilsson_resources.tex")
    manifest = {
        "source": str(args.source), "datasets": DATASETS, "methods": METHODS,
        "failure_policy": "Rows with ok_fraction=0 are written as unavailable; their stored zero placeholders are never interpreted as measured F1 values.",
        "metrics": "target_f1 / noise_f1 / pairwise_f1",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
