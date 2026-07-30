"""Summarize the locked v0.2.2 synthetic evidence matrix without selection.

The runner records controlled failures as ordinary job rows.  This script uses
zero quality for those rows, retains their status, and refuses to make tables
or figures until every manifest job is represented exactly once.  It is an
analysis utility, not a result snapshot: generated artifacts belong beside a
validated completed run, not in the source tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUALITY_METRICS = (
    "macro_target_f1",
    "noise_f1",
    "pairwise_f1",
    "target_background_structure_hmean_f1",
    "target_success_rate_f1_080",
)
STRUCTURAL_METRICS = (
    "mean_target_fragments",
    "spurious_background_majority_clusters",
    "merged_predicted_clusters",
)


def _case_id(dataset_id: str) -> str:
    return dataset_id.split("__quality__", maxsplit=1)[0]


def _bootstrap_mean_interval(values: pd.Series, *, seed: int = 20260730) -> tuple[float, float]:
    data = values.to_numpy(dtype=float)
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, data.size, size=(4000, data.size))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, (0.025, 0.975)))


def load_and_validate(run_dir: Path, protocol_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("complete"):
        raise ValueError("run structural validation is incomplete; no evidence outputs were produced")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    results = pd.read_csv(run_dir / "results.csv")
    expected_methods = list(protocol["methods"])
    expected_cases = [case["id"] for case in protocol["synthetic"]["cases"]]
    expected_seeds = list(protocol["synthetic"]["quality_seeds"])
    frame = results.loc[results["suite"].eq("synthetic")].copy()
    frame["case"] = frame["dataset_id"].map(_case_id)
    expected = len(expected_methods) * len(expected_cases) * len(expected_seeds)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} synthetic rows, found {len(frame)}")
    observed = frame[["case", "method", "seed"]]
    if observed.duplicated().any() or set(frame["case"]) != set(expected_cases):
        raise ValueError("synthetic rows do not form the locked case/method/seed matrix")
    if set(frame["method"]) != set(expected_methods) or set(frame["seed"].astype(int)) != set(expected_seeds):
        raise ValueError("synthetic methods or seeds do not match the locked protocol")
    for metric in QUALITY_METRICS + STRUCTURAL_METRICS:
        if metric not in frame:
            raise ValueError(f"results lack required metric {metric!r}")
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    # Failures remain visible and receive the predeclared zero quality penalty.
    failed = ~frame["status"].eq("ok")
    frame["failure_penalized"] = failed
    for metric in QUALITY_METRICS:
        frame.loc[failed, metric] = 0.0
        frame[metric] = frame[metric].fillna(0.0)
    for metric in STRUCTURAL_METRICS:
        frame.loc[failed, metric] = np.nan
    return frame, protocol


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (case, method), group in frame.groupby(["case", "method"], sort=True, observed=True):
        row: dict[str, object] = {
            "case": case,
            "method": method,
            "attempted": len(group),
            "completed": int(group["status"].eq("ok").sum()),
            "failures": int(group["failure_penalized"].sum()),
        }
        for metric in QUALITY_METRICS + STRUCTURAL_METRICS:
            values = group[metric]
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_q25"] = float(values.quantile(0.25))
            row[f"{metric}_q75"] = float(values.quantile(0.75))
            low, high = _bootstrap_mean_interval(values.dropna())
            row[f"{metric}_mean_ci95_low"] = low
            row[f"{metric}_mean_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def seed_level_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    primary = frame.loc[frame["method"].eq("StrataSCAN"), ["case", "seed", *QUALITY_METRICS]].copy()
    primary = primary.rename(columns={metric: f"stratascan_{metric}" for metric in QUALITY_METRICS})
    rows: list[pd.DataFrame] = []
    for method in sorted(set(frame["method"]) - {"StrataSCAN"}):
        other = frame.loc[frame["method"].eq(method), ["case", "seed", *QUALITY_METRICS]].copy()
        joined = primary.merge(other, on=["case", "seed"], how="inner", validate="one_to_one")
        joined.insert(2, "comparator", method)
        for metric in QUALITY_METRICS:
            joined[f"delta_{metric}"] = joined[f"stratascan_{metric}"] - joined[metric]
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def reversal_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, comparator), group in deltas.groupby(["case", "comparator"], sort=True, observed=True):
        for metric in QUALITY_METRICS:
            values = group[f"delta_{metric}"]
            rows.append({
                "case": case,
                "comparator": comparator,
                "metric": metric,
                "median_delta": float(values.median()),
                "q25_delta": float(values.quantile(0.25)),
                "q75_delta": float(values.quantile(0.75)),
                "stratascan_wins": int((values > 0).sum()),
                "ties": int((values == 0).sum()),
                "stratascan_losses": int((values < 0).sum()),
                "reversal_observed": bool((values > 0).any() and (values < 0).any()),
            })
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    metric = "target_background_structure_hmean_f1_median"
    table = summary.pivot(index="method", columns="case", values=metric)
    fig, ax = plt.subplots(figsize=(max(8, table.shape[1] * 0.75), max(4, table.shape[0] * 0.32)))
    image = ax.imshow(table.to_numpy(float), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=7)
    ax.set_title("Locked synthetic target–background–structure F1 (median over five seeds)")
    fig.colorbar(image, ax=ax, label="failure-penalized TBS-HF1")
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"fig_locked_synthetic_tbs_hf1.{extension}", dpi=300)
    plt.close(fig)


def write_report(output: Path, protocol: dict[str, object], summary: pd.DataFrame, reversals: pd.DataFrame) -> None:
    total_failures = int(summary["failures"].sum())
    reversal_count = int(reversals["reversal_observed"].sum())
    lines = [
        "# Locked v0.2.2 synthetic evidence",
        "",
        f"Protocol: `{protocol['protocol_version']}`. Every locked job was present exactly once.",
        f"The matrix contains {len(summary)} case-method summaries and {total_failures} controlled failures; failures receive zero only for quality endpoints.",
        f"Seed-level gain reversals were observed in {reversal_count} case/comparator/metric comparisons and are retained in `seed_level_deltas.csv` and `reversal_summary.csv`.",
        "",
        "The tables are descriptive: seeds quantify variation among generated datasets, not independent biological studies. No method-selection or significance claim is implied.",
    ]
    (output / "EVIDENCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the locked synthetic evidence matrix")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("benchmarks/protocol.v0.2.2-synthetic-evidence.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, protocol = load_and_validate(args.run_dir, args.protocol)
    summary = condition_summary(frame)
    deltas = seed_level_deltas(frame)
    reversals = reversal_summary(deltas)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "seed_level_results.csv", index=False)
    summary.to_csv(args.output / "condition_summary.csv", index=False)
    deltas.to_csv(args.output / "seed_level_deltas.csv", index=False)
    reversals.to_csv(args.output / "reversal_summary.csv", index=False)
    make_figure(summary, args.output)
    write_report(args.output, protocol, summary, reversals)


if __name__ == "__main__":
    main()
