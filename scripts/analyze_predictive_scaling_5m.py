from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    new = pd.concat([pd.read_csv(path) for path in args.new], ignore_index=True)
    new = new.drop_duplicates(["task_key", "q"], keep="last")
    new = new.loc[new["n"].isin([500_000, 1_000_000, 2_000_000, 5_000_000])].copy()
    if len(new) != 28 or new["task_key"].nunique() != 28:
        raise ValueError(
            f"expected 28 extended-scaling conditions, got rows={len(new)}, "
            f"tasks={new['task_key'].nunique()}"
        )
    if (new["status"] != "ok").any():
        raise ValueError("new-version extended scaling contains failed rows")
    new["new_seconds"] = new["full_primary_seconds"]
    new.to_csv(args.output / "extended-scaling-new-results.csv", index=False)

    reference = pd.read_csv(args.reference)
    reference = reference.loc[
        reference["n"].isin([500_000, 1_000_000, 2_000_000, 5_000_000])
    ].copy()
    reference["dataset_id"] = reference["dataset_id"].str.replace(
        r"__scaling__n\d+$", "", regex=True
    )
    comparison = new.merge(
        reference[
            [
                "dataset_id",
                "n",
                "runtime_seconds",
                "process_peak_rss_mb",
                "macro_target_f1",
                "pairwise_f1",
                "n_clusters",
            ]
        ],
        on=["dataset_id", "n"],
        suffixes=("_new", "_current"),
        validate="one_to_one",
    )
    comparison["delta_macro_target_f1"] = (
        comparison["macro_target_f1_new"] - comparison["macro_target_f1_current"]
    )
    comparison["runtime_ratio"] = comparison["new_seconds"] / comparison["runtime_seconds"]
    comparison.to_csv(args.output / "extended-scaling-comparison.csv", index=False)

    summary = (
        comparison.groupby("n", as_index=False)
        .agg(
            conditions=("dataset_id", "size"),
            new_median_f1=("macro_target_f1_new", "median"),
            current_median_f1=("macro_target_f1_current", "median"),
            mean_f1_delta=("delta_macro_target_f1", "mean"),
            min_f1_delta=("delta_macro_target_f1", "min"),
            max_f1_delta=("delta_macro_target_f1", "max"),
            new_median_seconds=("new_seconds", "median"),
            current_median_seconds=("runtime_seconds", "median"),
            median_runtime_ratio=("runtime_ratio", "median"),
        )
    )
    summary.to_csv(args.output / "extended-scaling-summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(summary["n"], summary["new_median_seconds"], marker="o", label="Predictive")
    axes[0].plot(
        summary["n"], summary["current_median_seconds"], marker="o", label="Current"
    )
    axes[0].set(xscale="log", yscale="log", xlabel="n", ylabel="median seconds", title="Runtime")
    axes[0].legend(frameon=False)
    axes[1].plot(summary["n"], summary["new_median_f1"], marker="o", label="Predictive")
    axes[1].plot(
        summary["n"], summary["current_median_f1"], marker="o", label="Current"
    )
    axes[1].set(xscale="log", xlabel="n", ylabel="median macro target F1", title="Quality")
    axes[1].legend(frameon=False)
    fig.savefig(args.output / "extended-scaling-overview.png", dpi=190)
    plt.close(fig)

    lines = [
        f"- n={int(row.n):,}: median runtime {row.new_median_seconds:.1f}s vs "
        f"{row.current_median_seconds:.1f}s ({row.median_runtime_ratio:.2f}x); "
        f"median F1 {row.new_median_f1:.3f} vs {row.current_median_f1:.3f}."
        for row in summary.itertuples()
    ]
    worst = comparison.loc[comparison["delta_macro_target_f1"].idxmin()]
    best = comparison.loc[comparison["delta_macro_target_f1"].idxmax()]
    report = f"""# Predictive StrataSCAN extended scalability to 5M

All 28 prespecified conditions completed successfully: seven scenarios at 500k, 1M, 2M and 5M.

{chr(10).join(lines)}

Largest quality gain: {best.dataset_id} at n={int(best.n):,}, delta {best.delta_macro_target_f1:+.3f}. Largest loss: {worst.dataset_id} at n={int(worst.n):,}, delta {worst.delta_macro_target_f1:+.3f}.

The new runner records end-of-task RSS, not a continuously sampled process peak. Memory comparisons therefore use only observed process RSS and must not be presented as strict peak-memory estimates.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
