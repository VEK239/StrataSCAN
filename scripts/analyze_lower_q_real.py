from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize_curve(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    return (
        frame.groupby("q", as_index=False)
        .agg(
            datasets=("dataset_id", "size"),
            mean_score=(endpoint, "mean"),
            median_score=(endpoint, "median"),
            success_rate_080=(endpoint, lambda values: np.mean(values >= 0.80)),
            median_clusters=("n_clusters", "median"),
            median_binary_macro_f1=("binary_macro_f1", "median"),
        )
        .sort_values("q")
    )


def paired_q(frame: pd.DataFrame, endpoint: str, output: Path, prefix: str) -> pd.DataFrame:
    baseline = frame.loc[np.isclose(frame["q"], 0.95), ["dataset_id", endpoint]].rename(
        columns={endpoint: "q95_score"}
    )
    rows = []
    rng = np.random.default_rng(42)
    for q, part in frame.groupby("q"):
        paired = part[["dataset_id", endpoint]].merge(
            baseline, on="dataset_id", validate="one_to_one"
        )
        delta = paired[endpoint].to_numpy(float) - paired["q95_score"].to_numpy(float)
        draws = rng.choice(delta, size=(10_000, len(delta)), replace=True).mean(axis=1)
        rows.append(
            {
                "q": q,
                "datasets": len(delta),
                "mean_delta_vs_q95": np.mean(delta),
                "ci025": np.quantile(draws, 0.025),
                "ci975": np.quantile(draws, 0.975),
                "improved": np.sum(delta > 1e-12),
                "degraded": np.sum(delta < -1e-12),
                "unchanged": np.sum(np.abs(delta) <= 1e-12),
            }
        )
    result = pd.DataFrame(rows).sort_values("q")
    result.to_csv(output / f"{prefix}-paired-vs-q95.csv", index=False)
    return result


def best_rows(frame: pd.DataFrame, endpoint: str, q_filter=None) -> pd.DataFrame:
    source = frame if q_filter is None else frame.loc[q_filter(frame["q"])]
    records = []
    for dataset_id, part in source.groupby("dataset_id"):
        maximum = part[endpoint].max()
        candidates = part.loc[np.isclose(part[endpoint], maximum)].copy()
        candidates["distance_to_q95"] = (candidates["q"] - 0.95).abs()
        records.append(candidates.sort_values("distance_to_q95").iloc[0])
    return pd.DataFrame(records).drop(columns="distance_to_q95")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cytometry", type=Path, required=True)
    parser.add_argument("--gaia", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cytometry = pd.read_csv(args.cytometry)
    gaia = pd.concat([pd.read_csv(path) for path in args.gaia], ignore_index=True)
    gaia = gaia.drop_duplicates(["task_key", "q"], keep="last")
    if gaia["dataset_id"].nunique() != 359:
        raise ValueError(f"expected 359 Gaia fields, got {gaia['dataset_id'].nunique()}")
    if (cytometry["status"] != "ok").any() or (gaia["status"] != "ok").any():
        raise ValueError("lower-q sweep contains failed rows")

    cy_curve = summarize_curve(cytometry, "macro_target_f1")
    ga_curve = summarize_curve(gaia, "best_cluster_f1")
    cy_curve.to_csv(args.output / "cytometry-global-q-summary.csv", index=False)
    ga_curve.to_csv(args.output / "gaia-global-q-summary.csv", index=False)
    cy_paired = paired_q(cytometry, "macro_target_f1", args.output, "cytometry")
    ga_paired = paired_q(gaia, "best_cluster_f1", args.output, "gaia")

    cy_best = best_rows(cytometry, "macro_target_f1")
    cy_best_low = best_rows(cytometry, "macro_target_f1", lambda q: q < 0.30)
    cy_q95 = cytometry.loc[np.isclose(cytometry["q"], 0.95), ["dataset_id", "macro_target_f1"]]
    cy_best = cy_best.merge(cy_q95, on="dataset_id", suffixes=("", "_q95"))
    cy_best["delta_vs_q95"] = cy_best["macro_target_f1"] - cy_best["macro_target_f1_q95"]
    cy_best.to_csv(args.output / "cytometry-best-q-by-dataset.csv", index=False)
    cy_best_low.to_csv(args.output / "cytometry-best-q-below-030.csv", index=False)

    ga_best_low = best_rows(gaia, "best_cluster_f1", lambda q: q < 0.30)
    ga_q95 = gaia.loc[np.isclose(gaia["q"], 0.95), ["dataset_id", "best_cluster_f1"]]
    ga_best_low = ga_best_low.merge(ga_q95, on="dataset_id", suffixes=("", "_q95"))
    ga_best_low["delta_vs_q95"] = (
        ga_best_low["best_cluster_f1"] - ga_best_low["best_cluster_f1_q95"]
    )
    ga_best_low.to_csv(args.output / "gaia-best-q-below-030-by-field.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for dataset_id, part in cytometry.groupby("dataset_id"):
        if dataset_id in {"levine", "mosmann", "nilsson"}:
            axes[0].plot(part["q"], part["macro_target_f1"], marker="o", label=dataset_id)
    axes[0].set(xlabel="q", ylabel="macro target F1", title="Cytometry lower-q sensitivity")
    axes[0].legend(frameon=False)
    axes[1].plot(ga_curve["q"], ga_curve["mean_score"], marker="o", label="mean")
    axes[1].plot(ga_curve["q"], ga_curve["success_rate_080"], marker="o", label="F1 >= 0.80")
    axes[1].set(xlabel="q", ylabel="score", title="Gaia: one fixed q over 359 fields")
    axes[1].legend(frameon=False)
    fig.savefig(args.output / "lower-q-real-overview.png", dpi=190)
    plt.close(fig)

    best_fixed_gaia = ga_curve.loc[ga_curve["mean_score"].idxmax()]
    best_fixed_cyto = cy_curve.loc[cy_curve["mean_score"].idxmax()]
    low_improved = int(np.sum(ga_best_low["delta_vs_q95"] > 1e-12))
    low_degraded = int(np.sum(ga_best_low["delta_vs_q95"] < -1e-12))
    report = f"""# Lower-q sensitivity on real data

All 13 cytometry datasets and all 359 Gaia fields were evaluated at q = 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 0.75 and 0.95. All jobs completed successfully.

## Cytometry

The best single fixed q by mean score across all 13 datasets is {best_fixed_cyto.q:.2f} (mean macro target F1 {best_fixed_cyto.mean_score:.3f}). Per-dataset supervised optima are recorded in `cytometry-best-q-by-dataset.csv`.

## Gaia

The best single fixed q is {best_fixed_gaia.q:.2f}: mean best-cluster F1 {best_fixed_gaia.mean_score:.3f}, with F1>=0.80 success rate {best_fixed_gaia.success_rate_080:.3f}. Selecting the best q below 0.30 separately using each field's labels improves {low_improved}/359 fields relative to q=0.95 and degrades {low_degraded}/359; this is an optimistic, supervised upper bound rather than a deployable rule.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
