from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "StrataSCAN",
    "kNN+Leiden",
    "VDBSCAN-2007",
    "SNN-DBSCAN",
    "kNN-DBSCAN",
    "DBSCAN",
    "OPTICS",
]
COLORS = {"StrataSCAN": "#d95f02", "kNN+Leiden": "#1b9e77"}
NEUTRAL = "#8a8f98"


def build_source(population_path: Path, result_paths: list[Path]) -> pd.DataFrame:
    population = pd.read_csv(population_path)
    population["sample_id"] = population["sample_id"].astype(int)
    quality = population.groupby(["sample_id", "method"], as_index=False).agg(
        macro_population_recall=("recall", "mean"),
        macro_population_precision=("precision", "mean"),
    )
    raw = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    raw = raw.loc[raw["status"].eq("ok") & raw["method"].isin(METHODS)].copy()
    raw["sample_id"] = raw["dataset_id"].str[-2:].astype(int)
    runtime = raw.groupby(["sample_id", "method"], as_index=False).agg(
        runtime_seconds=("runtime_seconds", "median")
    )
    return quality.merge(runtime, on=["sample_id", "method"], validate="one_to_one")


def summarize(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for method, frame in source.groupby("method"):
        row: dict[str, float | str] = {"method": method}
        for column in (
            "runtime_seconds",
            "macro_population_recall",
            "macro_population_precision",
        ):
            row[f"{column}_q1"] = float(frame[column].quantile(0.25))
            row[column] = float(frame[column].median())
            row[f"{column}_q3"] = float(frame[column].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows).set_index("method").reindex(METHODS).reset_index()


def pareto_methods(summary: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _, candidate in summary.iterrows():
        dominated = np.any(
            (summary["runtime_seconds"] <= candidate["runtime_seconds"])
            & (summary["macro_population_recall"] >= candidate["macro_population_recall"])
            & (
                (summary["runtime_seconds"] < candidate["runtime_seconds"])
                | (summary["macro_population_recall"] > candidate["macro_population_recall"])
            )
        )
        keep.append(not dominated)
    return summary.loc[keep].sort_values("runtime_seconds")


def point_color(method: str) -> str:
    return COLORS.get(method, NEUTRAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Samusik population recall versus runtime")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--population-by-sample",
        type=Path,
        default=Path("results/samusik_stability/population_by_sample.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/samusik_stability")
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = build_source(args.population_by_sample, args.results)
    source.to_csv(output / "recall_runtime_by_sample.csv", index=False)
    summary = summarize(source)
    summary.to_csv(output / "recall_runtime_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), constrained_layout=True)
    label_offsets_a = {
        "StrataSCAN": (7, 7),
        "kNN+Leiden": (7, -13),
        "VDBSCAN-2007": (-79, 8),
        "SNN-DBSCAN": (8, 8),
        "kNN-DBSCAN": (-5, -16),
        "DBSCAN": (7, 5),
        "OPTICS": (-42, 7),
    }
    label_offsets_b = {
        "StrataSCAN": (7, 5),
        "kNN+Leiden": (-80, 7),
        "VDBSCAN-2007": (-88, 7),
        "SNN-DBSCAN": (7, 5),
        "kNN-DBSCAN": (7, -13),
        "DBSCAN": (7, 5),
        "OPTICS": (7, 5),
    }

    ax = axes[0]
    frontier = pareto_methods(summary)
    ax.plot(
        frontier["runtime_seconds"],
        frontier["macro_population_recall"],
        color="#b8bdc5",
        lw=1.5,
        ls="--",
        zorder=1,
    )
    for _, row in summary.iterrows():
        method = str(row["method"])
        color = point_color(method)
        ax.errorbar(
            row["runtime_seconds"],
            row["macro_population_recall"],
            xerr=[
                [row["runtime_seconds"] - row["runtime_seconds_q1"]],
                [row["runtime_seconds_q3"] - row["runtime_seconds"]],
            ],
            yerr=[
                [row["macro_population_recall"] - row["macro_population_recall_q1"]],
                [row["macro_population_recall_q3"] - row["macro_population_recall"]],
            ],
            fmt="o",
            ms=8 if method == "StrataSCAN" else 6,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            method,
            (row["runtime_seconds"], row["macro_population_recall"]),
            xytext=label_offsets_a[method],
            textcoords="offset points",
            color=color,
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlim(8, 240)
    ax.set_ylim(-0.04, 1.0)
    ax.set_xlabel("Estimator runtime, seconds (log scale)  ↓")
    ax.set_ylabel("Macro population recall  ↑")
    ax.set_title("a  Speed–recall trade-off", loc="left", fontweight="bold")
    ax.grid(color="#d9dde3", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for _, row in summary.iterrows():
        method = str(row["method"])
        color = point_color(method)
        ax.errorbar(
            row["macro_population_recall"],
            row["macro_population_precision"],
            xerr=[
                [row["macro_population_recall"] - row["macro_population_recall_q1"]],
                [row["macro_population_recall_q3"] - row["macro_population_recall"]],
            ],
            yerr=[
                [row["macro_population_precision"] - row["macro_population_precision_q1"]],
                [row["macro_population_precision_q3"] - row["macro_population_precision"]],
            ],
            fmt="o",
            ms=8 if method == "StrataSCAN" else 6,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            method,
            (row["macro_population_recall"], row["macro_population_precision"]),
            xytext=label_offsets_b[method],
            textcoords="offset points",
            color=color,
            fontsize=8,
        )
    ax.set_xlim(-0.04, 1.0)
    ax.set_ylim(-0.02, 0.43)
    ax.set_xlabel("Macro population recall  ↑")
    ax.set_ylabel("Macro population precision  ↑")
    ax.set_title("b  Recall–precision trade-off", loc="left", fontweight="bold")
    ax.grid(color="#d9dde3", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Samusik per-population recovery across 10 samples",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.018,
        "Points are cross-sample medians; error bars are interquartile ranges. Precision and recall are averaged equally over 24 annotated populations within each sample.",
        ha="center",
        fontsize=8.5,
        color="#4f5661",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"fig-samusik-recall-runtime.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    caption = """**Figure | Population recall, runtime, and precision on Samusik.** Algorithms were fitted independently to each of 10 mouse samples. Within each sample, precision and recall were calculated for every annotated population against its best-matching predicted cluster and then averaged with equal population weight; stochastic seeds were first collapsed by the median. Points denote cross-sample medians and error bars denote interquartile ranges. (a) Estimator runtime versus macro population recall; the dashed line joins non-dominated median speed–recall solutions. (b) Macro population recall versus macro population precision. StrataSCAN is orange and kNN+Leiden is green. HDBSCAN and AMD-DBSCAN are omitted because they did not complete Samusik_01 screening.\n"""
    (output / "fig-samusik-recall-runtime-caption.md").write_text(
        caption, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
