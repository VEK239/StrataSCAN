from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "StrataSCAN",
    "kNN+Leiden",
    "VDBSCAN-2007",
    "SNN-DBSCAN",
    "kNN-DBSCAN",
    "DBSCAN",
    "OPTICS",
]
COLORS = {
    "StrataSCAN": "#d95f02",
    "kNN+Leiden": "#1b9e77",
}
NEUTRAL = "#8a8f98"


def load_results(paths: list[Path]) -> pd.DataFrame:
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    data = data.loc[data["method"].isin(METHOD_ORDER) & data["status"].eq("ok")].copy()
    data["sample_id"] = data["dataset_id"].str.removeprefix("samusik_")
    metrics = [
        "runtime_seconds",
        "signal_coverage",
        "background_rejection",
        "spurious_background_majority_clusters",
    ]
    return data.groupby(["sample_id", "method"], as_index=False)[metrics].median()


def strip_summary(
    ax: plt.Axes,
    data: pd.DataFrame,
    value: str,
    *,
    methods: list[str],
    xlabel: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    log: bool = False,
) -> None:
    rng = np.random.default_rng(142)
    for position, method in enumerate(methods):
        values = data.loc[data["method"].eq(method), value].to_numpy(dtype=float)
        color = COLORS.get(method, NEUTRAL)
        jitter = rng.uniform(-0.13, 0.13, size=values.size)
        ax.scatter(
            values,
            position + jitter,
            s=24,
            color=color,
            alpha=0.42 if method != "StrataSCAN" else 0.62,
            edgecolors="none",
            zorder=2,
        )
        median = float(np.median(values))
        ax.plot([median, median], [position - 0.24, position + 0.24], color=color, lw=3, zorder=3)
        ax.scatter(
            [median],
            [position],
            s=52,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            zorder=4,
        )
    ax.set_yticks(range(len(methods)), methods)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    if log:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#d9dde3", lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Samusik StrataSCAN trade-off figure")
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

    samples = load_results(args.results)
    population = pd.read_csv(args.population_by_sample)
    population["sample_id"] = population["sample_id"].astype(str).str.zfill(2)
    stochastic = population.loc[population["method"].isin(["StrataSCAN", "kNN+Leiden"])].copy()
    stochastic["population_f1_seed_range"] = (
        stochastic["seed_f1_max"] - stochastic["seed_f1_min"]
    )
    seed_stability = stochastic.groupby(["sample_id", "method"], as_index=False).agg(
        mean_population_f1_seed_range=("population_f1_seed_range", "mean")
    )

    source = samples.merge(
        seed_stability,
        on=["sample_id", "method"],
        how="left",
    )
    source.to_csv(output / "figure_tradeoff_source.csv", index=False)

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
    fig = plt.figure(figsize=(13.2, 8.4), constrained_layout=True)
    grid = GridSpec(2, 6, figure=fig)
    axes = [
        fig.add_subplot(grid[0, :3]),
        fig.add_subplot(grid[0, 3:]),
        fig.add_subplot(grid[1, :2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:]),
    ]

    strip_summary(
        axes[0],
        seed_stability,
        "mean_population_f1_seed_range",
        methods=["StrataSCAN", "kNN+Leiden"],
        xlabel="Mean population F1 range across seeds  ↓",
        title="a  Seed sensitivity",
        xlim=(-0.003, 0.09),
    )
    strip_summary(
        axes[1],
        samples,
        "runtime_seconds",
        methods=METHOD_ORDER,
        xlabel="Estimator runtime, seconds (log scale)  ↓",
        title="b  Speed",
        log=True,
    )
    strip_summary(
        axes[2],
        samples,
        "signal_coverage",
        methods=METHOD_ORDER,
        xlabel="Annotated cells assigned to clusters  ↑",
        title="c  Signal coverage",
        xlim=(-0.03, 1.03),
    )
    strip_summary(
        axes[3],
        samples,
        "background_rejection",
        methods=METHOD_ORDER,
        xlabel="Unassigned cells sent to noise  ↑",
        title="d  Background rejection",
        xlim=(-0.03, 1.03),
    )
    strip_summary(
        axes[4],
        samples,
        "spurious_background_majority_clusters",
        methods=METHOD_ORDER,
        xlabel="Background-majority clusters  ↓",
        title="e  Background fragmentation",
        xlim=(-1.0, 48.0),
    )

    fig.suptitle(
        "Samusik per-sample behavior: reproducibility, speed, signal retention and background control",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Points are mouse samples; vertical bars and open circles show medians. Stochastic seeds are collapsed within sample except in panel a.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4f5661",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"fig-samusik-stratascan-tradeoffs.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    caption = """**Figure | Samusik per-sample reproducibility, speed, signal retention, and background control.** Algorithms were fitted independently to each of 10 mouse samples. Points denote samples; vertical bars and open circles denote cross-sample medians. Stochastic seeds were collapsed by the within-sample median except in panel a. (a) Mean population-level F1 range across seeds for the two stochastic methods; lower values indicate greater seed reproducibility. (b) Estimator runtime on a logarithmic scale. (c) Signal coverage, defined as the fraction of manually annotated population cells assigned to a non-noise cluster. (d) Background rejection, defined as the fraction of manually unassigned cells assigned to noise. (e) Number of predicted clusters containing more manually unassigned than annotated cells. StrataSCAN is orange and kNN+Leiden is green; other completed methods are gray. HDBSCAN and AMD-DBSCAN are omitted because they did not complete the initial Samusik_01 screening.\n"""
    (output / "fig-samusik-stratascan-tradeoffs-caption.md").write_text(
        caption, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
