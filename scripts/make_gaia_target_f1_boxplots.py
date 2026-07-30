from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATASCAN = "StrataSCAN"


def prepare(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.loc[
        raw["suite"].eq("gaia")
        & raw["status"].eq("ok")
        & raw["best_cluster_f1"].notna()
    ].copy()
    field = (
        raw.groupby(["dataset_id", "method"], observed=True)["best_cluster_f1"]
        .median()
        .reset_index()
    )
    if field["dataset_id"].nunique() != 359:
        raise ValueError(f"Expected 359 Gaia fields, found {field['dataset_id'].nunique()}")
    return field


def summarize(field: pd.DataFrame) -> pd.DataFrame:
    return (
        field.groupby("method", observed=True)["best_cluster_f1"]
        .agg(
            fields="size",
            mean_f1="mean",
            sd_f1="std",
            median_f1="median",
            q1_f1=lambda x: x.quantile(0.25),
            q3_f1=lambda x: x.quantile(0.75),
            min_f1="min",
            max_f1="max",
        )
        .sort_values(["mean_f1", "median_f1"], ascending=False)
        .reset_index()
    )


def draw(field: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    order = summary["method"].tolist()[::-1]
    groups = [
        field.loc[field["method"].eq(method), "best_cluster_f1"].to_numpy(float)
        for method in order
    ]
    positions = np.arange(len(order))

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.7, 3.65), constrained_layout=True)
    boxes = ax.boxplot(
        groups,
        positions=positions,
        orientation="horizontal",
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
    )
    for box, method in zip(boxes["boxes"], order):
        box.set_facecolor("#E69F00" if method == STRATASCAN else "#D9D9D9")
        box.set_edgecolor("#8A5A00" if method == STRATASCAN else "#555555")
        box.set_linewidth(0.9)

    rng = np.random.default_rng(20260723)
    for y, (method, values) in enumerate(zip(order, groups)):
        jitter = rng.uniform(-0.17, 0.17, size=len(values))
        color = "#B66D00" if method == STRATASCAN else "#777777"
        ax.scatter(values, y + jitter, s=4, alpha=0.20, color=color, linewidths=0, zorder=2)
        mean = float(np.mean(values))
        ax.scatter(
            mean,
            y,
            marker="D",
            s=27,
            facecolor="#0072B2",
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
        ax.annotate(
            f"{mean:.3f}",
            (mean, y),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7,
            color="#333333",
            zorder=6,
        )

    ax.set_yticks(positions, order)
    for label in ax.get_yticklabels():
        if label.get_text() == STRATASCAN:
            label.set_fontweight("bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Best matched-cluster F1")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.scatter([], [], marker="D", s=27, color="#0072B2", label="Mean F1")
    ax.legend(loc="upper left", frameon=False)

    output.mkdir(parents=True, exist_ok=True)
    stem = output / "fig-gaia-target-f1-boxplots"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    field = prepare(args.results)
    summary = summarize(field)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "table-gaia-target-f1-summary.csv", index=False)
    draw(field, summary, args.output)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
