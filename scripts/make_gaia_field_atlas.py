from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "AMD-DBSCAN",
    "kNN-DBSCAN",
    "kNN+Leiden",
    "StrataSCAN",
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.svg")
    fig.savefig(output / f"{stem}.png", dpi=600)
    plt.close(fig)


def prepare(path: Path) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(path)
    raw = raw.loc[raw["suite"].eq("gaia")].copy()
    if raw["dataset_id"].nunique() != 359:
        raise ValueError(f"expected 359 Gaia fields, found {raw['dataset_id'].nunique()}")
    if set(raw["method"]) != set(METHODS):
        raise ValueError("Gaia method set does not match the prespecified nine-method panel")
    raw["reference_members"] = raw["dataset_metadata_json"].map(
        lambda value: int(json.loads(value)["reference_members"])
    )
    columns = [
        "best_cluster_f1",
        "background_rejection",
        "spurious_background_majority_clusters",
        "reference_members",
        "n",
    ]
    field = (
        raw.groupby(["dataset_id", "method"], observed=True)[columns]
        .median()
        .reset_index()
    )
    if len(field) != 359 * len(METHODS):
        raise ValueError("field-method matrix is incomplete")

    difficulty = field.groupby("dataset_id", observed=True).agg(
        consensus_f1=("best_cluster_f1", "median"),
        consensus_background_rejection=("background_rejection", "median"),
        reference_members=("reference_members", "first"),
    )
    difficulty["field_id_numeric"] = pd.to_numeric(
        difficulty.index.to_series().str.extract(r"(\d+)$", expand=False), errors="coerce"
    )
    order = (
        difficulty.sort_values(
            ["consensus_f1", "consensus_background_rejection", "reference_members", "field_id_numeric"],
            ascending=[True, True, True, True],
        )
        .index.astype(str)
        .tolist()
    )
    return field, order


def matrix(field: pd.DataFrame, order: list[str], metric: str) -> pd.DataFrame:
    return (
        field.pivot(index="method", columns="dataset_id", values=metric)
        .reindex(index=METHODS, columns=order)
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.055, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def heatmap_row(
    ax: plt.Axes,
    summary_ax: plt.Axes,
    colorbar_ax: plt.Axes,
    values: pd.DataFrame,
    title: str,
    colorbar_label: str,
    summary: pd.Series,
    summary_label: str,
    reverse: bool = False,
) -> None:
    cmap = mpl.colormaps["viridis_r" if reverse else "viridis"]
    if reverse:
        plotted = np.log1p(values.to_numpy(float))
        vmax = float(np.nanquantile(plotted, 0.99))
        image = ax.imshow(plotted, cmap=cmap, vmin=0, vmax=max(vmax, 1), aspect="auto", interpolation="nearest")
    else:
        image = ax.imshow(values.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(METHODS)), METHODS)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "StrataSCAN":
            tick.set_fontweight("bold")
    ax.set_xticks([])
    ax.set_title(title, loc="left")
    ax.tick_params(length=0)
    strata = METHODS.index("StrataSCAN")
    ax.axhline(strata - 0.5, color="black", linewidth=0.8)
    ax.axhline(strata + 0.5, color="black", linewidth=0.8)
    cbar = ax.figure.colorbar(image, cax=colorbar_ax)
    cbar.set_label(colorbar_label)

    y = np.arange(len(METHODS))
    summary_ax.barh(y, summary.reindex(METHODS), color="#808080", height=0.62)
    summary_ax.set_yticks([])
    summary_ax.invert_yaxis()
    summary_ax.set_xlabel(summary_label)
    summary_ax.spines["top"].set_visible(False)
    summary_ax.spines["right"].set_visible(False)
    summary_ax.grid(axis="x", color="#D9D9D9", linewidth=0.45, zorder=0)
    summary_ax.axhline(strata - 0.5, color="black", linewidth=0.8)
    summary_ax.axhline(strata + 0.5, color="black", linewidth=0.8)
    upper = max(float(summary.max()) * 1.12, 1.0 if summary.max() <= 1 else 1.0)
    summary_ax.set_xlim(0, upper)
    for yi, method in enumerate(METHODS):
        value = float(summary.loc[method])
        label = f"{100 * value:.0f}%" if summary.max() <= 1 else f"{value:.0f}"
        summary_ax.text(value + upper * 0.015, yi, label, va="center", fontsize=6.2)


def make_figure(field: pd.DataFrame, order: list[str], output: Path) -> None:
    target = matrix(field, order, "best_cluster_f1")
    background = matrix(field, order, "background_rejection")
    fragmentation = matrix(field, order, "spurious_background_majority_clusters")

    target_summary = (target >= 0.8).mean(axis=1)
    background_summary = background.median(axis=1)
    fragmentation_summary = fragmentation.median(axis=1)

    fig = plt.figure(figsize=(7.2, 5.45), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, width_ratios=[14.0, 2.6, 0.42], hspace=0.13, wspace=0.10)
    axes = []
    for row in range(3):
        axes.append((fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1]), fig.add_subplot(grid[row, 2])))

    heatmap_row(
        *axes[0], target, "Reference-cluster recovery", "Best matched-cluster F1",
        target_summary, "Fields with\nF1 ≥ 0.8",
    )
    heatmap_row(
        *axes[1], background, "Background rejected as noise", "Background rejection",
        background_summary, "Median background\nrejection",
    )
    heatmap_row(
        *axes[2], fragmentation, "Background fragmentation", "log(1 + background-majority clusters)",
        fragmentation_summary, "Median background-\nmajority clusters", reverse=True,
    )
    axes[2][0].set_xticks([0, len(order) - 1], ["harder fields", "easier fields"])
    axes[2][0].set_xlabel("All 359 Gaia fields in a shared order based on cross-method recovery difficulty")
    for label, (ax, _, _) in zip("abc", axes):
        panel_label(ax, label)
    save(fig, output, "fig1-gaia-field-level-atlas")


def write_caption(output: Path) -> None:
    caption = """# Figure caption

**Figure 1 | Field-level recovery and background behavior of nine clustering algorithms across 359 Gaia fields.** Each matrix contains one column per Gaia field and one row per algorithm. Stochastic methods are first aggregated by the within-field median over the three prespecified seeds. Fields use the same order in all panels and are arranged from lower to higher cross-method median best-cluster F1, with cross-method background rejection, reference membership and field identifier used only to break ties. (a) F1 of the predicted cluster giving the best match to the catalogued reference cluster; the right margin reports the fraction of fields with F1 at least 0.8. (b) Fraction of reference-negative field stars assigned to noise; the right margin reports the field median. (c) Number of predicted clusters dominated by reference-negative stars, shown on a log(1 + x) scale with lower values in yellow and higher values in purple; the right margin reports the field median. The identical column order permits direct field-wise comparison of target recovery, background rejection and background fragmentation. StrataSCAN is outlined only to aid row tracking; all algorithms use the same scales.
"""
    (output / "FIGURE_CAPTION.md").write_text(caption, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Gaia field-level algorithm atlas")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    field, order = prepare(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    make_figure(field, order, args.output)
    field.assign(field_order=field["dataset_id"].map({name: i + 1 for i, name in enumerate(order)})).sort_values(
        ["field_order", "method"]
    ).to_csv(args.output / "gaia-field-level-metrics.csv", index=False)
    pd.DataFrame({"field_order": np.arange(1, len(order) + 1), "dataset_id": order}).to_csv(
        args.output / "gaia-field-order.csv", index=False
    )
    write_caption(args.output)
    provenance = {
        "input": str(args.results.resolve()),
        "fields": len(order),
        "methods": METHODS,
        "seed_aggregation": "median within field and method",
        "field_order": "ascending cross-method median best_cluster_f1; background rejection, reference membership and numeric field id break ties",
        "panels": ["best_cluster_f1", "background_rejection", "spurious_background_majority_clusters"],
        "formats": ["PDF", "SVG", "PNG 600 dpi"],
    }
    (args.output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
