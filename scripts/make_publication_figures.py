from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


METHOD_ORDER = [
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

METHOD_COLORS = {
    "DBSCAN": "#4477AA",
    "HDBSCAN": "#66CCEE",
    "OPTICS": "#228833",
    "SNN-DBSCAN": "#CCBB44",
    "VDBSCAN-2007": "#EE6677",
    "AMD-DBSCAN": "#AA3377",
    "kNN-DBSCAN": "#BBBBBB",
    "kNN+Leiden": "#EE7733",
    "StrataSCAN": "#000000",
}

METHOD_MARKERS = {
    "DBSCAN": "o",
    "HDBSCAN": "s",
    "OPTICS": "^",
    "SNN-DBSCAN": "v",
    "VDBSCAN-2007": "D",
    "AMD-DBSCAN": "P",
    "kNN-DBSCAN": "X",
    "kNN+Leiden": "<",
    "StrataSCAN": "*",
}

DATASET_LABELS = {
    "multidensity_2d": "Multidensity\n2D",
    "ultrasparse_16d": "Ultrasparse\n16D",
    "overlapping_density_16d": "Overlapping density\n16D",
    "moons_2d": "Moons\n2D",
    "rings_2d": "Rings\n2D",
    "overlap_8d": "Gaussian overlap\n8D",
    "imbalanced_16d": "Imbalanced\n16D",
}


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
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(output / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)


def ok_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["status"].eq("ok")].copy()


def dataset_base(frame: pd.DataFrame) -> pd.Series:
    return frame["dataset_id"].astype(str).str.replace(r"__(quality|scaling)__n\d+$", "", regex=True)


def figure_quality_heatmap(frame: pd.DataFrame, output: Path) -> None:
    syn = ok_rows(frame.loc[frame["suite"].eq("synthetic")]).copy()
    syn["dataset"] = dataset_base(syn)
    quality = syn.loc[syn["dataset_id"].str.contains("__quality__")]
    max_n = int(quality["n"].max())
    quality = quality.loc[quality["n"].eq(max_n)]
    table = quality.pivot_table(
        index="method", columns="dataset", values="pairwise_f1", aggfunc="median"
    ).reindex(index=METHOD_ORDER, columns=list(DATASET_LABELS))

    fig, ax = plt.subplots(figsize=(7.15, 3.65), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("quality", ["#F7FBFF", "#6BAED6", "#08306B"])
    image = ax.imshow(table.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(table.shape[1]), [DATASET_LABELS[x] for x in table.columns])
    ax.set_yticks(np.arange(table.shape[0]), table.index)
    ax.set_xlabel(f"Synthetic scenario (n = {max_n:,}; median over five seeds)")
    ax.set_ylabel("Method")
    ax.tick_params(length=0)
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table.iat[row, col]
            label = "—" if pd.isna(value) else f"{value:.2f}"
            color = "white" if pd.notna(value) and value >= 0.62 else "black"
            weight = "bold" if table.index[row] == "StrataSCAN" else "normal"
            ax.text(col, row, label, ha="center", va="center", color=color, fontweight=weight)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Pairwise F1")
    ax.axhline(METHOD_ORDER.index("StrataSCAN") - 0.5, color="black", linewidth=1.0)
    save_figure(fig, output, "fig1-synthetic-quality")


def figure_efficiency(frame: pd.DataFrame, output: Path) -> None:
    syn = ok_rows(frame.loc[frame["suite"].eq("synthetic")]).copy()
    syn["dataset"] = dataset_base(syn)
    quality = syn.loc[syn["dataset_id"].str.contains("__quality__")]
    max_n = int(quality["n"].max())
    quality = quality.loc[quality["n"].eq(max_n)]
    summary = quality.groupby("method", observed=True).agg(
        pairwise_f1=("pairwise_f1", "median"),
        runtime_seconds=("runtime_seconds", "median"),
        process_peak_rss_mb=("process_peak_rss_mb", "median"),
    )

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)
    for method in METHOD_ORDER:
        if method not in summary.index:
            continue
        row = summary.loc[method]
        common = {
            "marker": METHOD_MARKERS[method],
            "s": 78 if method == "StrataSCAN" else 34,
            "facecolor": METHOD_COLORS[method],
            "edgecolor": "white" if method != "StrataSCAN" else "black",
            "linewidth": 0.6,
            "zorder": 4 if method == "StrataSCAN" else 3,
            "label": method,
        }
        ax_time.scatter(row["runtime_seconds"], row["pairwise_f1"], **common)
        ax_mem.scatter(row["process_peak_rss_mb"], row["pairwise_f1"], **common)
    for ax, xlabel in (
        (ax_time, "Estimator runtime (s; log scale)"),
        (ax_mem, "Peak process RSS (MB; log scale)"),
    ):
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Median pairwise F1")
        clean_axes(ax)
    panel_label(ax_time, "a")
    panel_label(ax_mem, "b")
    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(f"Quality–resource trade-off across seven synthetic scenarios (n = {max_n:,})", y=1.02)
    save_figure(fig, output, "fig2-quality-resource-pareto")


def _median_iqr(group: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = group.groupby("n", observed=True)[column]
    x = np.array(sorted(values.groups), dtype=float)
    median = values.median().reindex(x.astype(int)).to_numpy(float)
    low = values.quantile(0.25).reindex(x.astype(int)).to_numpy(float)
    high = values.quantile(0.75).reindex(x.astype(int)).to_numpy(float)
    return x, median, low, high


def figure_scaling(frame: pd.DataFrame, output: Path) -> None:
    syn = ok_rows(frame.loc[frame["suite"].eq("synthetic")]).copy()
    scaling = syn.loc[syn["dataset_id"].str.contains("__scaling__")]
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(7.15, 3.2), constrained_layout=True)
    for method in METHOD_ORDER:
        group = scaling.loc[scaling["method"].eq(method)]
        if group.empty:
            continue
        for ax, column in (
            (ax_time, "runtime_seconds"),
            (ax_mem, "process_peak_rss_mb"),
        ):
            x, median, low, high = _median_iqr(group, column)
            color = METHOD_COLORS[method]
            linewidth = 1.8 if method == "StrataSCAN" else 1.0
            ax.plot(
                x,
                median,
                color=color,
                marker=METHOD_MARKERS[method],
                markersize=6 if method == "StrataSCAN" else 3.5,
                linewidth=linewidth,
                label=method,
                zorder=4 if method == "StrataSCAN" else 2,
            )
            ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
    ax_time.set_ylabel("Estimator runtime (s)")
    ax_mem.set_ylabel("Peak process RSS (MB)")
    for ax in (ax_time, ax_mem):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of observations")
        clean_axes(ax)
    panel_label(ax_time, "a")
    panel_label(ax_mem, "b")
    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Scaling across seven synthetic scenarios (median and interquartile range)", y=1.02)
    save_figure(fig, output, "fig3-scaling")


def _styled_boxplot(ax: plt.Axes, groups: list[np.ndarray], positions: np.ndarray) -> None:
    result = ax.boxplot(
        groups,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
        boxprops={"linewidth": 0.7},
    )
    for patch, method in zip(result["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_edgecolor("black")


def figure_real_data(frame: pd.DataFrame, output: Path) -> None:
    cyto = ok_rows(frame.loc[frame["suite"].eq("cytometry")])
    gaia = ok_rows(frame.loc[frame["suite"].eq("gaia")])
    fig, (ax_cyto, ax_gaia) = plt.subplots(1, 2, figsize=(7.15, 3.35), constrained_layout=True)

    cyto_summary = cyto.groupby(["dataset_id", "method"], observed=True)["macro_target_f1"].median()
    datasets = [item for item in ("levine", "mosmann", "nilsson") if item in cyto["dataset_id"].unique()]
    width = 0.25
    x = np.arange(len(METHOD_ORDER))
    hatches = ("", "///", "xx")
    for index, dataset in enumerate(datasets):
        values = [cyto_summary.get((dataset, method), np.nan) for method in METHOD_ORDER]
        ax_cyto.bar(
            x + (index - (len(datasets) - 1) / 2) * width,
            values,
            width=width,
            facecolor=[METHOD_COLORS[m] for m in METHOD_ORDER],
            edgecolor="black",
            linewidth=0.45,
            hatch=hatches[index],
            label=dataset.capitalize(),
            zorder=3,
        )
    ax_cyto.set_xticks(x, METHOD_ORDER, rotation=55, ha="right")
    ax_cyto.set_ylabel("Macro target F1")
    ax_cyto.set_xlabel("Method")
    ax_cyto.set_ylim(0, 1.02)
    ax_cyto.legend(frameon=False, ncol=3, loc="upper center")
    clean_axes(ax_cyto)

    gaia_groups = [
        gaia.loc[gaia["method"].eq(method), "best_cluster_f1"].dropna().to_numpy(float)
        for method in METHOD_ORDER
    ]
    _styled_boxplot(ax_gaia, gaia_groups, x)
    ax_gaia.set_xticks(x, METHOD_ORDER, rotation=55, ha="right")
    ax_gaia.set_ylabel("Best matched-cluster F1")
    ax_gaia.set_xlabel("Method")
    ax_gaia.set_ylim(0, 1.02)
    clean_axes(ax_gaia)
    panel_label(ax_cyto, "a")
    panel_label(ax_gaia, "b")
    ax_cyto.set_title("Cytometry (median over stochastic seeds)")
    ax_gaia.set_title("Gaia (all 359 fields)")
    save_figure(fig, output, "fig4-real-data")


def write_summary(frame: pd.DataFrame, output: Path) -> None:
    ok = ok_rows(frame)
    primary = {"synthetic": "pairwise_f1", "cytometry": "macro_target_f1", "gaia": "best_cluster_f1"}
    rows: list[dict[str, object]] = []
    for suite, metric in primary.items():
        part = ok.loc[ok["suite"].eq(suite)]
        for method in METHOD_ORDER:
            group = part.loc[part["method"].eq(method)]
            rows.append(
                {
                    "suite": suite,
                    "method": method,
                    "jobs_ok": len(group),
                    "median_primary_metric": group[metric].median(),
                    "median_runtime_seconds": group["runtime_seconds"].median(),
                    "median_process_peak_rss_mb": group["process_peak_rss_mb"].median(),
                }
            )
    pd.DataFrame(rows).to_csv(output / "publication-summary.csv", index=False)
    failure = (
        frame.groupby(["suite", "method", "status"], observed=True)
        .size()
        .rename("jobs")
        .reset_index()
    )
    failure.to_csv(output / "job-status-summary.csv", index=False)


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"results table is missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication-ready benchmark figures")
    parser.add_argument("results", type=Path, help="benchmark results.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    require_columns(
        frame,
        (
            "status",
            "suite",
            "dataset_id",
            "method",
            "n",
            "pairwise_f1",
            "macro_target_f1",
            "best_cluster_f1",
            "runtime_seconds",
            "process_peak_rss_mb",
        ),
    )
    configure_style()
    args.output.mkdir(parents=True, exist_ok=True)
    figure_quality_heatmap(frame, args.output)
    figure_efficiency(frame, args.output)
    figure_scaling(frame, args.output)
    figure_real_data(frame, args.output)
    write_summary(frame, args.output)
    provenance = {
        "results": str(args.results.resolve()),
        "rows": int(len(frame)),
        "statuses": {str(k): int(v) for k, v in frame["status"].value_counts().items()},
        "formats": ["pdf", "svg", "png@600dpi"],
    }
    (args.output / "figure-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
