from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
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
DATASETS = ["levine", "mosmann", "nilsson"]
DATASET_LABELS = {"levine": "Levine", "mosmann": "Mosmann", "nilsson": "Nilsson"}
COLORS = {
    "DBSCAN": "#4477AA",
    "HDBSCAN": "#66CCEE",
    "OPTICS": "#228833",
    "SNN-DBSCAN": "#CCBB44",
    "VDBSCAN-2007": "#EE6677",
    "AMD-DBSCAN": "#AA3377",
    "kNN-DBSCAN": "#999999",
    "kNN+Leiden": "#EE7733",
    "StrataSCAN": "#000000",
}
MARKERS = {
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
PENALIZED_METRICS = [
    "macro_target_f1",
    "pairwise_f1",
    "noise_f1",
    "target_success_rate_f1_080",
    "signal_coverage",
    "background_rejection",
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
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.svg")
    fig.savefig(directory / f"{stem}.png", dpi=600)
    plt.close(fig)


def panel(ax: plt.Axes, label: str, x: float = -0.12) -> None:
    ax.text(x, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def clean(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(color="#D9D9D9", linewidth=0.45, zorder=0)


def prepare(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "suite",
        "dataset_id",
        "method",
        "seed",
        "status",
        "runtime_seconds",
        "process_runtime_seconds",
        "process_peak_rss_mb",
        "target_matches_json",
        *PENALIZED_METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"results table is missing required columns: {missing}")
    frame = frame.loc[frame["suite"].eq("cytometry")].copy()
    if len(frame) != 39:
        raise ValueError(f"expected 39 frozen cytometry jobs, found {len(frame)}")
    if set(frame["method"]) != set(METHODS) or set(frame["dataset_id"]) != set(DATASETS):
        raise ValueError("cytometry methods or datasets do not match the frozen protocol")
    frame["ok"] = frame["status"].eq("ok")
    for metric in PENALIZED_METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame[f"{metric}_penalized"] = frame[metric].where(frame["ok"], 0.0).fillna(0.0)
    frame["runtime_to_termination_seconds"] = pd.to_numeric(
        frame["runtime_seconds"], errors="coerce"
    ).where(frame["ok"], pd.to_numeric(frame["process_runtime_seconds"], errors="coerce"))
    frame["process_peak_rss_mb"] = pd.to_numeric(frame["process_peak_rss_mb"], errors="coerce")
    return frame


def collapse(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "attempted": ("ok", "size"),
        "completed": ("ok", "sum"),
        "completion_rate": ("ok", "mean"),
        "runtime_success_seconds": ("runtime_seconds", "median"),
        "runtime_to_termination_seconds": ("runtime_to_termination_seconds", "median"),
        "peak_rss_mb": ("process_peak_rss_mb", "median"),
        "n": ("n", "median"),
        "dimension": ("dimension", "median"),
        "truth_clusters": ("truth_clusters", "median"),
        "n_clusters": ("n_clusters", "median"),
        "mean_target_fragments": ("mean_target_fragments", "median"),
        "spurious_background_majority_clusters": (
            "spurious_background_majority_clusters",
            "median",
        ),
    }
    for metric in PENALIZED_METRICS:
        aggregations[metric] = (f"{metric}_penalized", "median")
    return (
        frame.groupby(["dataset_id", "method"], observed=True)
        .agg(**aggregations)
        .reset_index()
    )


def failure_counts(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame.groupby(["dataset_id", "method", "status"], observed=True)
        .size()
        .unstack("status", fill_value=0)
        .reindex(pd.MultiIndex.from_product([DATASETS, METHODS], names=["dataset_id", "method"]), fill_value=0)
    )
    for status in ("ok", "timeout", "memory_limit", "error"):
        if status not in counts:
            counts[status] = 0
    counts["attempted"] = counts[["ok", "timeout", "memory_limit", "error"]].sum(axis=1)
    counts["completion_rate"] = counts["ok"] / counts["attempted"]
    return counts.reset_index()


def _matrix(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        summary.pivot(index="method", columns="dataset_id", values=metric)
        .reindex(index=METHODS, columns=DATASETS)
    )


def _quality_heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    failures: pd.DataFrame,
) -> mpl.image.AxesImage:
    cmap = LinearSegmentedColormap.from_list("quality", ["#F7FBFF", "#6BAED6", "#08306B"])
    image = ax.imshow(table.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DATASETS)), [DATASET_LABELS[x] for x in DATASETS])
    ax.set_yticks(range(len(METHODS)), METHODS)
    ax.set_title(title)
    ax.tick_params(length=0)
    for i, method in enumerate(METHODS):
        for j, dataset in enumerate(DATASETS):
            value = table.loc[method, dataset]
            mark = "†" if failures.loc[(dataset, method)] > 0 else ""
            color = "white" if value >= 0.62 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}{mark}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold" if method == "StrataSCAN" else "normal",
            )
    ax.axhline(METHODS.index("StrataSCAN") - 0.5, color="black", linewidth=1.0)
    return image


def fig_quality(summary: pd.DataFrame, frame: pd.DataFrame, figures: Path) -> None:
    failed = (
        frame.assign(failed=~frame["ok"])
        .groupby(["dataset_id", "method"], observed=True)["failed"]
        .sum()
        .reindex(pd.MultiIndex.from_product([DATASETS, METHODS]), fill_value=0)
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.65), constrained_layout=True)
    metrics = [
        ("macro_target_f1", "Equal-weight target recovery"),
        ("pairwise_f1", "Global clustering agreement"),
        ("noise_f1", "Background classification"),
    ]
    image = None
    for label, ax, (metric, title) in zip("abc", axes, metrics):
        image = _quality_heatmap(ax, _matrix(summary, metric), title, failed)
        panel(ax, label, -0.20)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, fraction=0.022, pad=0.02)
    cbar.set_label("Failure-penalized F1")
    fig.suptitle("Cytometry benchmark across all methods and datasets", y=1.02)
    fig.text(0.5, -0.02, "† at least one controlled failure; failures score zero", ha="center", fontsize=7)
    save(fig, figures, "fig1-cytometry-quality")


def fig_recovery_tradeoff(summary: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.85), constrained_layout=True, sharex=True, sharey=True)
    for label, dataset, ax in zip("abc", DATASETS, axes):
        part = summary.loc[summary["dataset_id"].eq(dataset)].set_index("method")
        for method in METHODS:
            row = part.loc[method]
            ax.scatter(
                row["background_rejection"],
                row["macro_target_f1"],
                s=70 if method == "StrataSCAN" else 30,
                marker=MARKERS[method],
                facecolor=COLORS[method],
                edgecolor="black" if method == "StrataSCAN" else "white",
                linewidth=0.5,
                zorder=4 if method == "StrataSCAN" else 3,
                label=method,
            )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Background rejection")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.02, 0.32)
        clean(ax)
        panel(ax, label, -0.18)
    axes[0].set_ylabel("Macro target F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Target recovery versus background rejection", y=1.02)
    save(fig, figures, "fig2-cytometry-recovery-tradeoff")


def fig_resources(summary: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.65), constrained_layout=True)
    specifications = [
        ("runtime_to_termination_seconds", "Time to completion or termination (s)", True),
        ("peak_rss_mb", "Whole-process peak RSS (MB)", True),
        ("completion_rate", "Completed fraction", False),
    ]
    for label, ax, (metric, title, log_scale) in zip("abc", axes, specifications):
        table = _matrix(summary, metric)
        values = table.to_numpy(float)
        if log_scale:
            positive = values[np.isfinite(values) & (values > 0)]
            norm = LogNorm(vmin=max(1.0, float(positive.min())), vmax=float(positive.max()))
            image = ax.imshow(values, cmap="magma_r", norm=norm, aspect="auto")
        else:
            image = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(DATASETS)), [DATASET_LABELS[x] for x in DATASETS])
        ax.set_yticks(range(len(METHODS)), METHODS if ax is axes[0] else [])
        ax.tick_params(length=0)
        ax.set_title(title)
        for i in range(len(METHODS)):
            for j in range(len(DATASETS)):
                value = values[i, j]
                text = "—" if not np.isfinite(value) else (f"{value:.0f}" if log_scale else f"{value:.2f}")
                if not np.isfinite(value):
                    text_color = "black"
                elif log_scale:
                    text_color = "white" if norm(value) > 0.68 else "black"
                else:
                    text_color = "white" if value < 0.22 or value > 0.82 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color=text_color)
        ax.axhline(METHODS.index("StrataSCAN") - 0.5, color="black", linewidth=1.0)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        panel(ax, label, -0.20)
    fig.suptitle("Computational cost and robustness", y=1.02)
    save(fig, figures, "fig3-cytometry-resources")


def fig_cluster_diagnostics(summary: pd.DataFrame, figures: Path) -> None:
    work = summary.copy()
    work["cluster_count_ratio"] = work["n_clusters"] / work["truth_clusters"]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.65), constrained_layout=True)
    specs = [
        ("cluster_count_ratio", "Predicted / truth clusters", "viridis", 0.0, None),
        ("mean_target_fragments", "Mean target fragments", "magma_r", 0.0, None),
        (
            "spurious_background_majority_clusters",
            "Background-majority clusters",
            "magma_r",
            0.0,
            None,
        ),
    ]
    for label, ax, (metric, title, cmap, vmin, vmax) in zip("abc", axes, specs):
        table = _matrix(work, metric)
        values = table.to_numpy(float)
        finite = values[np.isfinite(values)]
        upper = float(np.quantile(finite, 0.95)) if vmax is None and finite.size else 1.0
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=max(upper, 1e-9), aspect="auto")
        ax.set_xticks(range(len(DATASETS)), [DATASET_LABELS[x] for x in DATASETS])
        ax.set_yticks(range(len(METHODS)), METHODS if ax is axes[0] else [])
        ax.tick_params(length=0)
        ax.set_title(title)
        for i in range(len(METHODS)):
            for j in range(len(DATASETS)):
                value = values[i, j]
                text_color = "white" if np.isfinite(value) and value >= 0.68 * max(upper, 1e-9) else "black"
                ax.text(
                    j,
                    i,
                    "—" if not np.isfinite(value) else f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=text_color,
                )
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02, extend="max")
        panel(ax, label, -0.20)
    fig.suptitle("Cluster-structure diagnostics (successful runs only)", y=1.02)
    save(fig, figures, "fig4-cytometry-cluster-diagnostics")


def target_table(frame: pd.DataFrame) -> pd.DataFrame:
    targets_by_dataset: dict[str, list[str]] = {}
    parsed: dict[int, list[dict[str, object]]] = {}
    for index, row in frame.iterrows():
        if row["ok"]:
            records = json.loads(row["target_matches_json"])
            parsed[index] = records
            targets_by_dataset.setdefault(row["dataset_id"], [])
            targets_by_dataset[row["dataset_id"]].extend(str(item["target"]) for item in records)
    targets_by_dataset = {key: sorted(set(value)) for key, value in targets_by_dataset.items()}
    rows: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        lookup = {str(item["target"]): item for item in parsed.get(index, [])}
        for target in targets_by_dataset[row["dataset_id"]]:
            item = lookup.get(target)
            rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "target": target,
                    "target_size": np.nan if item is None else item.get("target_size", np.nan),
                    "target_f1_penalized": 0.0 if item is None else float(item.get("f1", 0.0)),
                    "target_precision_penalized": 0.0 if item is None else float(item.get("precision", 0.0)),
                    "target_recall_penalized": 0.0 if item is None else float(item.get("recall", 0.0)),
                    "fragments": 0.0 if item is None else float(item.get("fragments", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def fig_target_recovery(targets: pd.DataFrame, figures: Path) -> None:
    collapsed = (
        targets.groupby(["dataset_id", "method", "target"], observed=True)["target_f1_penalized"]
        .median()
        .reset_index()
    )
    for dataset in DATASETS:
        table = (
            collapsed.loc[collapsed["dataset_id"].eq(dataset)]
            .pivot(index="method", columns="target", values="target_f1_penalized")
            .reindex(index=METHODS)
        )
        width = max(7.15, 0.38 * len(table.columns) + 2.2)
        fig, ax = plt.subplots(figsize=(width, 3.5), constrained_layout=True)
        image = ax.imshow(table.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=55, ha="right")
        ax.set_yticks(range(len(METHODS)), METHODS)
        ax.tick_params(length=0)
        for i in range(len(METHODS)):
            for j in range(len(table.columns)):
                value = table.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white" if value >= 0.62 else "black")
        ax.axhline(METHODS.index("StrataSCAN") - 0.5, color="black", linewidth=1.0)
        cbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.015)
        cbar.set_label("Failure-penalized target F1")
        ax.set_title(f"Per-population recovery — {DATASET_LABELS[dataset]}")
        save(fig, figures, f"figS1-target-recovery-{dataset}")


def write_table_variants(table: pd.DataFrame, stem: Path, index: bool = True) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.with_suffix(".csv"), index=index)
    display = table.reset_index() if index else table.copy()
    rows = [
        "| " + " | ".join(map(str, display.columns)) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = ["—" if pd.isna(x) else str(x).replace("|", "\\|") for x in row]
        rows.append("| " + " | ".join(values) + " |")
    stem.with_suffix(".md").write_text("\n".join(rows), encoding="utf-8")
    def escape(value: object) -> str:
        text = "--" if pd.isna(value) else str(value)
        mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
        return "".join(mapping.get(character, character) for character in text)
    latex = [r"\begin{tabular}{l" + "r" * (len(display.columns) - 1) + "}", r"\toprule"]
    latex.append(" & ".join(escape(x) for x in display.columns) + r" \\")
    latex.append(r"\midrule")
    for row in display.itertuples(index=False, name=None):
        latex.append(" & ".join(escape(x) for x in row) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    stem.with_suffix(".tex").write_text("\n".join(latex), encoding="utf-8")


def make_tables(
    frame: pd.DataFrame, summary: pd.DataFrame, targets: pd.DataFrame, tables: Path
) -> dict[str, pd.DataFrame]:
    target_sizes = (
        targets.groupby(["dataset_id", "target"], observed=True)["target_size"]
        .median()
        .groupby("dataset_id")
        .agg(["count", "sum"])
    )
    dataset_characteristics = (
        summary.groupby("dataset_id", observed=True)
        .agg(events=("n", "median"), dimensions=("dimension", "median"), annotated_populations=("truth_clusters", "median"))
        .reindex(DATASETS)
    )
    dataset_characteristics["annotated_signal_events"] = target_sizes["sum"]
    dataset_characteristics["background_fraction"] = (
        1.0 - dataset_characteristics["annotated_signal_events"] / dataset_characteristics["events"]
    )
    write_table_variants(dataset_characteristics.round(4), tables / "table0-dataset-characteristics")

    balanced = summary.groupby("method", observed=True).agg(
        macro_target_f1=("macro_target_f1", "mean"),
        pairwise_f1=("pairwise_f1", "mean"),
        noise_f1=("noise_f1", "mean"),
        target_success_rate_f1_080=("target_success_rate_f1_080", "mean"),
        background_rejection=("background_rejection", "mean"),
        signal_coverage=("signal_coverage", "mean"),
        completion_rate=("completion_rate", "mean"),
        runtime_to_termination_seconds=("runtime_to_termination_seconds", "median"),
        peak_rss_mb=("peak_rss_mb", "median"),
    ).reindex(METHODS)
    write_table_variants(balanced.round(4), tables / "table1-balanced-overview")

    by_dataset = summary.pivot(index="method", columns="dataset_id", values="macro_target_f1").reindex(index=METHODS, columns=DATASETS)
    by_dataset.columns = [f"{DATASET_LABELS[x]} macro target F1" for x in by_dataset.columns]
    write_table_variants(by_dataset.round(4), tables / "table2-macro-target-f1-by-dataset")

    full = summary.copy()
    numeric = full.select_dtypes(include=[np.number]).columns
    full[numeric] = full[numeric].round(5)
    write_table_variants(full, tables / "tableS1-full-dataset-method-summary", index=False)

    statuses = failure_counts(frame).set_index(["dataset_id", "method"])
    write_table_variants(statuses, tables / "tableS2-job-status")

    target_summary = (
        targets.groupby(["dataset_id", "method", "target"], observed=True)
        .agg(
            target_f1=("target_f1_penalized", "median"),
            target_precision=("target_precision_penalized", "median"),
            target_recall=("target_recall_penalized", "median"),
            target_size=("target_size", "median"),
            fragments=("fragments", "median"),
        )
        .reset_index()
    )
    write_table_variants(target_summary.round(5), tables / "tableS3-target-level-recovery", index=False)

    ranks = _matrix(summary, "macro_target_f1").rank(axis=0, ascending=False, method="average")
    rank_table = pd.DataFrame(
        {
            "mean_rank": ranks.mean(axis=1),
            "datasets_won": (_matrix(summary, "macro_target_f1").eq(_matrix(summary, "macro_target_f1").max(axis=0), axis=1)).sum(axis=1),
        }
    ).sort_values("mean_rank")
    write_table_variants(rank_table.round(3), tables / "tableS4-descriptive-ranks")
    return {"balanced": balanced, "ranks": rank_table, "datasets": dataset_characteristics}


def write_captions(output: Path) -> None:
    text = """# Figure captions

## Figure 1 — Cytometry clustering quality

Failure-penalized F1 scores for equal-weight recovery of annotated populations (a), global pairwise clustering agreement (b), and background-versus-signal classification (c). Values for stochastic methods are medians over three prespecified seeds; deterministic methods contribute one run. A dagger denotes at least one timeout, memory-limit termination, or numerical failure. Controlled failures remain in the analysis with quality set to zero.

## Figure 2 — Recovery–rejection trade-off

Failure-penalized macro target F1 versus background rejection for Levine (a), Mosmann (b), and Nilsson (c). The upper-right region is preferred. Marker identity is consistent across panels.

## Figure 3 — Computational cost and robustness

Wall time to successful completion or controlled termination (a), whole-process peak resident set size (b), and fraction of attempted jobs completed (c). Stochastic methods are summarized by the median within each dataset. Time and memory use logarithmic color normalization.

## Figure 4 — Cluster-structure diagnostics

Ratio of predicted to annotated cluster count (a), mean number of recovered fragments per annotated population (b), and number of clusters dominated by background events (c), using successful runs only. Blank cells indicate that the method did not complete on that dataset.

## Supplementary Figure S1 — Per-population recovery

Failure-penalized target F1 for every annotated population, method, and cytometry dataset. Stochastic methods are summarized by the median over prespecified seeds. Separate panels are supplied for Levine, Mosmann, and Nilsson to preserve readable population labels.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(text, encoding="utf-8")


def write_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    targets: pd.DataFrame,
    tables_data: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    balanced = tables_data["balanced"]
    ranks = tables_data["ranks"]
    best = balanced["macro_target_f1"].idxmax()
    completion = balanced["completion_rate"]
    strata = balanced.loc["StrataSCAN"]
    dataset_characteristics = tables_data["datasets"]
    strata_by_dataset = summary.loc[summary["method"].eq("StrataSCAN")].set_index("dataset_id")
    successful_methods = completion.loc[completion.eq(1.0)].index.tolist()
    failed = frame.loc[~frame["ok"], "status"].value_counts().to_dict()
    target_collapsed = (
        targets.groupby(["dataset_id", "method", "target"], observed=True)["target_f1_penalized"]
        .median()
        .reset_index()
    )
    strata_targets = target_collapsed.loc[target_collapsed["method"].eq("StrataSCAN")]
    hardest = strata_targets.nsmallest(5, "target_f1_penalized")
    easiest = strata_targets.nlargest(5, "target_f1_penalized")

    methods_text = """## Methods — cytometry benchmark

Three benchmark cytometry datasets (Levine, Mosmann, and Nilsson) were analyzed using the type-marker panels declared in the frozen protocol. Marker intensities were transformed with an arcsinh transform using dataset-specific cofactors (5 for Levine; 150 for Mosmann and Nilsson). Nine clustering algorithms were evaluated under fixed, predeclared parameter profiles and identical resource limits (one CPU thread, 600 s wall-time limit, and 8,192 MB resident-memory limit). DBSCAN, HDBSCAN, OPTICS, SNN-DBSCAN, VDBSCAN-2007, AMD-DBSCAN, and kNN-DBSCAN were deterministic and were run once per dataset; kNN+Leiden and StrataSCAN were evaluated at three prespecified seeds (23, 42, and 73).

The primary endpoint was macro target F1: each annotated population was matched to its best predicted cluster and population-level F1 scores were averaged with equal weight. Secondary endpoints were pairwise F1, background-classification F1, background rejection, signal coverage, and the proportion of annotated populations recovered at F1 ≥ 0.80. To prevent survivorship bias, a timeout, memory-limit termination, or numerical failure contributed zero to quality endpoints. Replicates of stochastic methods were first collapsed by the median within dataset; cross-dataset summaries then weighted the three datasets equally. Runtime summaries used estimator time for successful runs and process time to termination for failed runs. Because only three independent datasets were available, comparisons are reported descriptively; seed replicates were not treated as independent biological replicates and no confirmatory significance claims were made.
"""
    hard_text = ", ".join(
        f"{row.dataset_id}/{row.target} ({row.target_f1_penalized:.3f})" for row in hardest.itertuples()
    )
    easy_text = ", ".join(
        f"{row.dataset_id}/{row.target} ({row.target_f1_penalized:.3f})" for row in easiest.itertuples()
    )
    lines = [
        "# Publication-ready cytometry analysis",
        "",
        "## Scope and accounting",
        "",
        "The frozen cytometry matrix contains 39 attempted jobs spanning three datasets and nine algorithms. "
        "All expected jobs are present in the parent benchmark manifest. Deterministic algorithms contribute "
        "three dataset-level attempts each; kNN+Leiden and StrataSCAN contribute nine attempts each.",
        "",
        f"Controlled failures: {json.dumps(failed, sort_keys=True)}. Methods completing every attempted job: "
        f"{', '.join(successful_methods)}.",
        "",
        "Dataset sizes were "
        f"{int(dataset_characteristics.loc['levine', 'events']):,} events × {int(dataset_characteristics.loc['levine', 'dimensions'])} markers (Levine), "
        f"{int(dataset_characteristics.loc['mosmann', 'events']):,} × {int(dataset_characteristics.loc['mosmann', 'dimensions'])} (Mosmann), and "
        f"{int(dataset_characteristics.loc['nilsson', 'events']):,} × {int(dataset_characteristics.loc['nilsson', 'dimensions'])} (Nilsson).",
        "",
        "## Results — manuscript text",
        "",
        f"Across the three equally weighted cytometry datasets, {best} had the highest descriptive mean "
        f"failure-penalized macro target F1 ({balanced.loc[best, 'macro_target_f1']:.3f}) and mean rank "
        f"{ranks.loc[best, 'mean_rank']:.2f}. Completion materially affected the comparison: AMD-DBSCAN "
        "did not complete any dataset, while HDBSCAN, OPTICS, and VDBSCAN-2007 completed only a subset; "
        "kNN+Leiden completed only Nilsson. DBSCAN, SNN-DBSCAN, kNN-DBSCAN, and StrataSCAN completed all "
        "prespecified attempts.",
        "",
        f"StrataSCAN's balanced mean macro target F1 was {strata['macro_target_f1']:.3f}, with dataset-level "
        f"values of {strata_by_dataset.loc['levine', 'macro_target_f1']:.3f} (Levine), "
        f"{strata_by_dataset.loc['mosmann', 'macro_target_f1']:.3f} (Mosmann), and "
        f"{strata_by_dataset.loc['nilsson', 'macro_target_f1']:.3f} (Nilsson). Its balanced pairwise F1 was "
        f"{strata['pairwise_f1']:.3f}, background-classification F1 was {strata['noise_f1']:.3f}, and all "
        "nine runs completed. These results show robust execution but dataset-dependent biological recovery; "
        "the method should not be described as uniformly superior on cytometry.",
        "",
        "No method recovered any annotated population at the prespecified F1 ≥ 0.80 threshold. Accordingly, "
        "the principal cytometry result is low absolute biological recovery across the frozen parameter profiles, "
        "not evidence of publication-level annotation performance by any algorithm.",
        "",
        f"For StrataSCAN, the five lowest per-population recovery scores were {hard_text}. The five highest "
        f"were {easy_text}. This population-level heterogeneity is visible in Supplementary Figure S1 and "
        "should accompany aggregate scores in the manuscript.",
        "",
        "## Interpretation and limitations",
        "",
        "The comparison supports conclusions about robustness under the frozen computational protocol and about "
        "descriptive recovery on these three datasets. It does not establish general superiority across cytometry "
        "platforms. The small number of independent datasets, dataset-specific annotation conventions, fixed "
        "parameter profiles, and unequal replicate counts between deterministic and stochastic algorithms limit "
        "inferential claims. Failure penalization is intentionally conservative and should be reported together "
        "with the unambiguous completion table.",
        "",
        methods_text,
        "## Recommended article placement",
        "",
        "- Main text: Figures 1–3 and Table 1.",
        "- Supplement: Figure 4, the three Figure S1 panels, and Tables S1–S4.",
        "- Use `FIGURE_CAPTIONS.md` as the caption source and retain the failure-policy sentence.",
    ]
    (output / "CYTOMETRY_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-ready cytometry evidence")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    frame = prepare(args.results)
    summary = collapse(frame)
    targets = target_table(frame)
    figures = args.output / "figures"
    tables = args.output / "tables"
    args.output.mkdir(parents=True, exist_ok=True)
    fig_quality(summary, frame, figures)
    fig_recovery_tradeoff(summary, figures)
    fig_resources(summary, figures)
    fig_cluster_diagnostics(summary, figures)
    fig_target_recovery(targets, figures)
    tables_data = make_tables(frame, summary, targets, tables)
    write_captions(args.output)
    write_report(frame, summary, targets, tables_data, args.output)
    provenance = {
        "input": str(args.results.resolve()),
        "input_sha256": sha256(args.results),
        "rows_total_input": int(len(pd.read_csv(args.results, usecols=["suite"]))),
        "cytometry_rows": int(len(frame)),
        "datasets": DATASETS,
        "methods": METHODS,
        "failure_policy": "controlled failures retained with zero quality",
        "replicate_policy": "median within dataset and method; datasets weighted equally",
        "figure_formats": ["PDF", "SVG", "PNG 600 dpi"],
        "figure_groups": 7,
    }
    (args.output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
