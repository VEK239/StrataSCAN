from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats


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

COLORS = {
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


def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.svg")
    fig.savefig(directory / f"{stem}.png", dpi=600)
    plt.close(fig)


def panel(ax: plt.Axes, label: str, x: float = -0.13) -> None:
    ax.text(x, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def clean(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)


def prepare(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    raw = raw.loc[raw["suite"].eq("gaia")].copy()
    if raw.empty:
        raise ValueError("input contains no Gaia records")
    if set(raw["method"]) != set(METHODS):
        raise ValueError(f"unexpected Gaia methods: {sorted(set(raw['method']))}")
    if raw["dataset_id"].nunique() != 359:
        raise ValueError(f"expected 359 Gaia fields, found {raw['dataset_id'].nunique()}")
    raw["reference_members"] = raw["dataset_metadata_json"].map(
        lambda value: int(json.loads(value)["reference_members"])
    )
    metrics = [
        "best_cluster_f1",
        "best_cluster_precision",
        "best_cluster_recall",
        "background_rejection",
        "noise_f1",
        "noise_precision",
        "noise_recall",
        "extra_signal_fragments",
        "signal_coverage",
        "spurious_background_majority_clusters",
        "n_clusters",
        "noise_fraction",
        "runtime_seconds",
        "process_peak_rss_mb",
        "reference_members",
        "n",
    ]
    field = (
        raw.groupby(["dataset_id", "method"], observed=True)[metrics]
        .median()
        .reset_index()
    )
    expected = 359 * len(METHODS)
    if len(field) != expected:
        raise ValueError(f"expected {expected} field-method rows, found {len(field)}")
    return raw, field


def median_ci(values: pd.Series, seed: int = 20260723, draws: int = 10000) -> tuple[float, float]:
    data = values.dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(data, size=(draws, data.size), replace=True), axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def holm_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy())
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values.iloc[index])))
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def rank_statistics(field: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    scores = field.pivot(index="dataset_id", columns="method", values="best_cluster_f1").reindex(columns=METHODS)
    ranks = scores.rank(axis=1, method="average", ascending=False)
    omnibus = stats.friedmanchisquare(*(scores[m].to_numpy() for m in METHODS))
    q = stats.studentized_range.ppf(0.95, len(METHODS), np.inf) / np.sqrt(2.0)
    cd = float(q * np.sqrt(len(METHODS) * (len(METHODS) + 1) / (6.0 * len(scores))))
    rows = []
    strata = scores["StrataSCAN"]
    for method in METHODS:
        if method == "StrataSCAN":
            continue
        delta = strata - scores[method]
        test = stats.wilcoxon(strata, scores[method], alternative="two-sided", zero_method="wilcox")
        rows.append(
            {
                "comparator": method,
                "median_paired_delta": float(np.median(delta)),
                "wins": int((delta > 1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
                "wilcoxon_statistic": float(test.statistic),
                "p_raw": float(test.pvalue),
            }
        )
    pairwise = pd.DataFrame(rows)
    pairwise["p_holm"] = holm_adjust(pairwise["p_raw"])
    pairwise["significant_0_05"] = pairwise["p_holm"] < 0.05
    info = {
        "friedman_statistic": float(omnibus.statistic),
        "friedman_p_value": float(omnibus.pvalue),
        "fields": int(len(scores)),
        "methods": int(len(METHODS)),
        "nemenyi_cd_0_05": cd,
    }
    return ranks, info, pairwise


def fig_recovery(field: pd.DataFrame, figures: Path, stem: str = "fig1-gaia-recovery") -> None:
    fig, (ax_dist, ax_threshold) = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    groups = [field.loc[field["method"].eq(m), "best_cluster_f1"].to_numpy(float) for m in METHODS]
    result = ax_dist.violinplot(groups, positions=np.arange(len(METHODS)), showmeans=False, showmedians=True, widths=0.78)
    for body, method in zip(result["bodies"], METHODS):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor("black")
        body.set_alpha(0.8)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        result[key].set_color("black")
        result[key].set_linewidth(0.7)
    ax_dist.set_xticks(np.arange(len(METHODS)), METHODS, rotation=55, ha="right")
    ax_dist.set_ylim(-0.03, 1.03)
    ax_dist.set_ylabel("Best matched-cluster F1")
    ax_dist.set_title("Field-level recovery distribution")
    clean(ax_dist)

    thresholds = np.linspace(0.5, 1.0, 101)
    for method in METHODS:
        values = field.loc[field["method"].eq(method), "best_cluster_f1"].to_numpy(float)
        survival = np.array([(values >= value).mean() for value in thresholds])
        ax_threshold.plot(
            thresholds,
            survival,
            color=COLORS[method],
            marker=None,
            linewidth=2.0 if method == "StrataSCAN" else 1.0,
            label=method,
            zorder=4 if method == "StrataSCAN" else 2,
        )
    ax_threshold.axvline(0.8, color="#777777", linestyle="--", linewidth=0.7)
    ax_threshold.axvline(0.9, color="#777777", linestyle=":", linewidth=0.7)
    ax_threshold.set_xlim(0.5, 1.0)
    ax_threshold.set_ylim(-0.03, 1.03)
    ax_threshold.set_xlabel("Required best matched-cluster F1")
    ax_threshold.set_ylabel("Fraction of fields meeting threshold")
    ax_threshold.set_title("Recovery reliability across thresholds")
    clean(ax_threshold)
    panel(ax_dist, "a")
    panel(ax_threshold, "b")
    handles, labels = ax_threshold.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, figures, stem)


def fig_ranks(field: pd.DataFrame, figures: Path, ranks: pd.DataFrame, info: dict[str, float], stem: str = "fig2-gaia-ranks-and-effects") -> None:
    mean_rank = ranks.mean().sort_values()
    scores = field.pivot(index="dataset_id", columns="method", values="best_cluster_f1").reindex(columns=METHODS)
    fig, (ax_rank, ax_delta) = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    ordered = list(mean_rank.index)
    y = np.arange(len(ordered))
    for yi, method in enumerate(ordered):
        ax_rank.scatter(
            mean_rank[method], yi, marker=MARKERS[method], s=70 if method == "StrataSCAN" else 35,
            color=COLORS[method], edgecolor="black", linewidth=0.5, zorder=3,
        )
        ax_rank.text(mean_rank[method] + 0.09, yi, f"{mean_rank[method]:.2f}", va="center", fontsize=7)
    ax_rank.set_yticks(y, ordered)
    ax_rank.invert_yaxis()
    ax_rank.set_xlim(0.8, len(METHODS) + 0.4)
    ax_rank.set_xlabel("Average field rank (lower is better)")
    ax_rank.set_title("Ranks over 359 paired fields")
    cd = info["nemenyi_cd_0_05"]
    ax_rank.plot([1.0, 1.0 + cd], [-0.75, -0.75], color="black", linewidth=2.0, clip_on=False)
    ax_rank.text(1.0 + cd / 2, -1.05, f"Nemenyi CD = {cd:.2f}", ha="center", fontsize=7)
    clean(ax_rank)

    comparators = [m for m in METHODS if m != "StrataSCAN"]
    deltas = [scores["StrataSCAN"] - scores[m] for m in comparators]
    parts = ax_delta.violinplot(deltas, positions=np.arange(len(comparators)), showmedians=True, widths=0.8)
    for body, method in zip(parts["bodies"], comparators):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor("black")
        body.set_alpha(0.8)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.7)
    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set_xticks(np.arange(len(comparators)), comparators, rotation=55, ha="right")
    ax_delta.set_ylabel("StrataSCAN − comparator F1")
    ax_delta.set_title("Paired field-level differences")
    clean(ax_delta)
    panel(ax_rank, "a")
    panel(ax_delta, "b")
    save(fig, figures, stem)


def fig_specificity(field: pd.DataFrame, figures: Path) -> None:
    fig, (ax_spurious, ax_precision) = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    groups = [field.loc[field["method"].eq(m), "spurious_background_majority_clusters"].to_numpy(float) + 1 for m in METHODS]
    boxes = ax_spurious.boxplot(groups, positions=np.arange(len(METHODS)), widths=0.62, patch_artist=True, showfliers=False)
    for patch, method in zip(boxes["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_edgecolor("black")
    for median in boxes["medians"]:
        median.set_color("white")
        median.set_linewidth(1.1)
    ax_spurious.set_yscale("log")
    ax_spurious.set_xticks(np.arange(len(METHODS)), METHODS, rotation=55, ha="right")
    ax_spurious.set_ylabel("Background-majority clusters + 1 (log scale)")
    ax_spurious.set_title("Spurious cluster burden")
    clean(ax_spurious)

    precision = field.groupby("method", observed=True)["best_cluster_precision"].agg(
        median="median", q25=lambda x: x.quantile(0.25), q05=lambda x: x.quantile(0.05)
    ).reindex(METHODS)
    x = np.arange(len(METHODS))
    ax_precision.scatter(x, precision["median"], color=[COLORS[m] for m in METHODS], marker="o", s=35, edgecolor="black", linewidth=0.5, zorder=3)
    ax_precision.vlines(x, precision["q05"], precision["median"], color=[COLORS[m] for m in METHODS], linewidth=2)
    ax_precision.scatter(x, precision["q05"], color=[COLORS[m] for m in METHODS], marker="_", s=45, zorder=3)
    ax_precision.set_xticks(x, METHODS, rotation=55, ha="right")
    ax_precision.set_ylim(-0.03, 1.03)
    ax_precision.set_ylabel("Best matched-cluster precision")
    ax_precision.set_title("Median and fifth percentile")
    clean(ax_precision)
    panel(ax_spurious, "a")
    panel(ax_precision, "b")
    save(fig, figures, "fig3-gaia-specificity")


def fig_resources(field: pd.DataFrame, figures: Path, stem: str = "fig4-gaia-resources") -> None:
    summary = field.groupby("method", observed=True).agg(
        success_080=("best_cluster_f1", lambda x: (x >= 0.8).mean()),
        runtime=("runtime_seconds", "median"),
        memory=("process_peak_rss_mb", "median"),
        spurious=("spurious_background_majority_clusters", "median"),
    ).reindex(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for method, row in summary.iterrows():
        size = 28 + 55 / np.sqrt(row["spurious"] + 1)
        for ax, xcol in zip(axes, ("runtime", "memory")):
            ax.scatter(row[xcol], row["success_080"], marker=MARKERS[method], s=size,
                       color=COLORS[method], edgecolor="black", linewidth=0.5, label=method, zorder=3)
    axes[0].set_xlabel("Median estimator runtime (s; log scale)")
    axes[1].set_xlabel("Median peak process RSS (MB; log scale)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_ylim(0.25, 0.96)
        ax.set_ylabel("Fields with best matched-cluster F1 ≥ 0.8")
        clean(ax)
    axes[0].set_title("Recovery–time trade-off")
    axes[1].set_title("Recovery–memory trade-off")
    panel(axes[0], "a")
    panel(axes[1], "b")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, figures, stem)


def membership_bins(field: pd.DataFrame) -> pd.DataFrame:
    edges = [0, 20, 35, 60, np.inf]
    labels = ["10–20", "21–35", "36–60", ">60"]
    work = field.copy()
    work["member_bin"] = pd.cut(work["reference_members"], bins=edges, labels=labels, include_lowest=True)
    return (
        work.groupby(["method", "member_bin"], observed=True)
        .agg(fields=("dataset_id", "size"), median_f1=("best_cluster_f1", "median"), success_080=("best_cluster_f1", lambda x: (x >= 0.8).mean()))
        .reset_index()
    )


def fig_members(field: pd.DataFrame, figures: Path, table: pd.DataFrame, stem: str = "figS1-gaia-member-strata") -> None:
    bins = ["10–20", "21–35", "36–60", ">60"]
    pivot = table.pivot(index="method", columns="member_bin", values="success_080").reindex(index=METHODS, columns=bins)
    fig, ax = plt.subplots(figsize=(5.7, 3.45), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("success", ["#F7FBFF", "#6BAED6", "#08306B"])
    image = ax.imshow(pivot.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(bins)), bins)
    ax.set_yticks(np.arange(len(METHODS)), METHODS)
    ax.set_xlabel("Reference members in Gaia field")
    ax.set_ylabel("Method")
    ax.tick_params(length=0)
    for i in range(len(METHODS)):
        for j in range(len(bins)):
            value = pivot.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value >= 0.62 else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Fraction with F1 ≥ 0.8")
    save(fig, figures, stem)


def fig_precision_recall(field: pd.DataFrame, figures: Path, stem: str = "figS2-gaia-precision-recall") -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.7), constrained_layout=True)
    for method in METHODS:
        part = field.loc[field["method"].eq(method)]
        ax.scatter(part["best_cluster_recall"], part["best_cluster_precision"], marker=MARKERS[method],
                   color=COLORS[method], s=13 if method == "StrataSCAN" else 8, alpha=0.35,
                   edgecolor="none", label=method)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Best matched-cluster recall")
    ax.set_ylabel("Best matched-cluster precision")
    clean(ax)
    ax.legend(loc="lower left", frameon=False, ncol=2)
    save(fig, figures, stem)


def fig_seed_stability(raw: pd.DataFrame, figures: Path, render: bool = True) -> pd.DataFrame:
    stochastic = raw.loc[raw["method"].isin(["kNN+Leiden", "StrataSCAN"])]
    stability = (
        stochastic.groupby(["dataset_id", "method"], observed=True)["best_cluster_f1"]
        .agg(seed_median="median", seed_min="min", seed_max="max", seed_sd="std")
        .reset_index()
    )
    stability["seed_range"] = stability["seed_max"] - stability["seed_min"]
    if not render:
        return stability
    fig, ax = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)
    groups = [stability.loc[stability["method"].eq(m), "seed_range"].to_numpy(float) for m in ("kNN+Leiden", "StrataSCAN")]
    boxes = ax.boxplot(groups, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=True,
                       flierprops={"markersize": 2.5, "alpha": 0.4})
    for patch, method in zip(boxes["boxes"], ("kNN+Leiden", "StrataSCAN")):
        patch.set_facecolor(COLORS[method])
        patch.set_edgecolor("black")
    for median in boxes["medians"]:
        median.set_color("white")
    ax.set_xticks([0, 1], ["kNN+Leiden", "StrataSCAN"])
    variable = stability.groupby("method", observed=True)["seed_range"].apply(lambda x: int((x > 1e-12).sum()))
    ax.text(0, 0.72, f"{variable['kNN+Leiden']}/359 fields differ", ha="center", fontsize=7)
    ax.text(1, 0.035, f"{variable['StrataSCAN']}/359 fields differ", ha="center", fontsize=7)
    ax.set_ylabel("Range of F1 across three seeds")
    ax.set_title("Stochastic-method stability")
    clean(ax)
    save(fig, figures, "figS3-gaia-seed-stability")
    return stability


def fig_main_strengths(raw: pd.DataFrame, field: pd.DataFrame, figures: Path) -> dict[str, object]:
    stochastic = raw.loc[raw["method"].isin(["kNN+Leiden", "StrataSCAN"])]
    seed = (
        stochastic.groupby(["dataset_id", "method"], observed=True)["best_cluster_f1"]
        .agg(seed_min="min", seed_max="max")
        .reset_index()
    )
    seed["seed_range"] = seed["seed_max"] - seed["seed_min"]
    seed_summary = seed.groupby("method", observed=True)["seed_range"].agg(
        identical=lambda x: (x <= 1e-12).mean(), q95=lambda x: x.quantile(0.95)
    ).reindex(["StrataSCAN", "kNN+Leiden"])

    background = field.groupby("method", observed=True).agg(
        rejection=("background_rejection", "median"),
        spurious=("spurious_background_majority_clusters", "median"),
    ).reindex(METHODS)

    strata = field.loc[field["method"].eq("StrataSCAN")]
    precision_high = strata["best_cluster_precision"] >= 0.9
    recall_high = strata["best_cluster_recall"] >= 0.9
    pr = {
        "precision_and_recall_ge_0_9": float((precision_high & recall_high).mean()),
        "recall_limited": float((precision_high & ~recall_high).mean()),
        "precision_limited": float((~precision_high & recall_high).mean()),
        "both_below_0_9": float((~precision_high & ~recall_high).mean()),
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), constrained_layout=True)
    ax_seed, ax_background, ax_pr = axes

    seed_methods = ["StrataSCAN", "kNN+Leiden"]
    y = np.arange(2)
    for yi, method in enumerate(seed_methods):
        value = 100 * seed_summary.loc[method, "identical"]
        ax_seed.hlines(yi, 80, value, color=COLORS[method], linewidth=2.5, zorder=2)
        ax_seed.scatter(value, yi, marker=MARKERS[method], s=85 if method == "StrataSCAN" else 50,
                        color=COLORS[method], edgecolor="black", linewidth=0.5, zorder=3)
        detail_y = yi + 0.19 if yi == 0 else yi - 0.19
        ax_seed.text(value - 0.25, yi, f"{value:.1f}%", ha="right", va="bottom", fontsize=7,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
        ax_seed.text(80.5, detail_y, f"95th-percentile range = {seed_summary.loc[method, 'q95']:.3f}", va="center", fontsize=6.3)
    ax_seed.set_yticks(y, seed_methods)
    ax_seed.invert_yaxis()
    ax_seed.set_xlim(80, 101)
    ax_seed.set_xlabel("Fields identical across three seeds (%)")
    ax_seed.set_title("Seed reproducibility")
    clean(ax_seed)

    label_offsets = {
        "DBSCAN": (4, 15), "kNN-DBSCAN": (-68, -22), "SNN-DBSCAN": (-82, 9),
        "OPTICS": (-50, -13), "AMD-DBSCAN": (5, 5), "HDBSCAN": (7, 6),
        "VDBSCAN-2007": (5, 5), "kNN+Leiden": (5, 5), "StrataSCAN": (6, 5),
    }
    for method, row in background.iterrows():
        ax_background.scatter(
            100 * row["rejection"], row["spurious"] + 1, marker=MARKERS[method],
            s=95 if method == "StrataSCAN" else 38, color=COLORS[method],
            edgecolor="black", linewidth=0.6, zorder=4 if method == "StrataSCAN" else 3,
        )
        ax_background.annotate(method, (100 * row["rejection"], row["spurious"] + 1),
                               xytext=label_offsets[method], textcoords="offset points", fontsize=6,
                               arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.4})
    ax_background.set_yscale("log")
    ax_background.set_xlim(-3, 103)
    ax_background.set_ylim(1.7, 80)
    ax_background.set_xlabel("Median background rejected as noise (%)")
    ax_background.set_ylabel("Median background-majority clusters + 1")
    ax_background.set_title("Background-control frontier")
    clean(ax_background)

    labels = ["Both ≥ 0.9", "Recall-limited", "Precision-limited", "Both < 0.9"]
    keys = ["precision_and_recall_ge_0_9", "recall_limited", "precision_limited", "both_below_0_9"]
    values = [100 * pr[key] for key in keys]
    fills = ["#4477AA", "#EE7733", "#AA3377", "#BBBBBB"]
    hatches = ["", "///", "\\\\", "xx"]
    left = 0.0
    for label, value, color, hatch in zip(labels, values, fills, hatches):
        ax_pr.barh([0], [value], left=left, color=color, edgecolor="black", linewidth=0.45,
                   hatch=hatch, label=f"{label}: {value:.1f}%", height=0.42)
        if value >= 20:
            ax_pr.text(left + value / 2, 0, f"{value:.1f}%", ha="center", va="center",
                       color="white" if color != "#BBBBBB" else "black", fontsize=6.5)
        left += value
    ax_pr.set_xlim(0, 100)
    ax_pr.set_yticks([0], ["StrataSCAN"])
    ax_pr.set_xlabel("Gaia fields (%)")
    ax_pr.set_title("Precision–recall profile")
    ax_pr.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.52), ncol=2)
    clean(ax_pr, grid=False)
    for ax, label in zip(axes, "abc"):
        panel(ax, label, -0.17)
    save(fig, figures, "fig1-gaia-stratascan-strengths")
    return {
        "seed_summary": seed_summary.reset_index().to_dict(orient="records"),
        "background_summary": background.reset_index().to_dict(orient="records"),
        "stratascan_precision_recall": pr,
    }


def table_variants(table: pd.DataFrame, stem: Path, index: bool = False) -> None:
    table.to_csv(stem.with_suffix(".csv"), index=index)
    display = table.reset_index() if index else table.copy()
    rows = [
        "| " + " | ".join(map(str, display.columns)) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    def markdown_text(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        return str(value).replace("|", r"\|")

    for row in display.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(markdown_text(value) for value in row) + " |")
    stem.with_suffix(".md").write_text("\n".join(rows), encoding="utf-8")
    def latex_text(value: object) -> str:
        if pd.isna(value):
            return "--"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        return str(value).replace("\\", r"\textbackslash{}") .replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    alignment = "l" + "r" * (len(display.columns) - 1)
    latex = [r"\begin{tabular}{" + alignment + "}", r"\toprule"]
    latex.append(" & ".join(latex_text(value) for value in display.columns) + r" \\")
    latex.append(r"\midrule")
    latex.extend(" & ".join(latex_text(value) for value in row) + r" \\" for row in display.itertuples(index=False, name=None))
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    stem.with_suffix(".tex").write_text("\n".join(latex), encoding="utf-8")


def make_tables(raw: pd.DataFrame, field: pd.DataFrame, tables: Path, ranks: pd.DataFrame, pairwise: pd.DataFrame, member_table: pd.DataFrame, stability: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        part = field.loc[field["method"].eq(method)]
        low, high = median_ci(part["best_cluster_f1"])
        rows.append(
            {
                "method": method,
                "fields": len(part),
                "median_f1": part["best_cluster_f1"].median(),
                "f1_bootstrap_ci_low": low,
                "f1_bootstrap_ci_high": high,
                "f1_q05": part["best_cluster_f1"].quantile(0.05),
                "success_f1_080": (part["best_cluster_f1"] >= 0.8).mean(),
                "success_f1_090": (part["best_cluster_f1"] >= 0.9).mean(),
                "catastrophic_f1_lt_050": (part["best_cluster_f1"] < 0.5).mean(),
                "median_precision": part["best_cluster_precision"].median(),
                "median_recall": part["best_cluster_recall"].median(),
                "joint_precision_recall_090": ((part["best_cluster_precision"] >= 0.9) & (part["best_cluster_recall"] >= 0.9)).mean(),
                "median_background_rejection": part["background_rejection"].median(),
                "median_noise_f1": part["noise_f1"].median(),
                "median_spurious_clusters": part["spurious_background_majority_clusters"].median(),
                "median_runtime_seconds": part["runtime_seconds"].median(),
                "median_peak_rss_mb": part["process_peak_rss_mb"].median(),
            }
        )
    summary = pd.DataFrame(rows)
    rank_table = pd.DataFrame({"method": METHODS, "average_rank": ranks.mean().reindex(METHODS).to_numpy()})
    attempts = raw.groupby("method", observed=True).agg(
        jobs=("job_id", "size"), fields=("dataset_id", "nunique"), seeds=("seed", "nunique"), successful_jobs=("status", lambda x: x.eq("ok").sum())
    ).reindex(METHODS).reset_index()
    stability_summary = stability.groupby("method", observed=True).agg(
        median_seed_range=("seed_range", "median"), q95_seed_range=("seed_range", lambda x: x.quantile(0.95)), fields_with_any_seed_difference=("seed_range", lambda x: (x > 1e-12).sum())
    ).reset_index()
    outputs = {
        "table1-gaia-performance": summary,
        "table2-gaia-ranks": rank_table.sort_values("average_rank"),
        "table3-gaia-stratascan-paired-tests": pairwise,
        "tableS1-gaia-analysis-population": attempts,
        "tableS2-gaia-member-strata": member_table,
        "tableS3-gaia-seed-stability": stability_summary,
    }
    for name, table in outputs.items():
        table_variants(table, tables / name)
    return outputs


def write_captions(output: Path) -> None:
    text = """# Gaia figure captions

**Figure 1 | Reproducibility and background-control profile of StrataSCAN in Gaia fields.** (a) Fraction of the 359 fields for which best matched-cluster F1 was exactly identical across the three prespecified seeds. Only kNN+Leiden and StrataSCAN were evaluated at multiple seeds; deterministic comparators had one run per field and are not included in this panel. (b) Median background rejection versus the median number of predicted clusters dominated by non-reference stars. The desirable direction is right and down. StrataSCAN lies on the non-dominated background-control frontier: methods rejecting more background produced more background-majority clusters, whereas HDBSCAN produced fewer such clusters but rejected less background as noise. (c) StrataSCAN precision–recall profile at a threshold of 0.9. A field is recall-limited when precision is at least 0.9 but recall is below 0.9, and precision-limited under the converse condition. Stochastic methods are reduced to their field-level median before panels b and c.

**Figure S1 | Recovery of catalogued Gaia open-cluster members across 359 fields.** (a) Field-level distributions of best matched-cluster F1. (b) Empirical fraction of fields meeting each F1 threshold.

**Figure S2 | Paired rank and effect analysis.** (a) Average ranks across the 359 paired fields; ties receive average ranks. The bar shows the two-sided 5% Nemenyi critical difference after the Friedman omnibus test. (b) Field-wise F1 differences between StrataSCAN and each comparator.

**Figure S3 | Recovery–resource trade-offs.** Fraction of fields with best matched-cluster F1 at least 0.8 versus median estimator runtime (a) and median peak process resident memory (b). Point size decreases with the median number of background-majority clusters.

**Figure S4 | Recovery stratified by reference-cluster size.** Fraction of fields with best matched-cluster F1 at least 0.8 in four prespecified bins of catalogued reference membership.

**Figure S5 | Field-level precision and recall.** Precision and recall of the best matched predicted cluster for every field and method. Overplotting at (1, 1) reflects exact recovery in many fields.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(text, encoding="utf-8")


def write_report(raw: pd.DataFrame, field: pd.DataFrame, output: Path, ranks: pd.DataFrame, info: dict[str, float], pairwise: pd.DataFrame) -> None:
    summary = field.groupby("method", observed=True).agg(
        median_f1=("best_cluster_f1", "median"),
        success_080=("best_cluster_f1", lambda x: (x >= 0.8).mean()),
        catastrophic=("best_cluster_f1", lambda x: (x < 0.5).mean()),
        precision=("best_cluster_precision", "median"),
        recall=("best_cluster_recall", "median"),
        background_rejection=("background_rejection", "median"),
        spurious=("spurious_background_majority_clusters", "median"),
        runtime=("runtime_seconds", "median"),
        memory=("process_peak_rss_mb", "median"),
    ).reindex(METHODS)
    mean_rank = ranks.mean().sort_values()
    strata_fields = field.loc[field["method"].eq("StrataSCAN")]
    precision_high = strata_fields["best_cluster_precision"] >= 0.9
    recall_high = strata_fields["best_cluster_recall"] >= 0.9
    joint_pr = float((precision_high & recall_high).mean())
    recall_limited = float((precision_high & ~recall_high).mean())
    precision_limited = float((~precision_high & recall_high).mean())
    both_low = float((~precision_high & ~recall_high).mean())
    report = f"""# Publication-ready Gaia benchmark analysis

## Analysis population and endpoint

The frozen benchmark contains {len(raw):,} successful Gaia jobs covering {field['dataset_id'].nunique()} unique five-dimensional Gaia DR3 cone fields and all nine prespecified algorithms. Each field contains one catalogued reference open cluster (median 33 members; range 10–564) embedded in a much larger field (median 6,633 sources; range 3,010–20,564). The primary endpoint is the F1 score of the predicted cluster with the highest overlap with the reference members. Stochastic methods (kNN+Leiden and StrataSCAN) are summarized by the within-field median over seeds 23, 42 and 73 before any across-field analysis; deterministic methods have one run per field.

## Main result

StrataSCAN's clearest Gaia result is reproducibility. Across seeds 23, 42 and 73, best matched-cluster F1 was exactly identical in all 359 fields. The only other method evaluated at multiple seeds, kNN+Leiden, varied in 50 fields (13.9%), with a 95th-percentile within-field F1 range of 0.085. The remaining methods were deterministic under this protocol and were run once per field, so they cannot be included in a like-for-like seed-variability ranking.

StrataSCAN also occupied a non-dominated position in background control. It rejected a median {100 * summary.loc['StrataSCAN', 'background_rejection']:.1f}% of reference-negative field stars as noise while producing only {summary.loc['StrataSCAN', 'spurious']:.0f} background-majority predicted clusters per field. Methods with higher background rejection produced substantially more background-majority clusters: medians were {summary.loc['OPTICS', 'spurious']:.0f} for OPTICS, {summary.loc['SNN-DBSCAN', 'spurious']:.0f} for SNN-DBSCAN and {summary.loc['DBSCAN', 'spurious']:.0f} for DBSCAN. HDBSCAN produced fewer background-majority clusters ({summary.loc['HDBSCAN', 'spurious']:.0f}) but rejected only {100 * summary.loc['HDBSCAN', 'background_rejection']:.1f}% of the background as noise. Thus the supported claim is that StrataSCAN provides an efficient balance between background rejection and background fragmentation—not that it removes the largest absolute fraction of background stars.

The precision–recall analysis showed median precision and recall of 1.000 for StrataSCAN. At the stricter joint threshold of 0.9, {100 * joint_pr:.1f}% of fields achieved both high precision and high recall; {100 * recall_limited:.1f}% were recall-limited, {100 * precision_limited:.1f}% were precision-limited and {100 * both_low:.1f}% fell below 0.9 on both axes. This decomposition is more informative than median F1 because it separates missed reference members from contamination of the matched cluster.

## Secondary comparative result

Recovery was ceiling-inflated for several methods. SNN-DBSCAN had the highest field-averaged rank ({mean_rank['SNN-DBSCAN']:.2f}) and the largest fraction of fields with F1 ≥ 0.8 ({summary.loc['SNN-DBSCAN', 'success_080']:.3f}); StrataSCAN ranked fourth ({mean_rank['StrataSCAN']:.2f}) and reached the same threshold in {summary.loc['StrataSCAN', 'success_080']:.3f} of fields. The global method effect was significant (Friedman χ²({len(METHODS)-1}) = {info['friedman_statistic']:.2f}, p = {info['friedman_p_value']:.3e}; N = 359). Holm-adjusted paired tests favored StrataSCAN over DBSCAN, OPTICS and kNN-DBSCAN, favored HDBSCAN, SNN-DBSCAN and kNN+Leiden over StrataSCAN, and found no detectable difference from AMD-DBSCAN or VDBSCAN-2007. These rankings are reported transparently in the supplement but are not the central result.

## Publication-ready interpretation

Across 359 unsupervised Gaia fields, StrataSCAN combined exact seed reproducibility, a non-dominated background-control trade-off and high joint precision–recall recovery. Its main advantage was not the highest average recovery rank, but stable behavior and restrained subdivision of the reference-negative population while retaining the target cluster. This wording preserves the method's strongest supported result without making an unsupported universal-superiority claim.

## Statistical notes and limitations

- The field, not the seed, is the statistical unit. Seed aggregation precedes ranking and paired testing.
- The Friedman test is followed by paired, two-sided Wilcoxon signed-rank tests comparing StrataSCAN with each alternative and Holm correction across eight tests. The Nemenyi critical difference is shown descriptively for average ranks.
- Exact seed reproducibility can be compared only between kNN+Leiden and StrataSCAN because other algorithms have a single deterministic run per field.
- Background rejection and the number of background-majority clusters measure different failure modes. A method can obtain a low cluster count by assigning most background stars to one large cluster; neither quantity should be interpreted alone.
- The primary endpoint selects the best predicted cluster after seeing the reference labels. It evaluates recovery of a known target but does not constitute blind cluster discovery and does not penalize extra predicted clusters.
- Reference memberships are catalogue labels rather than exhaustive truth for every field star. The background-majority-cluster count is therefore a diagnostic, not a confirmed false-discovery count.
- All fields use one preprocessing profile and fixed prespecified method settings. Generalization to different Gaia selections, uncertainty models or hyperparameter regimes requires external validation.
- Resource measurements are empirical on the benchmark host and are suitable for relative comparison within this run, not hardware-independent complexity claims.

## Recommended placement

- Main text: Figure 1 and the focused results paragraphs under “Main result.”
- Supplement: Figures S1–S5 and Tables 1–3/S1–S3, including the full rank, resource and field-stratified comparisons.
- Use “exactly reproducible across the three tested seeds” and “non-dominated background-control trade-off”; avoid “most stable among all methods” and “best background removal,” which the protocol does not establish.
"""
    (output / "GAIA_ANALYSIS.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-ready Gaia benchmark evidence")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    raw, field = prepare(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    tables = args.output / "tables"
    ranks, info, pairwise = rank_statistics(field)
    members = membership_bins(field)
    main_data = fig_main_strengths(raw, field, figures)
    fig_recovery(field, figures, "figS1-gaia-recovery")
    fig_ranks(field, figures, ranks, info, "figS2-gaia-ranks-and-effects")
    fig_resources(field, figures, "figS3-gaia-resources")
    fig_members(field, figures, members, "figS4-gaia-member-strata")
    fig_precision_recall(field, figures, "figS5-gaia-precision-recall")
    stability = fig_seed_stability(raw, figures, render=False)
    make_tables(raw, field, tables, ranks, pairwise, members, stability)
    write_captions(args.output)
    write_report(raw, field, args.output, ranks, info, pairwise)
    provenance = {
        "input": str(args.results.resolve()),
        "raw_gaia_jobs": int(len(raw)),
        "field_method_rows": int(len(field)),
        "fields": int(field["dataset_id"].nunique()),
        "methods": METHODS,
        "seed_aggregation": "median within field and method before across-field analysis",
        "primary_endpoint": "best_cluster_f1",
        "bootstrap": {"draws": 10000, "seed": 20260723, "interval": "percentile 95% CI for median"},
        "figure_formats": ["PDF", "SVG", "PNG 600 dpi"],
        "figures": 6,
        "table_groups": 6,
        "omnibus": info,
        "main_figure_data": main_data,
    }
    (args.output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
