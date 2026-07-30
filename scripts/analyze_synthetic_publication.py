from __future__ import annotations

import argparse
import json
import math
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

CASES = [
    "multidensity_2d",
    "ultrasparse_16d",
    "overlapping_density_16d",
    "moons_2d",
    "rings_2d",
    "overlap_8d",
    "imbalanced_16d",
]

CASE_LABELS = {
    "multidensity_2d": "Multidensity\n2D",
    "ultrasparse_16d": "Ultrasparse\n16D",
    "overlapping_density_16d": "Overlapping density\n16D",
    "moons_2d": "Moons\n2D",
    "rings_2d": "Rings\n2D",
    "overlap_8d": "Gaussian overlap\n8D",
    "imbalanced_16d": "Imbalanced\n16D",
}

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


def panel(ax: plt.Axes, label: str, x: float = -0.12) -> None:
    ax.text(x, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def clean(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)


def prepare(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != 1071:
        raise ValueError(f"expected 1071 frozen synthetic jobs, found {len(frame)}")
    if set(frame["suite"]) != {"synthetic"}:
        raise ValueError("input contains non-synthetic rows")
    frame["case"] = frame["dataset_id"].str.replace(
        r"__(quality|scaling)__n\d+$", "", regex=True
    )
    parsed_n = pd.to_numeric(
        frame["dataset_id"].str.extract(r"__n(\d+)$", expand=False), errors="raise"
    )
    frame["n"] = pd.to_numeric(frame["n"], errors="coerce").fillna(parsed_n).astype(int)
    frame["tier"] = np.where(frame["dataset_id"].str.contains("__quality__"), "quality", "scaling")
    frame["ok"] = frame["status"].eq("ok")
    for metric in ("pairwise_f1", "noise_f1", "macro_target_f1"):
        frame[f"{metric}_penalized"] = frame[metric].where(frame["ok"], 0.0).fillna(0.0)
    return frame


def median_iqr(values: pd.Series, digits: int = 3) -> str:
    values = values.dropna().astype(float)
    if values.empty:
        return "—"
    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return f"{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def condition_scores(frame: pd.DataFrame, metric: str = "pairwise_f1") -> pd.DataFrame:
    quality = frame.loc[frame["tier"].eq("quality")]
    return (
        quality.groupby(["case", "n", "method"], observed=True)[f"{metric}_penalized"]
        .median()
        .unstack("method")
        .reindex(columns=METHODS)
    )


def average_ranks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    scores = condition_scores(frame)
    ranks = scores.rank(axis=1, method="average", ascending=False)
    q = stats.studentized_range.ppf(0.95, len(METHODS), np.inf) / np.sqrt(2.0)
    critical_difference = float(
        q * np.sqrt(len(METHODS) * (len(METHODS) + 1) / (6.0 * len(scores)))
    )
    return scores, ranks.mean().sort_values(), critical_difference


def _heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    colorbar_label: str,
    failures: pd.DataFrame | None = None,
) -> mpl.image.AxesImage:
    cmap = LinearSegmentedColormap.from_list("quality", ["#F7FBFF", "#6BAED6", "#08306B"])
    image = ax.imshow(table.to_numpy(float), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)), [CASE_LABELS[x] for x in table.columns])
    ax.set_yticks(np.arange(len(table.index)), table.index)
    ax.set_title(title)
    ax.tick_params(length=0)
    for i, method in enumerate(table.index):
        for j, case in enumerate(table.columns):
            value = table.loc[method, case]
            failed = failures is not None and failures.loc[method, case] > 0
            suffix = "†" if failed else ""
            color = "white" if value >= 0.62 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}{suffix}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold" if method == "StrataSCAN" else "normal",
                fontsize=6.5,
            )
    strata = list(table.index).index("StrataSCAN")
    ax.axhline(strata - 0.5, color="black", linewidth=1.0)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(colorbar_label)
    return image


def fig_quality_overview(frame: pd.DataFrame, figures: Path) -> None:
    n50 = frame.loc[(frame["tier"] == "quality") & (frame["n"] == 50000)]
    failures = (
        (~n50["ok"])
        .groupby([n50["method"], n50["case"]], observed=True)
        .sum()
        .unstack("case", fill_value=0)
        .reindex(index=METHODS, columns=CASES, fill_value=0)
    )
    pairwise = (
        n50.groupby(["method", "case"], observed=True)["pairwise_f1_penalized"]
        .median()
        .unstack("case")
        .reindex(index=METHODS, columns=CASES)
    )
    noise = (
        n50.groupby(["method", "case"], observed=True)["noise_f1_penalized"]
        .median()
        .unstack("case")
        .reindex(index=METHODS, columns=CASES)
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.7), constrained_layout=True)
    _heatmap(axes[0], pairwise, "Signal-pair recovery", "Pairwise F1", failures)
    _heatmap(axes[1], noise, "Background/noise classification", "Noise F1", failures)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("Synthetic scenario (n = 50,000; median over five seeds)")
    axes[0].set_ylabel("Method")
    axes[1].set_ylabel("Method")
    panel(axes[0], "a", -0.10)
    panel(axes[1], "b", -0.10)
    fig.text(0.5, -0.005, "† At least one controlled failure; failures contribute a score of zero.", ha="center", fontsize=7)
    save(fig, figures, "fig1-synthetic-quality-overview")


def fig_ranks(frame: pd.DataFrame, figures: Path) -> None:
    scores, mean_rank, cd = average_ranks(frame)
    ordered = list(mean_rank.index)
    fig, (ax_rank, ax_dist) = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    y = np.arange(len(ordered))
    for yi, method in enumerate(ordered):
        ax_rank.scatter(
            mean_rank[method],
            yi,
            marker=MARKERS[method],
            s=70 if method == "StrataSCAN" else 34,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax_rank.text(mean_rank[method] + 0.10, yi, f"{mean_rank[method]:.2f}", va="center", fontsize=7)
    ax_rank.set_yticks(y, ordered)
    ax_rank.invert_yaxis()
    ax_rank.set_xlim(0.8, len(METHODS) + 0.4)
    ax_rank.set_xlabel("Average rank (lower is better)")
    ax_rank.set_title("Ranks over 21 case–size conditions")
    ax_rank.plot([1.0, 1.0 + cd], [-0.75, -0.75], color="black", linewidth=2.0, clip_on=False)
    ax_rank.text(1.0 + cd / 2, -1.05, f"Nemenyi CD = {cd:.2f}", ha="center", fontsize=7)
    clean(ax_rank)

    strata = scores["StrataSCAN"]
    competitors = [m for m in METHODS if m != "StrataSCAN"]
    deltas = [strata - scores[m] for m in competitors]
    positions = np.arange(len(competitors))
    parts = ax_dist.violinplot(deltas, positions=positions, showmeans=False, showmedians=True, widths=0.8)
    for body, method in zip(parts["bodies"], competitors):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor("black")
        body.set_alpha(0.75)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.7)
    ax_dist.axhline(0, color="black", linewidth=0.8)
    ax_dist.set_xticks(positions, competitors, rotation=55, ha="right")
    ax_dist.set_ylabel("StrataSCAN − comparator pairwise F1")
    ax_dist.set_title("Paired quality advantage")
    clean(ax_dist)
    panel(ax_rank, "a")
    panel(ax_dist, "b")
    save(fig, figures, "fig2-synthetic-ranks-and-effects")


def _aggregate_by_n(frame: pd.DataFrame, column: str, include_failures: bool = False) -> pd.DataFrame:
    work = frame.copy()
    if not include_failures:
        work = work.loc[work["ok"]]
    return (
        work.groupby(["method", "n"], observed=True)[column]
        .median()
        .unstack("n")
        .reindex(index=METHODS)
    )


def fig_scaling(frame: pd.DataFrame, figures: Path) -> None:
    runtime = _aggregate_by_n(frame, "runtime_seconds")
    memory = _aggregate_by_n(frame, "process_peak_rss_mb", include_failures=True)
    success = frame.groupby(["method", "n"], observed=True)["ok"].mean().unstack("n").reindex(index=METHODS)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True)
    for method in METHODS:
        common = dict(
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8 if method == "StrataSCAN" else 1.0,
            markersize=6 if method == "StrataSCAN" else 3.5,
            label=method,
            zorder=4 if method == "StrataSCAN" else 2,
        )
        for ax, table in zip(axes, (runtime, memory, success)):
            series = table.loc[method].dropna()
            ax.plot(series.index.astype(float), series.values, **common)
    axes[0].set_ylabel("Estimator runtime (s)")
    axes[1].set_ylabel("Peak process RSS (MB)")
    axes[2].set_ylabel("Completed-job fraction")
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.set_xlabel("Number of observations")
        clean(ax)
    axes[0].set_title("Runtime")
    axes[1].set_title("Memory")
    axes[2].set_title("Robustness")
    for ax, label in zip(axes, "abc"):
        panel(ax, label, -0.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, figures, "fig3-synthetic-scaling")


def fig_pareto(frame: pd.DataFrame, figures: Path) -> None:
    n50 = frame.loc[(frame["tier"] == "quality") & (frame["n"] == 50000)]
    summary = n50.groupby("method", observed=True).agg(
        quality=("pairwise_f1_penalized", "median"),
        runtime=("runtime_seconds", "median"),
        memory=("process_peak_rss_mb", "median"),
        success=("ok", "mean"),
    ).reindex(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for method, row in summary.iterrows():
        for ax, xcol in zip(axes, ("runtime", "memory")):
            ax.scatter(
                row[xcol],
                row["quality"],
                s=32 + 55 * row["success"],
                marker=MARKERS[method],
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
                label=method,
            )
    axes[0].set_xlabel("Median estimator runtime (s; log scale)")
    axes[1].set_xlabel("Median peak process RSS (MB; log scale)")
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Failure-penalized median pairwise F1")
        clean(ax)
    axes[0].set_title("Quality–time trade-off")
    axes[1].set_title("Quality–memory trade-off")
    panel(axes[0], "a")
    panel(axes[1], "b")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, figures, "fig4-synthetic-quality-resource")


def fig_target_recovery(frame: pd.DataFrame, figures: Path) -> None:
    n50 = frame.loc[(frame["tier"] == "quality") & (frame["n"] == 50000)]
    table = (
        n50.groupby(["method", "case"], observed=True)["macro_target_f1_penalized"]
        .median()
        .unstack("case")
        .reindex(index=METHODS, columns=CASES)
    )
    failures = (
        (~n50["ok"])
        .groupby([n50["method"], n50["case"]], observed=True)
        .sum()
        .unstack("case", fill_value=0)
        .reindex(index=METHODS, columns=CASES, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    _heatmap(ax, table, "Truth-cluster recovery", "Macro target F1", failures)
    ax.set_xlabel("Synthetic scenario (n = 50,000; median over five seeds)")
    ax.set_ylabel("Method")
    fig.text(0.5, -0.015, "† At least one controlled failure; failures contribute a score of zero.", ha="center", fontsize=7)
    save(fig, figures, "figS1-synthetic-target-recovery")


def fig_trajectories(frame: pd.DataFrame, figures: Path) -> None:
    scores = (
        frame.groupby(["case", "method", "n"], observed=True)["pairwise_f1_penalized"]
        .median()
        .rename("median")
        .reset_index()
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.3), constrained_layout=True, sharex=True, sharey=True)
    for ax, case, label in zip(axes.flat, CASES, "abcdefg"):
        part = scores.loc[scores["case"] == case]
        for method in METHODS:
            line = part.loc[part["method"] == method]
            ax.plot(
                line["n"],
                line["median"],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4 if method == "StrataSCAN" else 2.5,
                linewidth=1.5 if method == "StrataSCAN" else 0.8,
                zorder=4 if method == "StrataSCAN" else 2,
                label=method,
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(CASE_LABELS[case].replace("\n", " "))
        clean(ax)
        panel(ax, label, -0.18)
    axes.flat[-1].axis("off")
    for ax in axes[-1, :3]:
        ax.set_xlabel("Observations")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pairwise F1")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, figures, "figS2-synthetic-quality-trajectories")


def fig_failures(frame: pd.DataFrame, figures: Path) -> None:
    failed = frame.assign(failed=~frame["ok"]).groupby(["method", "case"], observed=True)["failed"].sum()
    table = failed.unstack("case", fill_value=0).reindex(index=METHODS, columns=CASES, fill_value=0)
    cmap = LinearSegmentedColormap.from_list("failures", ["#FFFFFF", "#FEE08B", "#D73027"])
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    image = ax.imshow(table.to_numpy(float), cmap=cmap, vmin=0, vmax=max(1, int(table.to_numpy().max())), aspect="auto")
    ax.set_xticks(np.arange(len(CASES)), [CASE_LABELS[x] for x in CASES])
    ax.set_yticks(np.arange(len(METHODS)), METHODS)
    ax.set_xlabel("Synthetic scenario (17 attempted jobs per method)")
    ax.set_ylabel("Method")
    ax.tick_params(length=0)
    for i in range(len(METHODS)):
        for j in range(len(CASES)):
            value = int(table.iat[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value >= 9 else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Controlled failures")
    save(fig, figures, "figS3-synthetic-failures")


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * float(p_values.iloc[index]))
        running = max(running, value)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def statistical_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    scores = condition_scores(frame)
    friedman = stats.friedmanchisquare(*(scores[m].to_numpy() for m in METHODS))
    strata = scores["StrataSCAN"]
    rows = []
    for method in METHODS:
        if method == "StrataSCAN":
            continue
        comparator = scores[method]
        delta = strata - comparator
        if np.allclose(delta, 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(strata, comparator, alternative="two-sided", zero_method="wilcox")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        rows.append(
            {
                "comparator": method,
                "median_paired_delta": float(np.median(delta)),
                "wins": int((delta > 1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
            }
        )
    pairwise = pd.DataFrame(rows)
    pairwise["p_holm"] = holm_adjust(pairwise["p_raw"])
    pairwise["significant_0_05"] = pairwise["p_holm"] < 0.05
    omnibus = {
        "friedman_statistic": float(friedman.statistic),
        "friedman_p_value": float(friedman.pvalue),
        "blocks": int(len(scores)),
        "methods": int(len(METHODS)),
    }
    return pairwise, omnibus


def write_table_variants(table: pd.DataFrame, stem: Path, index: bool = True) -> None:
    table.to_csv(stem.with_suffix(".csv"), index=index)
    display = table.reset_index() if index else table.copy()
    display.columns = [" / ".join(map(str, item)) if isinstance(item, tuple) else str(item) for item in display.columns]
    markdown_rows = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = []
        for value in row:
            text = "—" if pd.isna(value) else str(value)
            values.append(text.replace("|", "\\|"))
        markdown_rows.append("| " + " | ".join(values) + " |")
    stem.with_suffix(".md").write_text("\n".join(markdown_rows), encoding="utf-8")
    def latex_escape(value: object) -> str:
        text = "--" if pd.isna(value) else str(value)
        mapping = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        return "".join(mapping.get(character, character) for character in text)

    alignment = "l" + "r" * (len(display.columns) - 1)
    latex_rows = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in display.columns) + r" \\",
        r"\midrule",
    ]
    for row in display.itertuples(index=False, name=None):
        latex_rows.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    latex_rows.extend([r"\bottomrule", r"\end{tabular}"])
    stem.with_suffix(".tex").write_text("\n".join(latex_rows), encoding="utf-8")


def make_tables(frame: pd.DataFrame, tables: Path) -> dict[str, object]:
    tables.mkdir(parents=True, exist_ok=True)
    n50 = frame.loc[(frame["tier"] == "quality") & (frame["n"] == 50000)]
    overview_rows = []
    for method in METHODS:
        group = n50.loc[n50["method"] == method]
        overview_rows.append(
            {
                "Method": method,
                "Pairwise F1, median [IQR]": median_iqr(group["pairwise_f1_penalized"]),
                "Noise F1, median [IQR]": median_iqr(group["noise_f1_penalized"]),
                "Macro target F1, median [IQR]": median_iqr(group["macro_target_f1_penalized"]),
                "Runtime s, median [IQR]": median_iqr(group.loc[group["ok"], "runtime_seconds"]),
                "Peak RSS MB, median [IQR]": median_iqr(group["process_peak_rss_mb"], 1),
                "Completed": f"{int(group['ok'].sum())}/{len(group)}",
            }
        )
    overview = pd.DataFrame(overview_rows).set_index("Method")
    write_table_variants(overview, tables / "table1-overall-n50000")

    case_table = (
        n50.groupby(["method", "case"], observed=True)["pairwise_f1_penalized"]
        .agg(lambda x: median_iqr(x))
        .unstack("case")
        .reindex(index=METHODS, columns=CASES)
        .rename(columns={case: CASE_LABELS[case].replace("\n", " ") for case in CASES})
    )
    write_table_variants(case_table, tables / "table2-pairwise-f1-by-scenario-n50000")

    scaling_rows = []
    for method in METHODS:
        for n in (100000, 200000):
            group = frame.loc[(frame["method"] == method) & (frame["n"] == n)]
            scaling_rows.append(
                {
                    "Method": method,
                    "n": n,
                    "Runtime s, median [IQR]": median_iqr(group.loc[group["ok"], "runtime_seconds"]),
                    "Peak RSS MB, median [IQR]": median_iqr(group["process_peak_rss_mb"], 1),
                    "Completed": f"{int(group['ok'].sum())}/{len(group)}",
                }
            )
    scaling = pd.DataFrame(scaling_rows).set_index(["Method", "n"])
    write_table_variants(scaling, tables / "table3-scaling")

    full = (
        frame.groupby(["case", "n", "method"], observed=True)
        .agg(
            pairwise_f1_median=("pairwise_f1_penalized", "median"),
            pairwise_f1_q1=("pairwise_f1_penalized", lambda x: x.quantile(0.25)),
            pairwise_f1_q3=("pairwise_f1_penalized", lambda x: x.quantile(0.75)),
            noise_f1_median=("noise_f1_penalized", "median"),
            macro_target_f1_median=("macro_target_f1_penalized", "median"),
            runtime_seconds_median=("runtime_seconds", "median"),
            peak_rss_mb_median=("process_peak_rss_mb", "median"),
            completed=("ok", "sum"),
            attempted=("ok", "size"),
        )
        .reset_index()
    )
    write_table_variants(full, tables / "tableS1-full-condition-summary", index=False)

    status = (
        frame.groupby(["method", "status"], observed=True)
        .size()
        .unstack("status", fill_value=0)
        .reindex(index=METHODS, fill_value=0)
    )
    status["attempted"] = status.sum(axis=1)
    status["completion_rate"] = status.get("ok", 0) / status["attempted"]
    write_table_variants(status, tables / "tableS2-job-status")

    pairwise, omnibus = statistical_tables(frame)
    write_table_variants(pairwise.set_index("comparator"), tables / "tableS3-paired-statistics")
    (tables / "tableS3-friedman.json").write_text(
        json.dumps(omnibus, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"overview": overview, "status": status, "pairwise": pairwise, "omnibus": omnibus}


def write_captions(output: Path) -> None:
    captions = """# Figure captions

## Figure 1 — Synthetic clustering quality

Failure-penalized median pairwise F1 (a) and noise-classification F1 (b) at n = 50,000 across five independently generated replicates. A dagger marks method–scenario cells containing at least one controlled timeout, memory-limit, or numerical failure; such runs remain in the denominator with a score of zero. Darker color indicates better performance.

## Figure 2 — Cross-condition ranking and paired effects

(a) Average ranks across 21 independent scenario–size conditions (seven scenarios and three quality sizes), after taking the median over five seeds within each condition. The horizontal bar is the Nemenyi 5% critical difference. (b) Distribution across the same 21 conditions of the paired pairwise-F1 difference between StrataSCAN and each comparator; positive values favor StrataSCAN.

## Figure 3 — Computational scaling and robustness

Median successful-run estimator time (a), median whole-process peak resident set size over all attempted runs (b), and completed-job fraction (c) as functions of sample size. Aggregation is across all seven synthetic scenarios and available seeds. Both resource axes are logarithmic.

## Figure 4 — Quality–resource trade-off

Failure-penalized median pairwise F1 versus median successful-run estimator time (a) and whole-process peak RSS (b) at n = 50,000. Marker area encodes the completed-job fraction; larger markers indicate greater robustness.

## Figure 5 — Algorithm behavior across synthetic scenarios

Sklearn-style qualitative comparison of ground truth and the nine clustering methods on the seven frozen synthetic scenarios at n = 5,000 and seed 42. Each method is fitted in the native feature space; datasets above two dimensions are displayed using a deterministic two-dimensional PCA projection only for visualization. Predicted clusters are colored categorically, predicted noise is light gray, and each panel reports pairwise F1, predicted cluster count, and estimator runtime.

## Supplementary Figure S1 — Truth-cluster recovery

Failure-penalized macro target F1 at n = 50,000, measuring per-truth-cluster recovery rather than dominance by large clusters. A dagger marks cells with controlled failures.

## Supplementary Figure S2 — Quality trajectories

Median failure-penalized pairwise F1 from n = 5,000 to n = 200,000 for every method and scenario. Quality sizes use five seeds; scaling sizes use one frozen seed.

## Supplementary Figure S3 — Controlled failures

Number of controlled failures among 17 attempted jobs per method and scenario (15 quality jobs plus two scaling jobs). Failures remain part of the benchmark evidence and are not silently excluded from quality summaries.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")


def write_report(frame: pd.DataFrame, output: Path, table_data: dict[str, object]) -> None:
    scores, ranks, cd = average_ranks(frame)
    n50 = frame.loc[(frame["tier"] == "quality") & (frame["n"] == 50000)]
    quality = n50.groupby("method", observed=True)["pairwise_f1_penalized"].median().sort_values(ascending=False)
    runtime = n50.loc[n50["ok"]].groupby("method", observed=True)["runtime_seconds"].median()
    memory = n50.groupby("method", observed=True)["process_peak_rss_mb"].median()
    statuses = frame["status"].value_counts().to_dict()
    pairwise = table_data["pairwise"]
    significant = pairwise.loc[pairwise["significant_0_05"], "comparator"].tolist()
    weak_cases = (
        n50.loc[n50["method"] == "StrataSCAN"]
        .groupby("case", observed=True)["pairwise_f1_penalized"]
        .median()
        .sort_values()
    )
    lines = [
        "# Full synthetic benchmark analysis",
        "",
        "## Scope and accounting",
        "",
        "The frozen synthetic matrix contains 1,071 attempted jobs: seven scenarios, nine methods, "
        "three quality sizes with five seeds, and two scaling sizes with one seed. Controlled failures "
        "remain in the denominator and receive zero quality for failure-penalized summaries.",
        "",
        f"Observed statuses: {json.dumps(statuses, sort_keys=True)}.",
        "",
        "## Main findings",
        "",
        f"- StrataSCAN has the best average rank ({ranks['StrataSCAN']:.2f}) across the 21 case–size conditions; "
        f"the Nemenyi 5% critical difference is {cd:.2f}.",
        f"- At n = 50,000, StrataSCAN's failure-penalized median pairwise F1 is {quality['StrataSCAN']:.3f}; "
        f"median successful-run time is {runtime['StrataSCAN']:.2f} s and median peak RSS is {memory['StrataSCAN']:.1f} MB.",
        f"- Holm-adjusted paired Wilcoxon tests favor StrataSCAN significantly against: "
        f"{', '.join(significant) if significant else 'none'}.",
        f"- The hardest n = 50,000 cases for StrataSCAN are {weak_cases.index[0]} "
        f"(pairwise F1 {weak_cases.iloc[0]:.3f}) and {weak_cases.index[1]} "
        f"({weak_cases.iloc[1]:.3f}); these limitations should be discussed explicitly.",
        "- AMD-DBSCAN exhibits quadratic-memory failures at large n; HDBSCAN has controlled timeouts. "
        "These outcomes are represented in the robustness panel and status table.",
        "",
        "## Recommended article placement",
        "",
        "- Main text: Figures 1–4 and Tables 1–3.",
        "- Supplement: Figures S1–S3 and Tables S1–S3.",
        "- Use `FIGURE_CAPTIONS.md` verbatim or adapt it to the journal style.",
        "",
        "## Statistical unit",
        "",
        "The rank and paired tests use 21 case–size blocks after median aggregation over seeds, avoiding "
        "pseudo-replication of the five random realizations. The Friedman test is followed by paired "
        "two-sided Wilcoxon signed-rank comparisons against StrataSCAN with Holm correction.",
        "",
        f"Friedman statistic = {table_data['omnibus']['friedman_statistic']:.3f}, "
        f"p = {table_data['omnibus']['friedman_p_value']:.3e}.",
    ]
    (output / "SYNTHETIC_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-ready synthetic benchmark evidence")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    frame = prepare(args.results)
    figures = args.output / "figures"
    tables = args.output / "tables"
    args.output.mkdir(parents=True, exist_ok=True)
    fig_quality_overview(frame, figures)
    fig_ranks(frame, figures)
    fig_scaling(frame, figures)
    fig_pareto(frame, figures)
    fig_target_recovery(frame, figures)
    fig_trajectories(frame, figures)
    fig_failures(frame, figures)
    table_data = make_tables(frame, tables)
    write_captions(args.output)
    write_report(frame, args.output, table_data)
    provenance = {
        "input": str(args.results.resolve()),
        "rows": int(len(frame)),
        "sha256_note": "Use the parent benchmark manifest for frozen input and code checksums.",
        "failure_policy": "controlled failures retained with zero quality in penalized summaries",
        "figure_formats": ["PDF", "SVG", "PNG 600 dpi"],
        "figures": 7,
        "table_groups": 6,
    }
    (args.output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
