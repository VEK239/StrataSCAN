from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
COLORS = {"StrataSCAN": "#d95f02", "kNN+Leiden": "#1b9e77"}
NEUTRAL = "#8a8f98"


def binary_metrics(row: pd.Series) -> dict[str, float | str]:
    matches = json.loads(row["target_matches_json"])
    n_signal = sum(int(match["target_size"]) for match in matches)
    n_background = int(row["n"]) - n_signal
    signal_tp = float(row["signal_coverage"]) * n_signal
    signal_fp = (1.0 - float(row["background_rejection"])) * n_background
    signal_precision = signal_tp / (signal_tp + signal_fp) if signal_tp + signal_fp else 0.0
    signal_recall = float(row["signal_coverage"])
    signal_f1 = (
        2.0 * signal_precision * signal_recall / (signal_precision + signal_recall)
        if signal_precision + signal_recall
        else 0.0
    )
    return {
        "sample_id": str(row["dataset_id"]).removeprefix("samusik_"),
        "method": str(row["method"]),
        "signal_precision": signal_precision,
        "signal_recall": signal_recall,
        "signal_f1": signal_f1,
        "background_precision": float(row["noise_precision"]),
        "background_recall": float(row["noise_recall"]),
        "background_f1": float(row["noise_f1"]),
        "binary_macro_f1": 0.5 * (signal_f1 + float(row["noise_f1"])),
    }


def load_binary_results(paths: list[Path]) -> pd.DataFrame:
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    raw = raw.loc[raw["status"].eq("ok") & raw["method"].isin(METHOD_ORDER)]
    jobs = pd.DataFrame([binary_metrics(row) for _, row in raw.iterrows()])
    metrics = [
        "signal_precision",
        "signal_recall",
        "signal_f1",
        "background_precision",
        "background_recall",
        "background_f1",
        "binary_macro_f1",
    ]
    return jobs.groupby(["sample_id", "method"], as_index=False)[metrics].median()


def distribution_panel(ax: plt.Axes, data: pd.DataFrame, value: str, title: str) -> None:
    rng = np.random.default_rng(342)
    for position, method in enumerate(METHOD_ORDER):
        values = data.loc[data["method"].eq(method), value].to_numpy(dtype=float)
        color = COLORS.get(method, NEUTRAL)
        ax.scatter(
            values,
            position + rng.uniform(-0.13, 0.13, values.size),
            s=25,
            color=color,
            alpha=0.43 if method != "StrataSCAN" else 0.68,
            edgecolors="none",
        )
        median = float(np.median(values))
        ax.plot([median, median], [position - 0.24, position + 0.24], color=color, lw=3)
        ax.scatter(
            median,
            position,
            s=55,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            zorder=4,
        )
        ax.text(min(0.985, median + 0.018), position - 0.16, f"{median:.2f}", color=color, fontsize=7.5)
    ax.set_yticks(range(len(METHOD_ORDER)), METHOD_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("F1 score")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9dde3", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot binary signal-background Samusik metrics")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/samusik_stability")
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_binary_results(args.results)
    data.to_csv(output / "binary_signal_background_by_sample.csv", index=False)

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
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.2), constrained_layout=True)
    distribution_panel(
        axes[0, 0], data, "background_f1", "a  Background/noise-class F1"
    )
    distribution_panel(axes[1, 0], data, "signal_f1", "c  Signal-class F1")
    distribution_panel(
        axes[1, 1], data, "binary_macro_f1", "d  Binary macro-F1"
    )

    summary = data.groupby("method").median(numeric_only=True).reindex(METHOD_ORDER)
    ax = axes[0, 1]
    label_offsets = {
        "StrataSCAN": (6, 6),
        "kNN+Leiden": (6, 6),
        "VDBSCAN-2007": (6, 7),
        "SNN-DBSCAN": (-60, -19),
        "kNN-DBSCAN": (-72, 8),
        "DBSCAN": (-58, -12),
        "OPTICS": (-22, 10),
    }
    for method, row in summary.iterrows():
        color = COLORS.get(method, NEUTRAL)
        ax.scatter(
            row["background_recall"],
            row["background_precision"],
            s=80 if method == "StrataSCAN" else 52,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.annotate(
            method,
            (row["background_recall"], row["background_precision"]),
            xytext=label_offsets[method],
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Background recall = rejected background fraction")
    ax.set_ylabel("Background precision = true background among predicted noise")
    ax.set_title("b  Why background F1 changes", loc="left", fontweight="bold")
    ax.grid(color="#d9dde3", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Samusik binary signal–background separation (cell types ignored)",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Points are mouse samples; vertical bars and open circles are medians. Binary macro-F1 gives equal weight to signal and background classes.",
        ha="center",
        fontsize=8.5,
        color="#4f5661",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"fig-samusik-binary-signal-background.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    caption = """**Figure | Binary signal–background separation on Samusik without cell-type matching.** Every manually annotated population is treated as the signal class and `unassigned` cells as the background class; predicted cluster identity is ignored. Points denote 10 mouse samples, and stochastic seeds are first collapsed by the within-sample median. (a) Background-class F1, where a true positive is an `unassigned` cell predicted as noise. (b) Median background precision–recall decomposition. (c) Signal-class F1, where a true positive is an annotated cell assigned to any non-noise cluster. (d) Binary macro-F1, the equal-weight mean of signal- and background-class F1. StrataSCAN is orange and kNN+Leiden is green. HDBSCAN and AMD-DBSCAN are omitted because they did not complete Samusik_01 screening.\n"""
    (output / "fig-samusik-binary-signal-background-caption.md").write_text(
        caption, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
