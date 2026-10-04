from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter


OUT = Path(__file__).resolve().parent
SOURCE = OUT / "runtime_distribution_source_rows.csv"
STATUS = OUT / "runtime_distribution_status.csv"
CAP_SECONDS = 7_200.0
METHODS = [
    "StrataSCAN",
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "kNN-DBSCAN",
    "kNN+Leiden",
]
SIZES = [500_000, 1_000_000, 2_000_000, 5_000_000]
STYLES = {
    "StrataSCAN": ("#005A9C", "o", "-", 2.2),
    "DBSCAN": ("#D55E00", "s", "-", 1.05),
    "HDBSCAN": ("#CC79A7", "^", "--", 1.05),
    "OPTICS": ("#7A5195", "v", ":", 1.15),
    "SNN-DBSCAN": ("#009E73", "D", "-", 1.05),
    "VDBSCAN-2007": ("#E69F00", "P", "--", 1.05),
    "kNN-DBSCAN": ("#56B4E9", "X", "-", 1.15),
    "kNN+Leiden": ("#444444", "h", "--", 1.05),
}


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load() -> tuple[list[dict], dict[tuple[str, int], dict]]:
    successful: dict[tuple[str, int], list[float]] = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            successful[(row["method"], int(row["n"]))].append(float(row["runtime_seconds"]))
    statuses: dict[tuple[str, int], dict] = {}
    with STATUS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            statuses[(row["method"], int(row["n"]))] = row

    summary: list[dict] = []
    for method in METHODS:
        for n_value in SIZES:
            success = successful[(method, n_value)]
            status = statuses.get((method, n_value), {})
            timeout_count = int(float(status.get("timeout", 0) or 0))
            values = success + [CAP_SECONDS] * timeout_count
            summary.append(
                {
                    "method": method,
                    "n": n_value,
                    "successful": len(success),
                    "timeouts_capped_at_2h": timeout_count,
                    "included": len(values),
                    "running": int(float(status.get("running", 0) or 0)),
                    "pending": int(float(status.get("pending", 0) or 0)),
                    "other_failure": int(float(status.get("other", 0) or 0)),
                    "runtime_median_capped_s": statistics.median(values) if values else None,
                    "runtime_q1_capped_s": quantile(values, 0.25),
                    "runtime_q3_capped_s": quantile(values, 0.75),
                    "runtime_min_capped_s": min(values) if values else None,
                    "runtime_max_capped_s": max(values) if values else None,
                }
            )
    return summary, statuses


def write_summary(rows: list[dict]) -> None:
    columns = list(rows[0])
    with (OUT / "runtime_distribution_capped_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.3,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.05, 3.02))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.265, top=0.965)
    by_key = {(row["method"], row["n"]): row for row in rows}
    for method in METHODS:
        color, marker, linestyle, width = STYLES[method]
        present = [by_key[(method, n_value)] for n_value in SIZES]
        present = [row for row in present if row["runtime_median_capped_s"] is not None]
        ax.plot(
            [row["n"] for row in present],
            [row["runtime_median_capped_s"] / 60 for row in present],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=width,
            markersize=4.4 if method == "StrataSCAN" else 3.7,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            alpha=1 if method == "StrataSCAN" else 0.88,
            zorder=6 if method == "StrataSCAN" else 3,
        )
        for row in present:
            ax.vlines(
                row["n"],
                row["runtime_q1_capped_s"] / 60,
                row["runtime_q3_capped_s"] / 60,
                color=color,
                linewidth=2.2 if method == "StrataSCAN" else 1.25,
                alpha=0.78,
                zorder=2,
            )
            if row["running"] or row["pending"]:
                ax.scatter(
                    [row["n"]],
                    [row["runtime_median_capped_s"] / 60],
                    marker=marker,
                    s=28 if method == "StrataSCAN" else 21,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    zorder=8,
                )

    ax.axhline(120, color="#8B1A1A", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.text(5_450_000, 126, "timeouts = 120 min", ha="right", va="bottom", color="#7A1616", fontsize=6.8)
    ax.axvline(735_000, color="#999999", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    ax.text(735_000, 0.065, "campaign\nboundary", ha="center", va="bottom", color="#666666", fontsize=6.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(455_000, 5_600_000)
    ax.set_ylim(0.055, 170)
    ax.set_xticks(SIZES, ["0.5M", "1M", "2M", "5M"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.1, 0.5, 1, 5, 10, 30, 60, 120])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", color="#E2E2E2", linewidth=0.5, zorder=0)
    ax.set_xlabel("Number of points")
    ax.set_ylabel("Runtime capped at 120 min (log scale)")
    ax.spines[["top", "right"]].set_visible(False)

    handles = []
    for method in METHODS:
        color, marker, linestyle, width = STYLES[method]
        handles.append(Line2D([0], [0], color=color, marker=marker, linestyle=linestyle, linewidth=width, markersize=4, label=method))
    handles.extend(
        [
            Line2D([0], [0], color="#555555", linewidth=2.2, label="IQR incl. capped timeouts"),
            Line2D([0], [0], color="#555555", marker="o", markerfacecolor="white", linestyle="None", label="running/pending remain"),
        ]
    )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.19), ncol=5, frameon=False, fontsize=6.45, handlelength=2.0, columnspacing=1.2, borderaxespad=0)
    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"fig_rq3_runtime_distribution_all_algorithms.{suffix}", bbox_inches="tight")
    fig.savefig(OUT / "fig_rq3_runtime_distribution_all_algorithms.png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows, _ = load()
    write_summary(rows)
    plot(rows)
    print("wrote capped-runtime summary and figure")


if __name__ == "__main__":
    main()
