from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent
SOURCE = OUT / "stratascan_5m_thread_scaling_results.csv"
THREADS = [1, 4, 8, 16]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def triple(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), quantile(values, 0.25), quantile(values, 0.75)


def load() -> tuple[dict[str, dict[int, dict]], list[dict]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 28 or any(row["status"] != "ok" for row in rows):
        raise ValueError("thread study must contain 28 successful results")
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        thread_count = int(row["threads"])
        row["runtime_seconds"] = float(row["runtime_seconds"])
        row["graph_seconds"] = float(row["graph_seconds"])
        row["macro_target_f1"] = float(row["macro_target_f1"])
        grouped[row["dataset_id"]][thread_count] = row
    if len(grouped) != 7 or any(sorted(group) != THREADS for group in grouped.values()):
        raise ValueError("thread study is not a complete paired 7 x 4 matrix")

    summary: list[dict] = []
    for threads in THREADS:
        runtimes = [group[threads]["runtime_seconds"] for group in grouped.values()]
        total_speedup = [group[1]["runtime_seconds"] / group[threads]["runtime_seconds"] for group in grouped.values()]
        graph_speedup = [group[1]["graph_seconds"] / group[threads]["graph_seconds"] for group in grouped.values()]
        f1_delta = [abs(group[threads]["macro_target_f1"] - group[1]["macro_target_f1"]) for group in grouped.values()]
        runtime_median, runtime_q1, runtime_q3 = triple(runtimes)
        total_median, total_q1, total_q3 = triple(total_speedup)
        graph_median, graph_q1, graph_q3 = triple(graph_speedup)
        summary.append(
            {
                "threads": threads,
                "families": 7,
                "runtime_median_s": runtime_median,
                "runtime_q1_s": runtime_q1,
                "runtime_q3_s": runtime_q3,
                "total_speedup_median": total_median,
                "total_speedup_q1": total_q1,
                "total_speedup_q3": total_q3,
                "graph_speedup_median": graph_median,
                "graph_speedup_q1": graph_q1,
                "graph_speedup_q3": graph_q3,
                "max_abs_macro_target_f1_delta": max(f1_delta),
            }
        )
    return grouped, summary


def write_summary(summary: list[dict]) -> None:
    with (OUT / "stratascan_5m_thread_scaling_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def error(values: list[dict], prefix: str) -> list[list[float]]:
    return [
        [row[f"{prefix}_median"] - row[f"{prefix}_q1"] for row in values],
        [row[f"{prefix}_q3"] - row[f"{prefix}_median"] for row in values],
    ]


def plot(summary: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 7.3,
            "axes.labelsize": 8.0, "axes.titlesize": 8.0,
            "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
            "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.38))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.88, wspace=0.28)
    blue, orange = "#005A9C", "#D55E00"
    xs = [row["threads"] for row in summary]

    runtime = [row["runtime_median_s"] / 60 for row in summary]
    runtime_error = [
        [(row["runtime_median_s"] - row["runtime_q1_s"]) / 60 for row in summary],
        [(row["runtime_q3_s"] - row["runtime_median_s"]) / 60 for row in summary],
    ]
    axes[0].errorbar(xs, runtime, yerr=runtime_error, color=blue, marker="o", linewidth=1.8, capsize=2.5, markersize=4.2)
    axes[0].set_title("(a) Five-million-point runtime")
    axes[0].set_xlabel("CPU threads")
    axes[0].set_ylabel("Runtime (min)")
    axes[0].set_xticks(THREADS)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(THREADS, THREADS, color="#888888", linestyle=(0, (3, 2)), linewidth=0.8, label="ideal")
    axes[1].errorbar(
        xs, [row["total_speedup_median"] for row in summary],
        yerr=error(summary, "total_speedup"), color=blue, marker="o",
        linewidth=1.8, capsize=2.5, markersize=4.2, label="end-to-end",
    )
    axes[1].errorbar(
        xs, [row["graph_speedup_median"] for row in summary],
        yerr=error(summary, "graph_speedup"), color=orange, marker="s",
        linewidth=1.35, capsize=2.5, markersize=3.8, label="kNN graph",
    )
    axes[1].set_title("(b) Paired speedup over one thread")
    axes[1].set_xlabel("CPU threads")
    axes[1].set_ylabel("Speedup (×)")
    axes[1].set_xticks(THREADS)
    axes[1].set_xlim(0.5, 16.5)
    axes[1].set_ylim(0, 16.8)
    axes[1].legend(frameon=False, fontsize=6.7, loc="upper left", ncol=1)
    for axis in axes:
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"fig_rq3_stratascan_5m_thread_scaling.{suffix}", bbox_inches="tight")
    fig.savefig(OUT / "fig_rq3_stratascan_5m_thread_scaling.png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _, summary = load()
    write_summary(summary)
    plot(summary)
    print("wrote validated end-to-end and graph-stage thread-scaling figure")


if __name__ == "__main__":
    main()
