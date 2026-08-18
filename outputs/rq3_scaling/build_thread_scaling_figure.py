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


def summarize(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), quantile(values, 0.25), quantile(values, 0.75)


def load() -> tuple[list[dict], list[dict]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ok = [row for row in rows if row["status"] == "ok"]
    if len(ok) != 28:
        raise ValueError(f"expected 28 successful paired results, found {len(ok)}")
    by_dataset: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in ok:
        row["threads"] = int(row["threads"])
        row["runtime_seconds"] = float(row["runtime_seconds"])
        row["macro_target_f1"] = float(row["macro_target_f1"])
        by_dataset[row["dataset_id"]][row["threads"]] = row
    if len(by_dataset) != 7 or any(sorted(group) != THREADS for group in by_dataset.values()):
        raise ValueError("thread-scaling source is not a complete 7 x 4 paired matrix")
    summary: list[dict] = []
    for threads in THREADS:
        runtimes = [group[threads]["runtime_seconds"] for group in by_dataset.values()]
        speedups = [group[1]["runtime_seconds"] / group[threads]["runtime_seconds"] for group in by_dataset.values()]
        f1_deltas = [group[threads]["macro_target_f1"] - group[1]["macro_target_f1"] for group in by_dataset.values()]
        runtime_median, runtime_q1, runtime_q3 = summarize(runtimes)
        speedup_median, speedup_q1, speedup_q3 = summarize(speedups)
        summary.append(
            {
                "threads": threads,
                "families": len(runtimes),
                "runtime_median_s": runtime_median,
                "runtime_q1_s": runtime_q1,
                "runtime_q3_s": runtime_q3,
                "speedup_median": speedup_median,
                "speedup_q1": speedup_q1,
                "speedup_q3": speedup_q3,
                "parallel_efficiency_median": speedup_median / threads,
                "max_abs_macro_target_f1_delta": max(abs(value) for value in f1_deltas),
            }
        )
    return rows, summary


def write_summary(rows: list[dict]) -> None:
    with (OUT / "stratascan_5m_thread_scaling_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.3,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.88, wspace=0.28)
    color = "#005A9C"
    xs = [row["threads"] for row in summary]

    runtime = [row["runtime_median_s"] / 60 for row in summary]
    runtime_lower = [(row["runtime_median_s"] - row["runtime_q1_s"]) / 60 for row in summary]
    runtime_upper = [(row["runtime_q3_s"] - row["runtime_median_s"]) / 60 for row in summary]
    axes[0].errorbar(xs, runtime, yerr=[runtime_lower, runtime_upper], color=color, marker="o", linewidth=1.8, capsize=2.5, markersize=4.2)
    axes[0].set_title("(a) Five-million-point runtime")
    axes[0].set_xlabel("CPU threads")
    axes[0].set_ylabel("Runtime (min)")
    axes[0].set_xticks(THREADS)
    axes[0].set_ylim(bottom=0)

    speedup = [row["speedup_median"] for row in summary]
    speedup_lower = [row["speedup_median"] - row["speedup_q1"] for row in summary]
    speedup_upper = [row["speedup_q3"] - row["speedup_median"] for row in summary]
    axes[1].plot(THREADS, THREADS, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8, label="ideal")
    axes[1].errorbar(xs, speedup, yerr=[speedup_lower, speedup_upper], color=color, marker="o", linewidth=1.8, capsize=2.5, markersize=4.2, label="StrataSCAN")
    axes[1].set_title("(b) Paired speedup over one thread")
    axes[1].set_xlabel("CPU threads")
    axes[1].set_ylabel("Speedup (×)")
    axes[1].set_xticks(THREADS)
    axes[1].set_xlim(0.5, 16.5)
    axes[1].set_ylim(0, 16.8)
    axes[1].legend(frameon=False, fontsize=6.8, loc="upper left")

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
    print("wrote StrataSCAN 5M thread-scaling summary and figure")


if __name__ == "__main__":
    main()
