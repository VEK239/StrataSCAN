from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/published/v0.2.4/target-discovery-v1/cytometry.csv"
PROVENANCE = ROOT / "results/published/v0.2.4/target-discovery-v1/cytometry-provenance.json"
OUT = Path(__file__).resolve().parent

METHOD_ORDER = [
    "StrataSCAN",
    "HDBSCAN",
    "kNN+Leiden",
    "SNN-DBSCAN",
    "DBSCAN",
    "kNN-DBSCAN",
    "OPTICS",
    "VDBSCAN-2007",
    "AMD-DBSCAN",
]


def f(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def load() -> tuple[list[dict[str, str]], dict]:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    return rows, provenance


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, provenance = load()
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    datasets = sorted({r["dataset_id"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    statuses = Counter(r["status"] for r in rows)
    protocols = sorted({r["evaluation_protocol_version"] for r in rows if r["evaluation_protocol_version"]})
    generations = sorted({r["evaluation_generation"] for r in rows if r["evaluation_generation"]})

    checks = {
        "source_sha_matches_provenance": source_sha == provenance["sha256"],
        "row_count_117": len(rows) == 117,
        "thirteen_datasets": len(datasets) == 13,
        "nine_methods": len(methods) == 9,
        "full_factorial": len({(r["dataset_id"], r["method"]) for r in rows}) == 117,
        "no_gaia": all("gaia" not in r["dataset_id"].lower() for r in rows),
        "target_discovery_v1_only": protocols == ["target-discovery-v1"],
        "native_evaluation_only": generations == ["native_target_discovery_v1"],
        "hungarian_only": sorted({r["matching_strategy"] for r in rows if r["matching_strategy"]}) == ["hungarian"],
        "pairwise_f1_objective_only": sorted({r["matching_objective"] for r in rows if r["matching_objective"]}) == ["pairwise_f1"],
        "thresholds_frozen": sorted({r["discovery_purity_threshold"] for r in rows if r["discovery_purity_threshold"]}) == ["0.9"]
        and sorted({r["discovery_coverage_threshold"] for r in rows if r["discovery_coverage_threshold"]}) == ["0.1"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Audit failed: {checks}")

    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_dataset[row["dataset_id"]].append(row)

    summaries: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_rows = by_method[method]
        ok = [r for r in method_rows if r["status"] == "ok"]
        metric = lambda name: [f(r[name]) for r in ok if math.isfinite(f(r[name]))]
        f1 = metric("macro_target_f1")
        purity = metric("macro_target_purity")
        coverage = metric("macro_target_coverage")
        discovery = metric("target_discovery_rate")
        runtime = metric("runtime_seconds")
        summaries.append(
            {
                "method": method,
                "completed": len(ok),
                "attempted": len(method_rows),
                "completion_fraction": len(ok) / len(method_rows),
                "median_target_f1": median(f1) if f1 else math.nan,
                "q1_target_f1": quantile(f1, 0.25) if f1 else math.nan,
                "q3_target_f1": quantile(f1, 0.75) if f1 else math.nan,
                "min_target_f1": min(f1) if f1 else math.nan,
                "max_target_f1": max(f1) if f1 else math.nan,
                "median_target_purity": median(purity) if purity else math.nan,
                "median_target_coverage": median(coverage) if coverage else math.nan,
                "median_target_discovery": median(discovery) if discovery else math.nan,
                "mean_target_discovery": float(np.mean(discovery)) if discovery else math.nan,
                "median_runtime_seconds": median(runtime) if runtime else math.nan,
                "error": sum(r["status"] == "error" for r in method_rows),
                "timeout": sum(r["status"] == "timeout" for r in method_rows),
                "memory_limit": sum(r["status"] == "memory_limit" for r in method_rows),
            }
        )

    summary_fields = list(summaries[0])
    with (OUT / "rq4_method_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    # Dataset-level primary evidence. Failed executions retain blank metric cells (NA), never zero.
    dataset_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for method in METHOD_ORDER:
            row = next(r for r in by_dataset[dataset] if r["method"] == method)
            ok = row["status"] == "ok"
            dataset_rows.append(
                {
                    "dataset_id": dataset,
                    "method": method,
                    "status": row["status"],
                    "macro_target_f1": row["macro_target_f1"] if ok else "",
                    "macro_target_purity": row["macro_target_purity"] if ok else "",
                    "macro_target_coverage": row["macro_target_coverage"] if ok else "",
                    "target_discovery_rate": row["target_discovery_rate"] if ok else "",
                    "runtime_seconds": row["runtime_seconds"] if ok else "",
                    "background_abstention_f1_diagnostic": row["background_abstention_f1_diagnostic"] if ok else "",
                }
            )
    with (OUT / "rq4_by_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0]))
        writer.writeheader()
        writer.writerows(dataset_rows)

    # Paired per-dataset winners among successful methods; failure never becomes a zero score.
    winners: list[dict[str, object]] = []
    strata_deltas: list[dict[str, object]] = []
    win_counts = Counter()
    for dataset in datasets:
        ok_rows = [r for r in by_dataset[dataset] if r["status"] == "ok" and math.isfinite(f(r["macro_target_f1"]))]
        best = max(f(r["macro_target_f1"]) for r in ok_rows)
        tied = sorted(r["method"] for r in ok_rows if abs(f(r["macro_target_f1"]) - best) <= 1e-12)
        for method in tied:
            win_counts[method] += 1 / len(tied)
        strata = next(r for r in by_dataset[dataset] if r["method"] == "StrataSCAN")
        strata_f1 = f(strata["macro_target_f1"]) if strata["status"] == "ok" else math.nan
        best_other_rows = [r for r in ok_rows if r["method"] != "StrataSCAN"]
        best_other = max(f(r["macro_target_f1"]) for r in best_other_rows)
        best_other_methods = sorted(r["method"] for r in best_other_rows if abs(f(r["macro_target_f1"]) - best_other) <= 1e-12)
        winners.append({"dataset_id": dataset, "best_target_f1": best, "winning_methods": ";".join(tied)})
        strata_deltas.append(
            {
                "dataset_id": dataset,
                "stratascan_target_f1": strata_f1,
                "best_other_target_f1": best_other,
                "best_other_methods": ";".join(best_other_methods),
                "stratascan_minus_best_other": strata_f1 - best_other,
            }
        )
    with (OUT / "rq4_dataset_winners.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(winners[0]))
        writer.writeheader()
        writer.writerows(winners)
    with (OUT / "rq4_stratascan_paired_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(strata_deltas[0]))
        writer.writeheader()
        writer.writerows(strata_deltas)

    # Compact IEEE single-column figure: completed-dataset F1 distribution plus completion annotation.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)
    y = np.arange(len(METHOD_ORDER))
    for i, method in enumerate(METHOD_ORDER):
        vals = [f(r["macro_target_f1"]) for r in by_method[method] if r["status"] == "ok"]
        vals = [v for v in vals if math.isfinite(v)]
        color = "#B2182B" if method == "StrataSCAN" else "#737373"
        ax.scatter(vals, np.full(len(vals), i), s=8, color=color, alpha=0.32, linewidths=0, zorder=2)
        if vals:
            med = median(vals)
            q1, q3 = quantile(vals, 0.25), quantile(vals, 0.75)
            ax.plot([q1, q3], [i, i], color=color, lw=2.2, solid_capstyle="round", zorder=3)
            ax.scatter([med], [i], s=22, marker="D", color=color, edgecolor="white", linewidth=0.45, zorder=4)
        completed = sum(r["status"] == "ok" for r in by_method[method])
        ax.text(1.015, i, f"{completed}/13", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5, color=color)
    ax.set_yticks(y, METHOD_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Hungarian macro target F1")
    ax.grid(axis="x", color="#D9D9D9", lw=0.45, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(1.015, -0.75, "complete", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.2, color="#444444")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig_rq4_cytometry.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Compact LaTeX table; NA preserved for no successful run.
    latex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & F1 (median) & IQR & Discovery & Complete \\",
        r"\midrule",
    ]
    for s in summaries:
        name = str(s["method"]).replace("kNN+Leiden", r"kNN+Leiden")
        iqr = "NA" if not math.isfinite(float(s["q1_target_f1"])) else f"{s['q1_target_f1']:.3f}--{s['q3_target_f1']:.3f}"
        latex.append(
            f"{name} & {fmt(float(s['median_target_f1']))} & {iqr} & {fmt(float(s['mean_target_discovery']))} & {s['completed']}/13 \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "table_rq4_cytometry.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    strata_summary = next(s for s in summaries if s["method"] == "StrataSCAN")
    audit = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": source_sha,
        "provenance_sha256": provenance["sha256"],
        "checks": checks,
        "rows": len(rows),
        "datasets": datasets,
        "methods": methods,
        "status_counts": dict(statuses),
        "stratascan_summary": strata_summary,
        "fractional_dataset_win_counts": dict(win_counts),
        "background_metric_policy": "diagnostic only; excluded from primary figure and method summary",
        "missing_metric_policy": "non-ok executions are blank/NA and never encoded as zero",
    }
    (OUT / "rq4_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
