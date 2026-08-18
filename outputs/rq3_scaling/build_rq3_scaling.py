from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

csv.field_size_limit(100_000_000)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

STRATA_CSV = (
    ROOT
    / "results"
    / "runs"
    / "v0.2.4-target-discovery-v1-scaling"
    / "results.csv"
)
STRATA_VALIDATION = STRATA_CSV.with_name("validation.json")
STRATA_PROTOCOL = ROOT / "benchmarks" / "protocol.v0.2.4-target-discovery-v1-scaling.json"
BASELINE_DIR = (
    ROOT
    / "results"
    / "runs"
    / "v0.2.3-seven-family-baselines-scalability-5m-gated-v1"
)
BASELINE_JOBS = BASELINE_DIR / "jobs"
BASELINE_MANIFEST = BASELINE_DIR / "manifest.json"

SIZES = [500_000, 1_000_000, 2_000_000, 5_000_000]
METHODS = [
    "StrataSCAN",
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "AMD-DBSCAN",
    "kNN-DBSCAN",
    "kNN+Leiden",
]


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def infer_n(dataset_id: str) -> int | None:
    match = re.search(r"__n(\d+)$", dataset_id)
    return int(match.group(1)) if match else None


def family(dataset_id: str) -> str:
    return re.sub(r"__scaling__n\d+$", "", dataset_id)


def load_strata() -> list[dict]:
    with STRATA_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["n"] = infer_n(row["dataset_id"])
        row["family"] = family(row["dataset_id"])
        row["source_campaign"] = "v0.2.4-target-discovery-v1-scaling"
    return rows


def load_baselines() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(BASELINE_JOBS.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        row["n"] = infer_n(row.get("dataset_id", ""))
        row["family"] = family(row.get("dataset_id", ""))
        row["source_campaign"] = "v0.2.3-seven-family-baselines-scalability-5m-gated-v1"
        rows.append(row)
    return rows


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status_summary(rows: list[dict]) -> str:
    counts = Counter(str(row.get("status", "missing")) for row in rows)
    return ";".join(f"{key}={counts[key]}" for key in sorted(counts))


def summarize_strata(rows: list[dict]) -> list[dict]:
    summary = []
    for n in SIZES:
        group = [row for row in rows if row["n"] == n]
        ok = [row for row in group if row.get("status") == "ok"]
        runtimes = [x for row in ok if (x := as_float(row.get("runtime_seconds"))) is not None]
        rss = [x for row in ok if (x := as_float(row.get("process_peak_rss_mb"))) is not None]
        f1s = [x for row in ok if (x := as_float(row.get("macro_target_f1"))) is not None]
        discoveries = [
            x for row in ok if (x := as_float(row.get("target_discovery_rate"))) is not None
        ]
        summary.append(
            {
                "n": n,
                "attempted": len(group),
                "completed": len(ok),
                "status_summary": status_summary(group),
                "runtime_median_s": median(runtimes),
                "runtime_min_s": min(runtimes) if runtimes else None,
                "runtime_max_s": max(runtimes) if runtimes else None,
                "process_peak_rss_median_mib": median(rss),
                "process_peak_rss_max_mib": max(rss) if rss else None,
                "macro_target_f1_median": median(f1s),
                "macro_target_f1_min": min(f1s) if f1s else None,
                "macro_target_f1_max": max(f1s) if f1s else None,
                "target_discovery_median": median(discoveries),
            }
        )
    return summary


def summarize_methods(strata: list[dict], baselines: list[dict]) -> list[dict]:
    all_rows = strata + baselines
    summary = []
    for method in METHODS:
        for n in SIZES:
            group = [row for row in all_rows if row.get("method") == method and row["n"] == n]
            ok = [row for row in group if row.get("status") == "ok"]
            runtimes = [x for row in ok if (x := as_float(row.get("runtime_seconds"))) is not None]
            rss = [x for row in ok if (x := as_float(row.get("process_peak_rss_mb"))) is not None]
            f1s = [x for row in ok if (x := as_float(row.get("macro_target_f1"))) is not None]
            summary.append(
                {
                    "method": method,
                    "n": n,
                    "records": len(group),
                    "completed": len(ok),
                    "status_summary": status_summary(group) if group else "missing",
                    "runtime_median_success_s": median(runtimes),
                    "runtime_min_success_s": min(runtimes) if runtimes else None,
                    "runtime_max_success_s": max(runtimes) if runtimes else None,
                    "process_peak_rss_median_success_mib": median(rss),
                    "macro_target_f1_median_success": median(f1s),
                    "quality_record_count": len(f1s),
                    "campaign_complete_at_size": method == "StrataSCAN" or n <= 2_000_000,
                }
            )
    return summary


def make_figure(strata_summary: list[dict], strata_rows: list[dict]) -> None:
    xs = [row["n"] for row in strata_summary]
    med = [row["runtime_median_s"] for row in strata_summary]
    lo = [row["runtime_min_s"] for row in strata_summary]
    hi = [row["runtime_max_s"] for row in strata_summary]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.18))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.93)

    ax.fill_between(xs, lo, hi, color="#c8c8c8", alpha=0.7, linewidth=0, label="family range")
    ax.plot(xs, med, color="#1b4f72", marker="o", markersize=4.0, linewidth=1.7, label="median")

    for row in strata_summary:
        ax.annotate(
            f"{row['completed']}/{row['attempted']}",
            (row["n"], row["runtime_median_s"]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#1b1b1b",
        )

    failures_5m = [
        row
        for row in strata_rows
        if row["n"] == 5_000_000 and row.get("status") != "ok"
    ]
    if failures_5m:
        ax.scatter(
            [5_000_000],
            [3_600],
            marker="x",
            s=26,
            linewidths=1.1,
            color="#8b1a1a",
            zorder=5,
        )
        ax.annotate(
            f"{len(failures_5m)} memory",
            (5_000_000, 3_600),
            xytext=(-4, -11),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=6.4,
            color="#6e1212",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(430_000, 5_700_000)
    ax.set_ylim(18, 4_500)
    ax.set_xticks(SIZES, ["0.5M", "1M", "2M", "5M"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.grid(axis="y", which="major", color="#dedede", linewidth=0.5)
    ax.set_xlabel("Observations")
    ax.set_ylabel("Runtime (s)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=6.4,
        handlelength=1.7,
        borderaxespad=0.2,
    )

    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"fig_rq3_million_scale.{suffix}", bbox_inches="tight")
    fig.savefig(OUT / "fig_rq3_million_scale.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def method_cell(summary: list[dict], method: str, n: int) -> dict:
    return next(row for row in summary if row["method"] == method and row["n"] == n)


def five_million_text(cell: dict, method: str) -> str:
    if method == "StrataSCAN":
        return f"{cell['completed']}/{cell['records']}"
    if cell["completed"]:
        return f"{cell['completed']}/{cell['records']}*"
    statuses = cell["status_summary"]
    if "memory_limit" in statuses:
        return "M*"
    if "skipped_prior_failure" in statuses:
        return "P*"
    if cell["records"] == 0:
        return "--"
    return "F*"


def write_latex_table(method_summary: list[dict]) -> None:
    labels = {"VDBSCAN-2007": "VDBSCAN"}
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & 1M & 2M & 5M & 2M time (s) \\",
        r"\midrule",
    ]
    for method in METHODS:
        c1 = method_cell(method_summary, method, 1_000_000)
        c2 = method_cell(method_summary, method, 2_000_000)
        c5 = method_cell(method_summary, method, 5_000_000)
        time = c2["runtime_median_success_s"]
        time_text = "--" if time is None else f"{time:.0f}"
        name = labels.get(method, method).replace("+", r"$+$")
        if method == "StrataSCAN":
            name = r"\textbf{StrataSCAN}"
        lines.append(
            f"{name} & {c1['completed']}/{c1['records']} & "
            f"{c2['completed']}/{c2['records']} & {five_million_text(c5, method)} & {time_text} \\\\" 
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% Entries are successful/recorded jobs. At 5M, * marks a structurally incomplete",
            "% baseline campaign (usually only the imbalanced family was recorded). M=memory",
            "% limit; P=skipped after an earlier ladder failure; F=other recorded failure.",
            "% Runtime is the median over successful cells only and must be read with completion.",
        ]
    )
    (OUT / "table_rq3_scaling.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    strata = load_strata()
    baselines = load_baselines()
    strata_summary = summarize_strata(strata)
    method_summary = summarize_methods(strata, baselines)

    with STRATA_VALIDATION.open(encoding="utf-8") as handle:
        strata_validation = json.load(handle)
    with BASELINE_MANIFEST.open(encoding="utf-8") as handle:
        baseline_manifest = json.load(handle)

    assert len(strata) == 28
    assert strata_validation["complete"] is True
    assert Counter(row["status"] for row in strata) == Counter({"ok": 26, "memory_limit": 2})
    assert {row["n"] for row in strata} == set(SIZES)
    assert all(sum(row["n"] == n for row in strata) == 7 for n in SIZES)
    assert method_cell(method_summary, "StrataSCAN", 1_000_000)["completed"] == 7
    assert method_cell(method_summary, "StrataSCAN", 2_000_000)["completed"] == 7
    assert method_cell(method_summary, "StrataSCAN", 5_000_000)["completed"] == 5
    failed_5m = {
        row["family"] for row in strata if row["n"] == 5_000_000 and row["status"] != "ok"
    }
    assert failed_5m == {"overlap_8d", "rings_2d"}
    assert len(baselines) == 177
    assert baseline_manifest["expected_job_count"] == 224
    assert len(baselines) < baseline_manifest["expected_job_count"]

    clean_strata = []
    for row in strata:
        clean_strata.append(
            {
                "family": row["family"],
                "n": row["n"],
                "status": row["status"],
                "runtime_seconds": row.get("runtime_seconds"),
                "process_peak_rss_mb": row.get("process_peak_rss_mb"),
                "macro_target_f1": row.get("macro_target_f1"),
                "macro_target_purity": row.get("macro_target_purity"),
                "macro_target_coverage": row.get("macro_target_coverage"),
                "target_discovery_rate": row.get("target_discovery_rate"),
                "noise_evidence_f1": row.get("noise_evidence_f1"),
                "error_type": row.get("error_type"),
            }
        )
    write_csv(
        OUT / "stratascan_scaling_cells.csv",
        clean_strata,
        list(clean_strata[0]),
    )
    write_csv(OUT / "stratascan_by_size.csv", strata_summary, list(strata_summary[0]))
    write_csv(OUT / "scaling_by_method_size.csv", method_summary, list(method_summary[0]))

    matched = []
    for row in strata + baselines:
        if row["n"] == 5_000_000 and row["family"] == "imbalanced_16d":
            matched.append(
                {
                    "method": row.get("method"),
                    "status": row.get("status"),
                    "runtime_seconds": row.get("runtime_seconds"),
                    "process_peak_rss_mb": row.get("process_peak_rss_mb"),
                    "macro_target_f1": row.get("macro_target_f1"),
                    "target_discovery_rate": row.get("target_discovery_rate"),
                    "quality_available": as_float(row.get("macro_target_f1")) is not None,
                    "source_campaign": row.get("source_campaign"),
                }
            )
    matched.sort(key=lambda row: METHODS.index(row["method"]))
    write_csv(
        OUT / "five_million_imbalanced_matched.csv",
        matched,
        list(matched[0]),
    )

    make_figure(strata_summary, strata)
    write_latex_table(method_summary)

    provenance = []
    for path in (STRATA_CSV, STRATA_VALIDATION, STRATA_PROTOCOL, BASELINE_MANIFEST):
        provenance.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    baseline_hash = hashlib.sha256()
    for path in sorted(BASELINE_JOBS.glob("*.json")):
        baseline_hash.update(path.name.encode("utf-8"))
        baseline_hash.update(bytes.fromhex(sha256(path)))
    provenance.append(
        {
            "path": str(BASELINE_JOBS.relative_to(ROOT)).replace("\\", "/") + "/*.json",
            "bytes": sum(path.stat().st_size for path in BASELINE_JOBS.glob("*.json")),
            "sha256": baseline_hash.hexdigest(),
        }
    )
    write_csv(OUT / "source_provenance.csv", provenance, ["path", "bytes", "sha256"])

    validation = {
        "stratascan_campaign_structurally_complete": True,
        "stratascan_expected_records": 28,
        "stratascan_found_records": 28,
        "stratascan_statuses": dict(Counter(row["status"] for row in strata)),
        "stratascan_completion": {
            str(n): {
                "completed": sum(row["n"] == n and row["status"] == "ok" for row in strata),
                "attempted": sum(row["n"] == n for row in strata),
            }
            for n in SIZES
        },
        "baseline_campaign_structurally_complete": False,
        "baseline_expected_records": baseline_manifest["expected_job_count"],
        "baseline_found_records": len(baselines),
        "baseline_five_million_records_are_incomplete": True,
        "baseline_scaling_quality_metrics_available": False,
        "figure_width_inches": 3.45,
        "figure_height_inches": 2.18,
        "validated_claim": (
            "StrataSCAN completed all seven families at one and two million observations and "
            "five of seven at five million under an 8192-MiB limit; it is not universally fastest."
        ),
    }
    (OUT / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
