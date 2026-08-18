"""Audit canonical RQ1 synthetic target-discovery-v1 evidence.

This script is intentionally self-contained and read-only with respect to the
canonical evidence. It writes compact audit tables into its own output folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/published/v0.2.4/target-discovery-v1/synthetic_release.csv"
PROVENANCE = SOURCE.with_name("synthetic_release-provenance.json")
OUT = Path(__file__).resolve().parent

FRESH = {211, 223, 227}
DEV = {23, 42, 73, 101, 151}
METHODS = [
    "AMD-DBSCAN",
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "StrataSCAN",
    "VDBSCAN-2007",
    "kNN+Leiden",
    "kNN-DBSCAN",
]


def fnum(value: str) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def med(values: list[float]) -> float:
    values = [x for x in values if math.isfinite(x)]
    return median(values) if values else math.nan


def fmt(x: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(x) else f"{x:.{digits}f}"


def stage(seed: int) -> str:
    if seed in FRESH:
        return "fresh"
    if seed in DEV:
        return "development"
    raise AssertionError(f"unexpected seed {seed}")


def main() -> None:
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert source_sha == prov["sha256"], (source_sha, prov["sha256"])
    assert prov["evaluation_protocol_version"] == "target-discovery-v1"
    assert prov["matching_strategy"] == "hungarian"
    assert prov["matching_objective"] == "pairwise_f1"

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 504 == prov["rows"]
    assert Counter(r["status"] for r in rows) == Counter(prov["status_counts"])
    assert set(r["method"] for r in rows) == set(METHODS)
    assert set(r["evaluation_protocol_version"] for r in rows if r["status"] == "ok") == {
        "target-discovery-v1"
    }
    assert set(r["matching_strategy"] for r in rows if r["status"] == "ok") == {"hungarian"}

    for r in rows:
        r["seed_i"] = int(r["seed"])
        r["stage_calc"] = stage(r["seed_i"])
        assert r["stage_calc"] == r["evidence_stage"]

    case_ids = sorted({r["dataset_id"].split("__quality__", 1)[0] for r in rows})
    assert len(case_ids) == 7

    # Audit per-cell uniqueness and declared completion.
    keys = [(r["dataset_id"], r["method"], r["seed_i"]) for r in rows]
    assert len(set(keys)) == len(keys)
    for s, seeds in (("fresh", FRESH), ("development", DEV)):
        subset = [r for r in rows if r["stage_calc"] == s]
        assert len(subset) == 7 * 9 * len(seeds)

    # Stage/method summary. Failures are not imputed as quality zero here; both
    # declared completion and conditional quality are reported explicitly.
    summary_rows = []
    for s in ("fresh", "development"):
        declared = 7 * (len(FRESH) if s == "fresh" else len(DEV))
        for method in METHODS:
            group = [r for r in rows if r["stage_calc"] == s and r["method"] == method]
            ok = [r for r in group if r["status"] == "ok"]
            summary_rows.append(
                {
                    "stage": s,
                    "method": method,
                    "completed": len(ok),
                    "declared": declared,
                    "completion_fraction": len(ok) / declared,
                    "mean_macro_target_f1_successes": (
                        sum(fnum(r["macro_target_f1"]) for r in ok) / len(ok) if ok else math.nan
                    ),
                    "median_macro_target_f1_successes": med(
                        [fnum(r["macro_target_f1"]) for r in ok]
                    ),
                    "mean_target_discovery_rate_successes": (
                        sum(fnum(r["target_discovery_rate"]) for r in ok) / len(ok)
                        if ok
                        else math.nan
                    ),
                }
            )

    # Family-level fresh table: median over the three fresh seeds, with failures
    # kept as NA and completion reported. The comparator is the baseline with the
    # highest conditional median among successful fresh cells in that family.
    family_rows = []
    fresh_rows = [r for r in rows if r["stage_calc"] == "fresh"]
    for case in case_ids:
        case_rows = [r for r in fresh_rows if r["dataset_id"].startswith(case + "__quality__")]
        per_method = {}
        for method in METHODS:
            group = [r for r in case_rows if r["method"] == method]
            ok = [r for r in group if r["status"] == "ok"]
            per_method[method] = {
                "f1": med([fnum(r["macro_target_f1"]) for r in ok]),
                "discovery": med([fnum(r["target_discovery_rate"]) for r in ok]),
                "completed": len(ok),
                "declared": 3,
            }
        strata = per_method["StrataSCAN"]
        baseline, bstats = max(
            ((m, v) for m, v in per_method.items() if m != "StrataSCAN" and math.isfinite(v["f1"])),
            key=lambda item: item[1]["f1"],
        )
        rank = 1 + sum(
            1
            for m, v in per_method.items()
            if m != "StrataSCAN" and math.isfinite(v["f1"]) and v["f1"] > strata["f1"] + 1e-12
        )
        family_rows.append(
            {
                "case": case,
                "stratascan_f1": strata["f1"],
                "best_baseline": baseline,
                "best_baseline_f1": bstats["f1"],
                "delta": strata["f1"] - bstats["f1"],
                "stratascan_discovery": strata["discovery"],
                "stratascan_completion": f"{strata['completed']}/{strata['declared']}",
                "stratascan_rank": rank,
            }
        )

    # Paired fresh deltas for every baseline over jointly successful cells.
    by_key = {(r["dataset_id"], r["seed_i"], r["method"]): r for r in fresh_rows}
    paired_rows = []
    for baseline in [m for m in METHODS if m != "StrataSCAN"]:
        deltas = []
        wins = ties = losses = 0
        for case in case_ids:
            ds = f"{case}__quality__n20000"
            for seed in sorted(FRESH):
                srow = by_key[(ds, seed, "StrataSCAN")]
                brow = by_key[(ds, seed, baseline)]
                if srow["status"] != "ok" or brow["status"] != "ok":
                    continue
                delta = fnum(srow["macro_target_f1"]) - fnum(brow["macro_target_f1"])
                deltas.append(delta)
                if delta > 1e-12:
                    wins += 1
                elif delta < -1e-12:
                    losses += 1
                else:
                    ties += 1
        paired_rows.append(
            {
                "baseline": baseline,
                "joint_successful_pairs": len(deltas),
                "declared_pairs": 21,
                "median_delta": med(deltas),
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
        )

    def write_csv(path: Path, records: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv(OUT / "rq1_method_summary.csv", summary_rows)
    write_csv(OUT / "rq1_fresh_family_table.csv", family_rows)
    write_csv(OUT / "rq1_fresh_paired_deltas.csv", paired_rows)

    report = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": source_sha,
        "rows": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "case_ids": case_ids,
        "fresh_seeds": sorted(FRESH),
        "development_seeds": sorted(DEV),
        "evaluation_protocol_version": prov["evaluation_protocol_version"],
        "matching_strategy": prov["matching_strategy"],
        "matching_objective": prov["matching_objective"],
        "family_table": [
            {k: (fmt(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in family_rows
        ],
        "paired_deltas": [
            {k: (fmt(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in paired_rows
        ],
    }
    (OUT / "rq1_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
