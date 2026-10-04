from __future__ import annotations

"""Recompute predeclared discovery thresholds from frozen target matches.

This is analysis-only: assignments, predictions, and fitted parameters are
never changed.  The locked 0.90/0.10 threshold is verified against every
successful row before any sensitivity summary is written.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_manuscript_statistics import (  # noqa: E402
    EVIDENCE_ROOT,
    EvidenceDesign,
    EvidencePaths,
    _validate_sources,
)


THRESHOLDS = ((.90, .05), (.90, .10), (.90, .25), (.95, .10))
PRIMARY_THRESHOLD = (.90, .10)
DEFAULT_CSV = ROOT / "tables/discovery_threshold_sensitivity.csv"
DEFAULT_MD = ROOT / "tables/discovery_threshold_sensitivity.md"
METHOD_DISPLAY = {"AMD-DBSCAN": "AMD-inspired"}


def parse_target_matches(value: object) -> list[tuple[float, float]]:
    """Parse purity/coverage pairs, retaining unmatched targets as zeros."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("successful row has empty target_matches_json")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("malformed target_matches_json") from error
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("target_matches_json must be a nonempty list")
    matches: list[tuple[float, float]] = []
    for index, item in enumerate(decoded):
        if not isinstance(item, Mapping):
            raise ValueError(f"target match {index} is not an object")
        unmatched = item.get("matched_predicted_cluster") is None
        purity = item.get("purity", item.get("precision"))
        coverage = item.get("coverage", item.get("recall"))
        if unmatched and (purity is None or coverage is None):
            purity, coverage = 0.0, 0.0
        if isinstance(purity, bool) or not isinstance(purity, (int, float)):
            raise ValueError(f"target match {index} has invalid purity")
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
            raise ValueError(f"target match {index} has invalid coverage")
        pair = (float(purity), float(coverage))
        if not all(np.isfinite(number) and 0 <= number <= 1 for number in pair):
            raise ValueError(f"target match {index} has an out-of-range score")
        if unmatched and not np.allclose(pair, (0.0, 0.0), atol=1e-12, rtol=0):
            raise ValueError(f"unmatched target {index} must have zero purity and coverage")
        matches.append(pair)
    return matches


def discovery_rate(matches: Sequence[tuple[float, float]], purity: float, coverage: float) -> float:
    return float(np.mean([p >= purity and c >= coverage for p, c in matches]))


def summarize_block(frame: pd.DataFrame, block: str) -> pd.DataFrame:
    required = {"method", "status", "target_matches_json", "target_discovery_rate"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{block} is missing sensitivity fields: {missing}")
    rows: list[dict[str, object]] = []
    declared = frame.groupby("method", observed=True).size()
    successful = frame.loc[frame["status"].eq("ok")].copy()
    parsed: dict[int, list[tuple[float, float]]] = {}
    for index, row in successful.iterrows():
        matches = parse_target_matches(row["target_matches_json"])
        parsed[index] = matches
        locked = discovery_rate(matches, *PRIMARY_THRESHOLD)
        recorded = float(row["target_discovery_rate"])
        if not np.isclose(locked, recorded, atol=1e-12, rtol=1e-12):
            raise ValueError(
                f"{block} locked discovery mismatch for row {index}: "
                f"recomputed={locked}, recorded={recorded}"
            )
    for method in declared.index:
        part = successful.loc[successful["method"].eq(method)]
        match_lists = [parsed[index] for index in part.index]
        target_count = sum(len(matches) for matches in match_lists)
        for purity, coverage in THRESHOLDS:
            rates = [discovery_rate(matches, purity, coverage) for matches in match_lists]
            discovered = sum(
                int(p >= purity and c >= coverage)
                for matches in match_lists for p, c in matches
            )
            rows.append({
                "evidence_block": block,
                "method": str(method),
                "purity_threshold": purity,
                "coverage_threshold": coverage,
                "is_locked_primary": (purity, coverage) == PRIMARY_THRESHOLD,
                "declared_executions": int(declared.loc[method]),
                "successful_executions": int(len(part)),
                "target_count_successful_only": int(target_count),
                "discovered_target_count": int(discovered),
                "mean_execution_discovery_rate": float(np.mean(rates)) if rates else np.nan,
                "micro_target_discovery_rate": float(discovered / target_count) if target_count else np.nan,
            })
    result = pd.DataFrame(rows)
    verify_threshold_monotonicity(result)
    return result


def verify_threshold_monotonicity(summary: pd.DataFrame) -> None:
    for (block, method), part in summary.groupby(["evidence_block", "method"], observed=True):
        rates = {
            (float(row.purity_threshold), float(row.coverage_threshold)):
                float(row.mean_execution_discovery_rate)
            for row in part.itertuples()
        }
        for loose, strict in (((.90, .05), (.90, .10)), ((.90, .10), (.90, .25)), ((.90, .10), (.95, .10))):
            if np.isfinite(rates[loose]) and np.isfinite(rates[strict]) and rates[loose] + 1e-12 < rates[strict]:
                raise ValueError(f"threshold monotonicity failed for {block}/{method}: {loose} -> {strict}")


def build_sensitivity(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    release = frames["synthetic_release"]
    blocks = {
        "synthetic_development": release.loc[release["evidence_stage"].eq("development")],
        "synthetic_fresh": release.loc[release["evidence_stage"].eq("fresh")],
        "synthetic_scaling": frames["synthetic_scaling"],
        "cytometry": frames["cytometry"],
        "gaia": frames["gaia"],
        "noise_sweep": frames["noise_factorial"],
    }
    result = pd.concat([summarize_block(frame, block) for block, frame in blocks.items()], ignore_index=True)
    result["rank_at_threshold"] = result.groupby(
        ["evidence_block", "purity_threshold", "coverage_threshold"], observed=True
    )["mean_execution_discovery_rate"].rank(method="min", ascending=False).astype("Int64")
    locked = result.loc[result["is_locked_primary"], ["evidence_block", "method", "rank_at_threshold"]].rename(
        columns={"rank_at_threshold": "locked_primary_rank"}
    )
    result = result.merge(locked, on=["evidence_block", "method"], validate="many_to_one")
    result["rank_shift_from_locked"] = result["rank_at_threshold"] - result["locked_primary_rank"]
    return result


def render_markdown(summary: pd.DataFrame) -> str:
    primary = summary.loc[summary["is_locked_primary"]].copy()
    extrema = summary.groupby(["evidence_block", "method"], observed=True)["mean_execution_discovery_rate"].agg(["min", "max"])
    lines = [
        "# Discovery-threshold sensitivity", "",
        "Assignments and target matches are frozen. The locked primary remains purity 0.90 and coverage 0.10; the other three pairs are descriptive sensitivity checks.", "",
        "| Evidence block | Method | Locked rate | Sensitivity range | Maximum absolute rank shift | Successful/declared executions |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary.sort_values(["evidence_block", "method"]).itertuples():
        limits = extrema.loc[(row.evidence_block, row.method)]
        rate = "NA" if pd.isna(row.mean_execution_discovery_rate) else f"{row.mean_execution_discovery_rate:.3f}"
        rate_range = "NA" if pd.isna(limits["min"]) else f"{limits['min']:.3f}--{limits['max']:.3f}"
        shifts = summary.loc[
            summary["evidence_block"].eq(row.evidence_block) & summary["method"].eq(row.method),
            "rank_shift_from_locked",
        ].dropna()
        max_shift = "NA" if shifts.empty else str(int(shifts.abs().max()))
        lines.append(
            f"| {row.evidence_block} | {METHOD_DISPLAY.get(row.method, row.method)} | "
            f"{rate} | {rate_range} | {max_shift} | "
            f"{row.successful_executions}/{row.declared_executions} |"
        )
    lines.extend(["", "The CSV contains all four predeclared threshold pairs and both execution-macro and target-micro rates.", ""])
    return "\n".join(lines)


def write_outputs(summary: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build discovery-threshold sensitivity tables")
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    paths = EvidencePaths.canonical(args.evidence_root)
    frames = _validate_sources(paths, EvidenceDesign.canonical())
    write_outputs(build_sensitivity(frames), args.output_csv, args.output_md)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
