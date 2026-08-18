from __future__ import annotations

"""Synchronize target-discovery-v1 statistics into keyed LaTeX blocks."""

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "tables/final_statistical_summary.json"
DEFAULT_MANUSCRIPT = ROOT / "manuscript.tex"
BLOCK_KEYS = (
    "abstract", "synthetic-contrasts", "scaling-envelope",
    "noise-sensitivity", "biological", "gaia", "discussion", "conclusion",
)
HEADER_PATTERN = re.compile(r"^% FINAL-STATS: ([a-z0-9-]+)$")
EXCLUDED_RESULT_METHODS = {"AMD-DBSCAN"}


class SynchronizationError(ValueError):
    pass


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SynchronizationError(f"required mapping is missing or invalid: {path}")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SynchronizationError(f"required number is missing or invalid: {path}")
    return float(value)


def _integer(value: Any, path: str) -> int:
    number = _number(value, path)
    if not number.is_integer():
        raise SynchronizationError(f"required integer is invalid: {path}")
    return int(number)


def _fmt(value: Any, path: str, digits: int = 3) -> str:
    return f"{_number(value, path):.{digits}f}"


def _latex(value: Any) -> str:
    value = {"AMD-DBSCAN": "AMD-inspired"}.get(str(value), value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _contract(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = _mapping(summary.get("evaluation_contract"), "evaluation_contract")
    if contract.get("protocol_version") != "target-discovery-v1":
        raise SynchronizationError("summary does not use target-discovery-v1")
    discovery = _mapping(contract.get("discovery_rule"), "evaluation_contract.discovery_rule")
    if _number(discovery.get("purity"), "discovery purity") != .9 or _number(
        discovery.get("coverage"), "discovery coverage"
    ) != .1:
        raise SynchronizationError("summary discovery thresholds are not locked at 0.90/0.10")
    return contract


def _comparisons(summary: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    synthetic = _mapping(summary.get("synthetic"), "synthetic")
    block = _mapping(synthetic.get(stage), f"synthetic.{stage}")
    comparisons = _mapping(block.get("comparisons"), f"synthetic.{stage}.comparisons")
    return {
        method: value
        for method, value in comparisons.items()
        if method not in EXCLUDED_RESULT_METHODS
    }


def _comparison_extremes(summary: Mapping[str, Any], stage: str) -> tuple[tuple[float, str], tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for baseline, value in _comparisons(summary, stage).items():
        item = _mapping(value, f"{stage}.{baseline}")
        effect = _mapping(item.get("overall_delta_joint_success"), f"{stage}.{baseline}.effect")
        rows.append((_number(effect.get("mean"), f"{stage}.{baseline}.mean"), str(baseline)))
        _integer(item.get("joint_successful_pairs"), f"{stage}.{baseline}.joint")
        _integer(item.get("declared_pairs"), f"{stage}.{baseline}.declared")
    if not rows:
        raise SynchronizationError(f"no comparisons for {stage}")
    return min(rows), max(rows)


def _render_abstract(summary: Mapping[str, Any]) -> str:
    _contract(summary)
    low, high = _comparison_extremes(summary, "fresh")
    scaling = _mapping(summary.get("scaling"), "scaling")
    complete = _integer(scaling.get("successful_cells"), "scaling.success")
    total = _integer(scaling.get("prespecified_cells"), "scaling.total")
    return (
        "On the fresh-seed block, joint-success StrataSCAN-minus-baseline mean one-to-one "
        f"target-F1 effects ranged from {low[0]:+.3f} ({_latex(low[1])}) to "
        f"{high[0]:+.3f} ({_latex(high[1])}). The separate StrataSCAN-only execution "
        f"envelope completed {complete}/{total} prespecified cells."
    )


def _render_synthetic(summary: Mapping[str, Any]) -> str:
    entries: list[str] = []
    for stage, label in (("development", "development"), ("fresh", "fresh-seed")):
        low, high = _comparison_extremes(summary, stage)
        comparisons = _comparisons(summary, stage)
        denominator_groups: dict[tuple[int, int], list[str]] = {}
        for baseline, value in comparisons.items():
            item = _mapping(value, "comparison")
            denominator = (
                _integer(item.get("joint_successful_pairs"), "joint"),
                _integer(item.get("declared_pairs"), "declared"),
            )
            denominator_groups.setdefault(denominator, []).append(str(baseline))
        denom = "; ".join(
            f"{joint}/{declared} for {', '.join(_latex(method) for method in sorted(methods))}"
            for (joint, declared), methods in sorted(denominator_groups.items())
        )
        entries.append(
            f"{label.capitalize()} effects ranged from {low[0]:+.3f} ({_latex(low[1])}) "
            f"to {high[0]:+.3f} ({_latex(high[1])}). Joint-success coverage was {denom}."
        )
    return (
        "Under Hungarian one-to-one matching, "
        + " ".join(entries)
        + " Failed pairs remain in completion counts but do not receive fabricated quality scores."
    )


def _render_scaling(summary: Mapping[str, Any]) -> str:
    scaling = _mapping(summary.get("scaling"), "scaling")
    success = _integer(scaling.get("successful_cells"), "scaling.success")
    total = _integer(scaling.get("prespecified_cells"), "scaling.total")
    endpoint = _mapping(scaling.get("endpoint_resources_successful_only"), "scaling.endpoint")
    runtime = _mapping(endpoint.get("runtime_seconds"), "scaling.endpoint.runtime")
    rss = _mapping(endpoint.get("process_peak_rss_mb"), "scaling.endpoint.rss")
    size = _integer(scaling.get("endpoint_size"), "scaling.endpoint_size")
    return (
        f"StrataSCAN completed {success}/{total} cells. At {size:,} observations, successful-family "
        f"runtime ranged from {_fmt(runtime.get('min'), 'runtime min', 0)} to "
        f"{_fmt(runtime.get('max'), 'runtime max', 0)}~s and per-process RSS from "
        f"{_fmt(rss.get('min'), 'rss min', 0)} to {_fmt(rss.get('max'), 'rss max', 0)}~MiB."
    )


def _render_noise(summary: Mapping[str, Any]) -> str:
    noise = _mapping(summary.get("noise_factorial"), "noise_factorial")
    trajectories = _mapping(noise.get("trajectories"), "noise.trajectories")
    strata = _mapping(trajectories.get("StrataSCAN"), "noise.StrataSCAN")
    entries: list[str] = []
    for fraction, value in sorted(strata.items(), key=lambda item: float(item[0])):
        item = _mapping(value, f"noise.{fraction}")
        target = _mapping(item.get("macro_target_f1"), f"noise.{fraction}.target")
        discovery = _mapping(item.get("target_discovery_rate"), f"noise.{fraction}.discovery")
        true_noise = _mapping(item.get("noise_evidence_f1"), f"noise.{fraction}.true_noise")
        percent = float(fraction) * 100
        percent_text = f"{percent:.0f}" if abs(percent - round(percent)) < 1e-9 else f"{percent:.1f}"
        entries.append(
            f"{percent_text}\\%: {_fmt(target.get('median'), 'target')}/"
            f"{_fmt(discovery.get('median'), 'discovery')}/"
            f"{_fmt(true_noise.get('median'), 'true noise')}"
        )
    cells = _integer(noise.get("cells"), "noise.cells")
    return (
        f"The {cells}-cell controlled synthetic-noise factorial reports median one-to-one target "
        f"F1/discovery rate/true-noise F1 for released StrataSCAN as {', '.join(entries)}."
    )


def _render_biological(summary: Mapping[str, Any]) -> str:
    bio = _mapping(summary.get("biological"), "biological")
    weighted = _mapping(bio.get("equal_study_weighted"), "biological.equal_study")
    strata = _mapping(weighted.get("StrataSCAN"), "biological.StrataSCAN")
    ranking = bio.get("equal_study_ranking_by_target_f1")
    if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
        raise SynchronizationError("biological ranking is invalid")
    rank = list(ranking).index("StrataSCAN") + 1
    coverage = _mapping(_mapping(bio.get("coverage"), "biological.coverage").get("StrataSCAN"), "bio coverage")
    return (
        f"Under equal-study weighting, StrataSCAN ranked {rank}/{len(ranking)} by one-to-one target F1. "
        f"Its target F1/purity/coverage/discovery rate was "
        f"{_fmt(strata.get('macro_target_f1'), 'bio f1')}/"
        f"{_fmt(strata.get('macro_target_purity'), 'bio purity')}/"
        f"{_fmt(strata.get('macro_target_coverage'), 'bio coverage')}/"
        f"{_fmt(strata.get('target_discovery_rate'), 'bio discovery')}; background-abstention F1 "
        f"was {_fmt(strata.get('background_abstention_f1_diagnostic'), 'bio abstention')} (diagnostic). "
        f"Successful-dataset coverage was {_integer(coverage.get('successful_datasets'), 'bio success')}/"
        f"{_integer(coverage.get('declared_datasets'), 'bio declared')}."
    )


def _render_gaia(summary: Mapping[str, Any]) -> str:
    gaia = _mapping(summary.get("gaia"), "gaia")
    methods = _mapping(gaia.get("methods"), "gaia.methods")
    strata = _mapping(methods.get("StrataSCAN"), "gaia.StrataSCAN")
    target = _mapping(strata.get("target_metrics_successful_only"), "gaia target")
    burden = _mapping(strata.get("candidate_burden_successful_only"), "gaia burden")
    return (
        f"Native 0.2.4 StrataSCAN completed {_integer(strata.get('successful_fields'), 'gaia successful')}/"
        f"{_integer(strata.get('fields'), 'gaia fields')} fields and discovered "
        f"{_integer(strata.get('discovered_fields'), 'gaia discovered')}/"
        f"{_integer(strata.get('fields'), 'gaia fields')} fields at purity $\\geq0.90$ and coverage "
        f"$\\geq0.10$. Successful-field median target F1/purity/coverage was "
        f"{_fmt(_mapping(target.get('macro_target_f1'), 'gaia f1').get('median'), 'gaia f1 median')}/"
        f"{_fmt(_mapping(target.get('macro_target_purity'), 'gaia purity').get('median'), 'gaia purity median')}/"
        f"{_fmt(_mapping(target.get('macro_target_coverage'), 'gaia coverage').get('median'), 'gaia coverage median')}; "
        f"median unmatched candidate clusters was "
        f"{_fmt(_mapping(burden.get('unmatched_predicted_cluster_count'), 'gaia unmatched').get('median'), 'gaia unmatched median', 1)}."
    )


def _render_discussion(summary: Mapping[str, Any]) -> str:
    low, high = _comparison_extremes(summary, "fresh")
    return (
        f"Fresh-seed one-to-one target-F1 effects ranged from {low[0]:+.3f} to {high[0]:+.3f} "
        "across named baselines. Target recovery, discovery, candidate burden, and abstention "
        "diagnostics expose distinct trade-offs and do not support a universal ranking."
    )


def _render_conclusion(summary: Mapping[str, Any]) -> str:
    scaling = _mapping(summary.get("scaling"), "scaling")
    return (
        "The target-discovery-v1 evaluation uses one-to-one matching and retains failed executions "
        f"without fabricated quality; the separate execution envelope completed "
        f"{_integer(scaling.get('successful_cells'), 'scaling success')}/"
        f"{_integer(scaling.get('prespecified_cells'), 'scaling total')} cells."
    )


RENDERERS: Mapping[str, Callable[[Mapping[str, Any]], str]] = {
    "abstract": _render_abstract, "synthetic-contrasts": _render_synthetic,
    "scaling-envelope": _render_scaling,
    "noise-sensitivity": _render_noise, "biological": _render_biological,
    "gaia": _render_gaia, "discussion": _render_discussion, "conclusion": _render_conclusion,
}


def synchronize_text(manuscript: str, summary: Mapping[str, Any]) -> str:
    _contract(summary)
    normalized = manuscript.replace("\r\n", "\n")
    lines = normalized.splitlines(keepends=True)
    blocks: list[tuple[str, int, int]] = []
    headers: list[str] = []
    index = 0
    while index < len(lines):
        match = HEADER_PATTERN.fullmatch(lines[index].rstrip("\n"))
        if match is None:
            index += 1
            continue
        key = match.group(1)
        headers.append(key)
        begin = index + 1
        while begin < len(lines) and lines[begin].rstrip("\n") != f"% FINAL-STATS-BEGIN: {key}":
            if not lines[begin].lstrip().startswith("%"):
                break
            begin += 1
        end = begin + 1
        while end < len(lines) and lines[end].rstrip("\n") != f"% FINAL-STATS-END: {key}":
            end += 1
        blocks.append((key, begin, end))
        index = end + 1
    valid = [key for key, begin, end in blocks if begin < len(lines) and end < len(lines)]
    if len(headers) != len(set(headers)) or set(headers) != set(BLOCK_KEYS) or set(valid) != set(BLOCK_KEYS):
        raise SynchronizationError(
            f"FINAL-STATS marker mismatch: expected={sorted(BLOCK_KEYS)}, observed={sorted(valid)}"
        )
    output: list[str] = []
    cursor = 0
    for key, begin, end in blocks:
        output.extend(lines[cursor : begin + 1])
        output.append(RENDERERS[key](summary).strip() + "\n")
        output.append(lines[end])
        cursor = end + 1
    output.extend(lines[cursor:])
    return "".join(output)


def synchronize_file(summary_path: Path = DEFAULT_SUMMARY, manuscript_path: Path = DEFAULT_MANUSCRIPT, *, check: bool = False) -> bool:
    if not summary_path.is_file():
        raise FileNotFoundError(f"final statistical summary is missing: {summary_path}")
    if not manuscript_path.is_file():
        raise FileNotFoundError(f"manuscript is missing: {manuscript_path}")
    summary = _mapping(json.loads(summary_path.read_text(encoding="utf-8")), "summary")
    original = manuscript_path.read_text(encoding="utf-8")
    synchronized = synchronize_text(original, summary)
    changed = original.replace("\r\n", "\n") != synchronized
    if check and changed:
        raise SynchronizationError("manuscript statistics are out of sync")
    if changed and not check:
        manuscript_path.write_text(synchronized, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize target-discovery-v1 manuscript prose")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = synchronize_file(args.summary, args.manuscript, check=args.check)
    print("manuscript statistics synchronized" if changed else "manuscript statistics already synchronized")


if __name__ == "__main__":
    main()
