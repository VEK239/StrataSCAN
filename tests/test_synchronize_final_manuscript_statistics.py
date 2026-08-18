from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "manuscript/oedm2026/scripts/synchronize_final_manuscript_statistics.py"
SPEC = importlib.util.spec_from_file_location("sync_final_stats", SCRIPT)
assert SPEC and SPEC.loader
sync = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = sync; SPEC.loader.exec_module(sync)


def distribution(value: float) -> dict[str, object]:
    return {"n": 1, "min": value, "q1": value, "median": value, "q3": value, "max": value, "mean": value}


def comparison(mean: float) -> dict[str, object]:
    return {"overall_delta_joint_success": distribution(mean), "joint_successful_pairs": 20,
            "declared_pairs": 21, "families": {}}


def summary() -> dict[str, object]:
    comparisons = {"DBSCAN": comparison(.1), "HDBSCAN": comparison(-.05)}
    comparisons["HDBSCAN"]["joint_successful_pairs"] = 19
    resources = {"successful_cells": 35, "runtime_seconds": distribution(2), "process_peak_rss_mb": distribution(100)}
    target = {name: distribution(.7) for name in ("macro_target_f1", "macro_target_purity", "macro_target_coverage", "target_discovery_rate")}
    burden = {"unmatched_predicted_cluster_count": distribution(1)}
    return {
        "schema_version": 2,
        "evaluation_contract": {"protocol_version": "target-discovery-v1", "discovery_rule": {"purity": .9, "coverage": .1}},
        "synthetic": {"development": {"comparisons": comparisons}, "fresh": {"comparisons": comparisons}},
        "scaling": {"successful_cells": 28, "prespecified_cells": 28, "endpoint_size": 5_000_000, "endpoint_resources_successful_only": resources},
        "noise_factorial": {"cells": 75, "trajectories": {"StrataSCAN": {
            ".25": {"macro_target_f1": distribution(.8), "target_discovery_rate": distribution(.7), "noise_evidence_f1": distribution(.9)},
            ".50": {"macro_target_f1": distribution(.7), "target_discovery_rate": distribution(.6), "noise_evidence_f1": distribution(.8)},
            ".975": {"macro_target_f1": distribution(.6), "target_discovery_rate": distribution(.5), "noise_evidence_f1": distribution(.7)},
        }}},
        "biological": {"equal_study_weighted": {"StrataSCAN": {
            "macro_target_f1": .7, "macro_target_purity": .8, "macro_target_coverage": .65,
            "target_discovery_rate": .6, "background_abstention_f1_diagnostic": .5,
        }}, "equal_study_ranking_by_target_f1": ["StrataSCAN", "DBSCAN"],
            "coverage": {"StrataSCAN": {"successful_datasets": 13, "declared_datasets": 13}}},
        "gaia": {"methods": {"StrataSCAN": {"discovered_fields": 200, "successful_fields": 350, "fields": 359,
            "target_metrics_successful_only": target, "candidate_burden_successful_only": burden}}},
    }


def manuscript_text() -> str:
    path = Path(__file__).resolve().parents[1] / "manuscript/oedm2026/manuscript.tex"
    return path.read_text(encoding="utf-8")


def test_all_blocks_synchronize_and_are_idempotent() -> None:
    first = sync.synchronize_text(manuscript_text(), summary())
    second = sync.synchronize_text(first, summary())
    assert first == second
    assert "joint-success" in first
    assert "background-abstention F1" in first
    assert "purity $\\geq0.90$" in first
    assert "97.5\\%" in first
    assert "975\\%" not in first
    assert "19/21 for HDBSCAN" in first
    assert "20/21 for DBSCAN" in first
    assert "Native 0.2.4 StrataSCAN completed 350/359" in first


def test_rejects_wrong_evaluator_contract() -> None:
    value = summary(); value["evaluation_contract"]["protocol_version"] = "legacy"
    with pytest.raises(sync.SynchronizationError, match="does not use target-discovery-v1"):
        sync.synchronize_text(manuscript_text(), value)


def test_rejects_wrong_discovery_thresholds() -> None:
    value = summary(); value["evaluation_contract"]["discovery_rule"]["purity"] = .8
    with pytest.raises(sync.SynchronizationError, match="not locked"):
        sync.synchronize_text(manuscript_text(), value)


def test_marker_topology_has_no_legacy_s2_or_s6_blocks() -> None:
    text = manuscript_text()
    observed = {match.group(1) for line in text.splitlines() if (match := sync.HEADER_PATTERN.fullmatch(line))}
    assert observed == set(sync.BLOCK_KEYS)
    assert "FINAL-STATS: supplement-s2" not in text
    assert "FINAL-STATS: solver" not in text


def test_amd_artifact_identifier_is_reviewer_safe() -> None:
    assert sync._latex("AMD-DBSCAN") == "AMD-inspired"
