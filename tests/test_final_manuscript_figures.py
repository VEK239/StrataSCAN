from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "manuscript/oedm2026/scripts/final_manuscript_figures.py"
SPEC = importlib.util.spec_from_file_location("final_manuscript_figures", SCRIPT)
assert SPEC and SPEC.loader
figures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = figures
SPEC.loader.exec_module(figures)


def test_failed_quality_remains_missing_and_resources_survive() -> None:
    frame = pd.DataFrame({"status": ["ok", "timeout"], "macro_target_f1": [.8, .4],
                          "target_discovery_rate": [.7, .2], "runtime_seconds": [1., 1200.]})
    result = figures.successful_quality(frame, ["macro_target_f1", "target_discovery_rate"])
    assert np.isnan(result.loc[1, "macro_target_f1"])
    assert np.isnan(result.loc[1, "target_discovery_rate"])
    assert result.loc[1, "runtime_seconds"] == 1200.


def test_joint_success_delta_does_not_impute_failed_pair() -> None:
    frame = pd.DataFrame([
        {"dataset_id": "a", "seed": 1, "method": "StrataSCAN", "status": "ok", "macro_target_f1": .8},
        {"dataset_id": "a", "seed": 1, "method": "DBSCAN", "status": "timeout", "macro_target_f1": np.nan},
        {"dataset_id": "b", "seed": 1, "method": "StrataSCAN", "status": "ok", "macro_target_f1": .7},
        {"dataset_id": "b", "seed": 1, "method": "DBSCAN", "status": "ok", "macro_target_f1": .6},
    ])
    result = figures.joint_success_delta(frame, "DBSCAN")
    assert result["joint_success"].tolist() == [False, True]
    assert np.isnan(result.loc[0, "delta"])
    assert np.isclose(result.loc[1, "delta"], .1)


def test_completion_eligibility_uses_displayed_slice() -> None:
    frame = pd.DataFrame({"method": ["DBSCAN"]*5 + ["StrataSCAN"]*5,
                          "status": ["ok"]*5 + ["ok", "timeout", "ok", "timeout", "ok"]})
    result = figures.comparison_eligibility(frame)
    assert result["DBSCAN"] == 1
    assert result["StrataSCAN"] == .6


def test_noise_fraction_parser_is_synthetic_only_naming_helper() -> None:
    values = pd.Series(["moons_noise50__quality__n20000", "moons_noise975__quality__n20000", "moons_noise99__quality__n20000"])
    assert figures.noise_rate_from_id(values).tolist() == [.5, .975, .99]


def test_incomplete_biological_matrix_fails_closed(tmp_path: Path) -> None:
    frame = pd.DataFrame({"dataset_id": ["levine"], "method": ["StrataSCAN"]})
    written, issues = figures.figure_biological_validation(frame, tmp_path)
    assert written == []
    assert issues == ["biological matrix missing 116 cells"]


def test_default_figure_evidence_root_is_v024() -> None:
    assert figures.EVIDENCE_ROOT.as_posix().endswith("results/published/v0.2.4/target-discovery-v1")
    assert any(path.name == "protocol.v0.2.4-target-discovery-v1-gaia-stratascan.json" for path in figures.CANONICAL_PROTOCOLS)
    assert figures.method_label("AMD-DBSCAN") == "AMD-inspired"
    assert len(figures.CANONICAL_DATA_FILES) == 7
    assert "fig3_execution_envelope-scaling-data.csv" in figures.CANONICAL_DATA_FILES


def test_biological_figure_represents_all_failed_method_as_na(tmp_path: Path) -> None:
    rows = []
    for dataset in figures.EvidenceDesign.canonical().biological_datasets:
        for method in figures.METHOD_ORDER:
            ok = method != "AMD-DBSCAN"
            rows.append({
                "dataset_id": dataset, "method": method,
                "status": "ok" if ok else "error",
                "macro_target_f1": .5 if ok else np.nan,
                "macro_target_purity": .6 if ok else np.nan,
                "macro_target_coverage": .5 if ok else np.nan,
                "target_discovery_rate": .2 if ok else np.nan,
                "candidate_clusters_per_target": 1.5 if ok else np.nan,
            })
    written, issues = figures.figure_biological_validation(pd.DataFrame(rows), tmp_path)
    assert issues == []
    assert {path.suffix for path in written} == {".pdf", ".svg", ".png"}
    audit = pd.read_csv(tmp_path / "fig4_biological_validation-data.csv")
    failed = audit.loc[audit["method"].eq("AMD-DBSCAN")]
    assert len(failed) == 13
    assert failed["status"].eq("error").all()
    assert failed["macro_target_f1"].isna().all()
