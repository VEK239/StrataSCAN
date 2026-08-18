from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "manuscript/oedm2026/scripts/final_manuscript_statistics.py"
SPEC = importlib.util.spec_from_file_location("final_manuscript_statistics", SCRIPT)
assert SPEC and SPEC.loader
stats = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = stats
SPEC.loader.exec_module(stats)


def design() -> object:
    return stats.EvidenceDesign(
        methods=("DBSCAN", "StrataSCAN"), cases=("moons_2d",),
        development_sizes=(20_000,), development_seeds=(23,),
        fresh_seeds=(211,), comparison_size=20_000, scaling_sizes=(500_000,),
        biological_datasets=("levine",), noise_methods=("StrataSCAN",),
        noise_cases=("moons_noise25",), noise_seeds=(211,),
    )


def row(suite: str, dataset_id: str, method: str, seed: int = 23) -> dict[str, object]:
    value: dict[str, object] = {
        "suite": suite, "dataset_id": dataset_id, "method": method, "seed": seed,
        "status": "ok", "package_version": "0.2.4", "evaluation_protocol_version": "target-discovery-v1",
        "evaluation_generation": (
            "legacy_single_target_exact_rescore"
            if suite == "gaia" and method != "StrataSCAN"
            else "native_target_discovery_v1"
        ),
        "matching_strategy": "hungarian", "matching_objective": "pairwise_f1",
        "primary_target_aggregation": "macro", "discovery_purity_threshold": .9,
        "discovery_coverage_threshold": .1, "macro_target_f1": .7,
        "macro_target_purity": .8, "macro_target_coverage": .65,
        "target_discovery_rate": .6, "predicted_cluster_count": 3,
        "unmatched_predicted_cluster_count": 1,
        "background_majority_predicted_clusters": 1,
        "candidate_clusters_per_target": 1.5, "runtime_seconds": 2.0,
        "process_peak_rss_mb": 100.0,
    }
    if suite == "synthetic":
        value |= {"evaluation_mode": "full_partition", "background_semantics": "known_synthetic_noise",
                  "noise_evidence_f1": .75, "pairwise_f1": .68}
    else:
        mode = "full_partition" if suite == "cytometry" else "target_discovery"
        semantics = "heterogeneous_biological_background" if suite == "cytometry" else "reference_negative_background"
        value |= {"evaluation_mode": mode, "background_semantics": semantics,
                  "background_abstention_precision_diagnostic": .8,
                  "background_abstention_fraction": .7,
                  "background_abstention_f1_diagnostic": .75,
                  "predicted_abstention_fraction": .6}
    return value


def provenance(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"sha256": digest, "evaluation_protocol_version": "target-discovery-v1",
               "matching_strategy": "hungarian", "matching_objective": "pairwise_f1",
               "discovery_purity_threshold": .9, "discovery_coverage_threshold": .1}
    path.with_name(path.stem + "-provenance.json").write_text(json.dumps(payload), encoding="utf-8")


def write_freezes(root: Path) -> object:
    paths = stats.EvidencePaths.canonical(root); methods = design().methods
    release = []
    for size in (20_000,):
        for method in methods:
            item = row("synthetic", f"moons_2d__quality__n{size}", method)
            item["evidence_stage"] = "development"; release.append(item)
    for method in methods:
        item = row("synthetic", "moons_2d__quality__n20000", method, 211)
        item["evidence_stage"] = "fresh"; release.append(item)
    scaling = [row("synthetic", "moons_2d__scaling__n500000", "StrataSCAN", 42)]
    cyto = [row("cytometry", "levine", method) for method in methods]
    gaia = [row("gaia", f"field_{index:03d}", method) for index in range(359) for method in methods]
    noise = [row("synthetic", "moons_noise25__quality__n20000", "StrataSCAN", 211)]
    for path, values in zip((paths.synthetic_release, paths.synthetic_scaling, paths.cytometry, paths.gaia, paths.noise_factorial),
                            (release, scaling, cyto, gaia, noise), strict=True):
        pd.DataFrame(values).to_csv(path, index=False); provenance(path)
    return paths


def test_generate_statistics_accepts_only_new_contract(tmp_path: Path) -> None:
    summary = stats.generate_statistics(write_freezes(tmp_path), design())
    assert summary["schema_version"] == 2
    assert summary["evaluation_contract"]["protocol_version"] == "target-discovery-v1"
    assert summary["synthetic"]["fresh"]["comparisons"]["DBSCAN"]["joint_successful_pairs"] == 1
    assert summary["gaia"]["methods"]["StrataSCAN"]["fields"] == 359


def test_rejects_legacy_matching_even_when_matrix_is_complete(tmp_path: Path) -> None:
    paths = write_freezes(tmp_path)
    frame = pd.read_csv(paths.synthetic_release); frame["matching_strategy"] = "many_to_one"
    frame.to_csv(paths.synthetic_release, index=False); provenance(paths.synthetic_release)
    with pytest.raises(ValueError, match="wrong matching_strategy"):
        stats.generate_statistics(paths, design())


def test_rejects_non_synthetic_noise_field(tmp_path: Path) -> None:
    paths = write_freezes(tmp_path)
    frame = pd.read_csv(paths.cytometry); frame["noise_f1"] = .9
    frame.to_csv(paths.cytometry, index=False); provenance(paths.cytometry)
    with pytest.raises(ValueError, match="synthetic-noise names"):
        stats.generate_statistics(paths, design())


def test_rejects_wrong_discovery_threshold(tmp_path: Path) -> None:
    paths = write_freezes(tmp_path)
    frame = pd.read_csv(paths.gaia); frame["discovery_purity_threshold"] = .8
    frame.to_csv(paths.gaia, index=False); provenance(paths.gaia)
    with pytest.raises(ValueError, match="wrong discovery_purity_threshold"):
        stats.generate_statistics(paths, design())


def test_rejects_predecessor_stratascan_rows(tmp_path: Path) -> None:
    paths = write_freezes(tmp_path)
    frame = pd.read_csv(paths.synthetic_release)
    frame.loc[frame["method"].eq("StrataSCAN"), "package_version"] = "0.2.3"
    frame.to_csv(paths.synthetic_release, index=False); provenance(paths.synthetic_release)
    with pytest.raises(ValueError, match="wrong StrataSCAN release version"):
        stats.generate_statistics(paths, design())


def test_rejects_non_native_synthetic_baselines(tmp_path: Path) -> None:
    paths = write_freezes(tmp_path)
    frame = pd.read_csv(paths.synthetic_release)
    frame.loc[frame["method"].eq("DBSCAN"), "evaluation_generation"] = "legacy_single_target_exact_rescore"
    frame.to_csv(paths.synthetic_release, index=False); provenance(paths.synthetic_release)
    with pytest.raises(ValueError, match="non-native evaluation generation"):
        stats.generate_statistics(paths, design())


def test_rejects_unlabeled_gaia_legacy_external_rows(tmp_path: Path) -> None:
    paths = write_freezes(tmp_path)
    frame = pd.read_csv(paths.gaia)
    frame.loc[frame["method"].eq("DBSCAN"), "evaluation_generation"] = "native_target_discovery_v1"
    frame.to_csv(paths.gaia, index=False); provenance(paths.gaia)
    with pytest.raises(ValueError, match="not exact single-target legacy rescores"):
        stats.generate_statistics(paths, design())


def test_missing_canonical_freeze_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="synthetic release freeze is missing"):
        stats.generate_statistics(stats.EvidencePaths.canonical(tmp_path), design())


def test_default_root_is_repaired_release() -> None:
    assert stats.EVIDENCE_ROOT.as_posix().endswith("results/published/v0.2.4/target-discovery-v1")
    locked = stats.EvidenceDesign.canonical()
    assert len(locked.cases) * (len(locked.development_seeds) + len(locked.fresh_seeds)) * len(locked.methods) == 504
    assert locked.development_sizes == (20_000,)
    assert len(locked.noise_cases) * len(locked.noise_seeds) * len(locked.noise_methods) == 270


def test_noise_fraction_975_is_encoded_as_point_975() -> None:
    locked = design()
    frame = pd.DataFrame([
        row("synthetic", "moons_noise975__quality__n20000", "StrataSCAN", 211)
    ])
    result = stats._noise(frame, locked)
    assert ".975" in result["trajectories"]["StrataSCAN"] or "0.975" in result["trajectories"]["StrataSCAN"]
