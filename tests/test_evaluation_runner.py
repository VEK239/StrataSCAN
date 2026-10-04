from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.datasets import Dataset
from benchmarks.run_benchmark import (
    DEFAULT_EVALUATION_PROTOCOL,
    evaluation_for_job,
    expand_jobs,
    read_evaluation_protocol,
    validate,
    write_csv,
)
from benchmarks.worker import atomic_evaluation_state


def test_versioned_protocol_resolves_dataset_modes_and_is_frozen_into_job_ids() -> None:
    evaluation = read_evaluation_protocol(DEFAULT_EVALUATION_PROTOCOL)
    mosmann = evaluation_for_job(evaluation, "cytometry", "mosmann")
    levine = evaluation_for_job(evaluation, "cytometry", "levine")
    synthetic = evaluation_for_job(evaluation, "synthetic", "toy__quality__n4")

    assert mosmann["mode"] == "target_discovery"
    assert levine["mode"] == "full_partition"
    assert synthetic["background_semantics"] == "known_synthetic_noise"
    protocol = {
        "protocol_version": "toy-v1",
        "methods": ["DBSCAN"],
        "stochastic_methods": [],
        "synthetic": {
            "quality_sizes": [4],
            "scaling_sizes": [],
            "quality_seeds": [42],
            "scaling_seeds": [],
            "cases": [{"id": "toy", "family": "moons", "dimension": 2}],
        },
    }
    job = expand_jobs(protocol, "synthetic", evaluation)[0]
    assert job["evaluation"] == synthetic
    without_evaluation = expand_jobs(protocol, "synthetic")[0]
    assert job["job_id"] != without_evaluation["job_id"]


def test_lossless_artifact_is_linked_validated_and_exposed_in_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    jobs_dir = run_dir / "jobs"
    state_dir = run_dir / "evaluation_state"
    jobs_dir.mkdir(parents=True)
    state_dir.mkdir()
    job_id = "synthetic__toy__DBSCAN__42__abc"
    reference = f"evaluation_state/{job_id}.npz"
    state_path = state_dir / f"{job_id}.npz"
    dataset = Dataset(
        X=np.zeros((5, 2), dtype=np.float32),
        y=np.array([0, 0, 1, 1, -1], dtype=np.int64),
        target_names=["a", "b"],
        metadata={},
    )
    artifact = atomic_evaluation_state(
        state_path, dataset, np.array([7, 7, 9, -1, -1]), reference
    )
    result = {
        "schema_version": 2,
        "job_id": job_id,
        "status": "ok",
        "suite": "synthetic",
        "dataset_id": "toy",
        "method": "DBSCAN",
        "seed": 42,
        "n": 5,
        "evaluation_artifact": artifact,
        "metrics": {
            "macro_target_f1": 0.5,
            "target_matches": [{"target": "a", "f1": 1.0}],
        },
    }
    (jobs_dir / f"{job_id}.json").write_text(json.dumps(result), encoding="utf-8")
    manifest = {
        "jobs": [
            {
                "job_id": job_id,
                "suite": "synthetic",
                "dataset_id": "toy",
                "method": "DBSCAN",
                "seed": 42,
            }
        ],
        "evaluation_artifacts": {"required_for_status": "ok"},
    }

    report, results = validate(manifest, jobs_dir)
    assert report["complete"]
    assert report["invalid_evaluation_artifact_job_ids"] == []
    with np.load(state_path, allow_pickle=False) as state:
        np.testing.assert_array_equal(state["y_true"], dataset.y)
        np.testing.assert_array_equal(state["y_pred"], [7, 7, 9, -1, -1])

    csv_path = run_dir / "results.csv"
    write_csv(csv_path, results)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["evaluation_artifact_path"] == reference
    assert row["evaluation_artifact_sha256"] == artifact["sha256"]
    assert json.loads(row["target_matches_json"])[0]["target"] == "a"

    state_path.write_bytes(state_path.read_bytes() + b"tampered")
    report, _ = validate(manifest, jobs_dir)
    assert not report["complete"]
    assert report["invalid_evaluation_artifact_job_ids"] == [job_id]
