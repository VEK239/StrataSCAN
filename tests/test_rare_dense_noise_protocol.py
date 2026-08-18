from __future__ import annotations

from pathlib import Path

from benchmarks.run_benchmark import expand_jobs, read_evaluation_protocol, read_protocol


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "benchmarks/protocol.v0.2.4-target-discovery-v1-rare-dense-noise-all-methods.json"
EVALUATION = REPO / "benchmarks/evaluation_protocol.v1.json"


def test_rare_dense_noise_protocol_is_the_complete_paired_all_method_matrix() -> None:
    protocol = read_protocol(PROTOCOL)
    evaluation = read_evaluation_protocol(EVALUATION)
    jobs = expand_jobs(protocol, "synthetic", evaluation)

    assert len(jobs) == protocol["analysis"]["expected_records"] == 270
    assert len({job["job_id"] for job in jobs}) == 270
    assert {job["seed"] for job in jobs} == {211, 223, 227, 229, 233}
    assert len({job["method"] for job in jobs}) == 9
    assert sorted({job["dataset"]["n"] for job in jobs}) == [
        400,
        600,
        1200,
        3000,
        6000,
        30000,
    ]
    assert {job["dataset"]["signal_size"] for job in jobs} == {300}
    assert {job["dataset"]["background_intensity"] for job in jobs} == {3.0}
    assert all(job["evaluation"]["background_semantics"] == "known_synthetic_noise" for job in jobs)
