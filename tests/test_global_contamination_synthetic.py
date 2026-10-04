from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.datasets import load_synthetic
from benchmarks.run_benchmark import expand_jobs, read_evaluation_protocol, read_protocol


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "benchmarks" / "protocols/global_contamination.json"
EVALUATION = REPO / "benchmarks" / "evaluation_protocol.v1.json"


def _spec(noise_fraction: float) -> dict[str, object]:
    signal_size = 1200
    background_size = int(round(signal_size * noise_fraction / (1.0 - noise_fraction)))
    return {
        "family": "global_contamination_invariance",
        "dimension": 2,
        "noise_fraction": noise_fraction,
        "signal_size": signal_size,
        "n": signal_size + background_size,
        "background_intensity": 12.0,
        "background_strip_height": 10.0,
        "background_inner_edge": 6.0,
    }


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    order = np.lexsort(tuple(values[:, column] for column in reversed(range(values.shape[1]))))
    return values[order]


def test_targets_are_fixed_and_background_is_nested_across_noise_fractions() -> None:
    lower = load_synthetic(_spec(0.25), seed=211)
    higher = load_synthetic(_spec(0.75), seed=211)

    for target in range(3):
        np.testing.assert_array_equal(
            _sorted_rows(lower.X[lower.y == target]),
            _sorted_rows(higher.X[higher.y == target]),
        )
    lower_background = {tuple(row) for row in lower.X[lower.y < 0].tolist()}
    higher_background = {tuple(row) for row in higher.X[higher.y < 0].tolist()}
    assert lower_background < higher_background
    assert lower.metadata["signal_size"] == higher.metadata["signal_size"] == 1200
    assert lower.metadata["background_intensity"] == higher.metadata["background_intensity"] == 12.0
    assert lower.metadata["background_support_area"] < higher.metadata["background_support_area"]


def test_realized_fraction_and_remote_support_match_the_contract() -> None:
    dataset = load_synthetic(_spec(0.90), seed=223)
    assert dataset.X.shape == (12000, 2)
    assert np.count_nonzero(dataset.y >= 0) == 1200
    assert np.count_nonzero(dataset.y < 0) == 10800
    assert dataset.metadata["realized_noise_fraction"] == pytest.approx(0.90)
    assert np.all(np.abs(dataset.X[dataset.y < 0, 0]) >= 6.0)
    assert dataset.metadata["controlled_factor"] == (
        "global_background_support_at_fixed_local_intensity"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dimension": 3}, "defined only in 2D"),
        ({"background_intensity": 0.0}, "background_intensity"),
        ({"n": 999}, "requires n consistent"),
    ],
)
def test_invalid_global_contamination_specs_fail_closed(
    change: dict[str, object], message: str
) -> None:
    spec = {**_spec(0.50), **change}
    with pytest.raises(ValueError, match=message):
        load_synthetic(spec, seed=211)


def test_frozen_protocol_expands_to_the_declared_paired_matrix() -> None:
    protocol = read_protocol(PROTOCOL)
    evaluation = read_evaluation_protocol(EVALUATION)
    jobs = expand_jobs(protocol, "synthetic", evaluation)

    assert len(jobs) == protocol["analysis"]["expected_records"] == 30
    assert {job["seed"] for job in jobs} == {211, 223, 227, 229, 233}
    assert {job["dataset"]["signal_size"] for job in jobs} == {1200}
    assert sorted({job["dataset"]["n"] for job in jobs}) == [
        1600,
        2400,
        4800,
        12000,
        24000,
        120000,
    ]
    assert all(job["method"] == "StrataSCAN" for job in jobs)
    assert all(job["evaluation"]["background_semantics"] == "known_synthetic_noise" for job in jobs)
    assert len({job["job_id"] for job in jobs}) == 30
