from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from benchmarks.datasets import load_synthetic
from benchmarks.run_benchmark import expand_jobs, read_evaluation_protocol, read_protocol
from stratascan.synthetic import make_density_contrast


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "protocol.v0.2.4-density-contrast.json"
EVALUATION = ROOT / "benchmarks" / "evaluation_protocol.v1.json"


def test_density_contrast_is_reproducible_and_holds_counts_fixed() -> None:
    first_x, first_y = make_density_contrast(
        12_000,
        dimension=2,
        density_ratio=256,
        shape="gaussian",
        noise_fraction=0.25,
        seed=211,
    )
    second_x, second_y = make_density_contrast(
        12_000,
        dimension=2,
        density_ratio=256,
        shape="gaussian",
        noise_fraction=0.25,
        seed=211,
    )

    assert first_x.shape == (12_000, 2)
    assert first_x.dtype == np.float32
    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)
    assert np.mean(first_y < 0) == 0.25
    assert [np.sum(first_y == cluster) for cluster in range(6)] == [1_500] * 6


@pytest.mark.parametrize("shape", ["gaussian", "anisotropic"])
def test_declared_peak_density_ratio_is_realized_by_covariance_determinants(
    shape: str,
) -> None:
    x, y = make_density_contrast(
        12_000,
        dimension=2,
        density_ratio=256,
        shape=shape,
        noise_fraction=0.25,
        seed=211,
    )
    covariances = [np.cov(x[y == cluster], rowvar=False) for cluster in range(6)]
    determinants = np.asarray([np.linalg.det(covariance) for covariance in covariances])
    empirical_peak_ratio = float(np.sqrt(determinants[-1] / determinants[0]))

    # Sampling error around the analytic ratio is small at 1,500 rows/target.
    assert empirical_peak_ratio == pytest.approx(256.0, rel=0.12)
    if shape == "anisotropic":
        eigenvalues = np.linalg.eigvalsh(covariances[0])
        empirical_axis_ratio = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
        assert empirical_axis_ratio == pytest.approx(4.0, rel=0.12)


def test_demanding_2d_case_spans_sixteen_fold_local_distance_scale() -> None:
    x, y = make_density_contrast(
        12_000,
        dimension=2,
        density_ratio=256,
        shape="gaussian",
        noise_fraction=0.25,
        seed=211,
    )
    distances, _ = NearestNeighbors(n_neighbors=5, n_jobs=1).fit(x).kneighbors(x)
    dense_median = float(np.median(distances[y == 0, 4]))
    sparse_median = float(np.median(distances[y == 5, 4]))
    assert sparse_median / dense_median == pytest.approx(16.0, rel=0.15)


def test_density_contrast_loader_preserves_declared_geometry() -> None:
    spec = {
        "id": "density_r64_anisotropic_8d",
        "family": "density_contrast",
        "dimension": 8,
        "density_ratio": 64,
        "shape": "anisotropic",
        "noise_fraction": 0.25,
        "n_clusters": 6,
        "n": 1_200,
    }
    dataset = load_synthetic(spec, seed=223)
    assert dataset.X.shape == (1_200, 8)
    assert dataset.metadata["family"] == "density_contrast"
    assert dataset.metadata["dimension"] == 8
    assert set(np.unique(dataset.y[dataset.y >= 0])) == set(range(6))


def test_protocol_is_paired_and_uses_locked_target_discovery_evaluation() -> None:
    protocol = read_protocol(PROTOCOL)
    evaluation = read_evaluation_protocol(EVALUATION)
    jobs = expand_jobs(protocol, "synthetic", evaluation)

    assert protocol["methods"] == protocol["stochastic_methods"]
    assert len(protocol["synthetic"]["cases"]) == 9
    assert len(jobs) == 9 * 9 * 3
    assert {job["seed"] for job in jobs} == {211, 223, 227}
    assert {job["dataset"]["density_ratio"] for job in jobs} == {16, 64, 256}
    assert {job["dataset"]["dimension"] for job in jobs} == {2, 8}
    assert {job["dataset"]["shape"] for job in jobs} == {"gaussian", "anisotropic"}
    assert {job["evaluation"]["protocol_version"] for job in jobs} == {
        "target-discovery-v1"
    }
    assert {job["evaluation"]["mode"] for job in jobs} == {"full_partition"}

    # The JSON itself remains machine-readable and records the noncanonical scope.
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert "separate_noncanonical" in raw["purpose"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dimension": 1, "density_ratio": 16}, "dimension"),
        ({"dimension": 2, "density_ratio": 0.5}, "density_ratio"),
        ({"dimension": 2, "density_ratio": 16, "shape": "rings"}, "shape"),
        ({"dimension": 2, "density_ratio": 16, "noise_fraction": 1.0}, "noise_fraction"),
    ],
)
def test_density_contrast_rejects_invalid_designs(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        make_density_contrast(1_000, **kwargs)
