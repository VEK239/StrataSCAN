from __future__ import annotations

import numpy as np

from stratascan.experimental import ResidualStrataSCAN, _merge_pass_labels


def test_merge_pass_labels_filters_small_components_and_offsets_labels() -> None:
    output = np.full(8, -1, dtype=np.int64)
    remaining = np.array([0, 2, 3, 5, 6, 7], dtype=np.int64)
    local = np.array([0, 0, 1, 1, 1, -1], dtype=np.int64)
    next_label, accepted = _merge_pass_labels(
        output,
        remaining,
        local,
        next_label=4,
        min_cluster_size=3,
    )
    assert next_label == 5
    assert np.array_equal(np.flatnonzero(accepted), np.array([2, 3, 4]))
    assert np.array_equal(output[[3, 5, 6]], np.array([4, 4, 4]))
    assert np.all(output[[0, 1, 2, 4, 7]] == -1)


def test_merge_pass_labels_rejects_residual_dominating_component() -> None:
    output = np.full(10, -1, dtype=np.int64)
    remaining = np.arange(10, dtype=np.int64)
    local = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1], dtype=np.int64)
    next_label, accepted = _merge_pass_labels(
        output,
        remaining,
        local,
        next_label=2,
        min_cluster_size=2,
        max_cluster_fraction=0.3,
    )
    assert next_label == 3
    assert np.array_equal(np.flatnonzero(accepted), np.array([6, 7]))
    assert np.all(output[:6] == -1)
    assert np.all(output[6:8] == 2)


def test_residual_estimator_smoke_and_profile() -> None:
    rng = np.random.default_rng(42)
    values = np.vstack(
        [
            rng.normal(-2.5, 0.2, size=(100, 4)),
            rng.normal(2.5, 0.2, size=(100, 4)),
            rng.uniform(-7.0, 7.0, size=(120, 4)),
        ]
    ).astype(np.float32)
    model = ResidualStrataSCAN(
        backend="brute",
        max_passes=2,
        accepted_cluster_size=10,
        min_residual_size=20,
    ).fit(values)
    assert model.labels_.shape == (len(values),)
    assert model.n_clusters_ >= 2
    assert model.profile_["experimental_variant"] == "intrinsic-multistratum-residual-v2"
    assert 1 <= len(model.profile_["experimental_passes"]) <= 2
