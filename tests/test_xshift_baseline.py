from __future__ import annotations

import numpy as np

from stratascan.baselines import dispatch_baseline


def test_xshift_separates_two_angular_modes_deterministically() -> None:
    rng = np.random.default_rng(7)
    first = rng.normal([4.0, 0.0], [0.15, 0.15], size=(120, 2))
    second = rng.normal([0.0, 4.0], [0.15, 0.15], size=(120, 2))
    X = np.asarray(np.vstack([first, second]), dtype=np.float32)

    result = dispatch_baseline("X-shift", X, "low_dim", seed=42)

    assert np.array_equal(np.bincount(result.labels), np.array([120, 120]))
    assert np.unique(result.labels[:120]).size == 1
    assert np.unique(result.labels[120:]).size == 1
    assert result.labels[0] != result.labels[-1]
    assert result.metadata["density_k_including_self"] == 62
    assert result.metadata["density_k_selection"] == "published_nilsson_label_free_elbow"
    assert result.metadata["implementation"] == "controlled_python_port"
    assert result.metadata["knn_backend"] == "kd_tree"
    assert result.metadata["official_source_commit"] == (
        "75ab0746f3184c3c1bb48180494aa0a753f077e7"
    )


def test_xshift_handles_tiny_inputs_without_requesting_an_invalid_graph() -> None:
    X = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    result = dispatch_baseline("X-shift", X, "low_dim")

    assert np.array_equal(result.labels, np.array([0, 1]))
    assert result.metadata["density_k_including_self"] == 2
