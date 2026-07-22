from __future__ import annotations

import numpy as np

import stratascan
from stratascan import StrataSCAN


def test_release_exports_only_current_algorithm() -> None:
    assert stratascan.__version__ == "0.1.0"
    assert set(stratascan.__all__) == {
        "GammaStrictCoreConfig",
        "GammaStrictCoreResult",
        "HNSWConfig",
        "KNNGraph",
        "StrataSCAN",
        "UniformTailConfig",
        "build_knn_graph",
        "gamma_strict_core_from_graph",
    }


def test_default_estimator_is_frozen_strict_core() -> None:
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(-3.0, 0.25, size=(160, 4)),
            rng.normal(3.0, 0.25, size=(160, 4)),
            rng.uniform(-8.0, 8.0, size=(180, 4)),
        ]
    ).astype(np.float32)
    first = StrataSCAN(backend="brute").fit_predict(X)
    second = StrataSCAN(backend="brute").fit_predict(X)
    assert np.array_equal(first, second)
    assert np.unique(first[first >= 0]).size >= 2


def test_estimator_records_release_profile() -> None:
    X = np.random.default_rng(7).normal(size=(250, 3)).astype(np.float32)
    model = StrataSCAN(backend="brute").fit(X)
    assert model.labels_.shape == (250,)
    assert model.core_sample_indices_.ndim == 1
    assert model.profile_["strict_core_dense_quantile"] == 0.95
    assert model.profile_["strict_core_tail_probe_alpha"] == 0.05
