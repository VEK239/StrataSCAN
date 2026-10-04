from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import stratascan
from stratascan import StrataSCAN


def test_release_exports_only_current_algorithm() -> None:
    assert stratascan.__version__ == "0.2.4"
    assert set(stratascan.__all__) == {
        "GammaMDLConfig",
        "GammaStrictCoreConfig",
        "GammaStrictCoreResult",
        "HNSWConfig",
        "KNNGraph",
        "MultiscaleConfig",
        "OptimizationStrataSCAN",
        "OptimizationStrictCoreConfig",
        "OptimizationStrictCoreResult",
        "PredictiveMultiscaleConfig",
        "PredictiveMultiscaleStrataSCAN",
        "StrataSCAN",
        "UniformTailConfig",
        "build_knn_graph",
        "gamma_strict_core_from_graph",
    }


def test_default_estimator_is_semantic_mdl_strict_core() -> None:
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
    assert model.profile_["algorithm_version"] == "0.2.3"
    assert model.profile_["algorithm"] == "OptimizationStrataSCAN"
    assert model.profile_["mdl_gamma_background_distribution"] == "gamma_components"
    assert model.profile_["mdl_gamma_grouping"] == "largest_rate_gap"
    assert model.profile_["mdl_gamma_adaptive_search"] is True
    assert model.profile_["optimization_unified_background_scan"] is True


def test_short_public_name_aliases_explicit_release_class() -> None:
    assert stratascan.StrataSCAN is stratascan.OptimizationStrataSCAN


def test_release_protocol_matches_public_defaults() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "protocols/release_defaults.json"
    )
    parameters = json.loads(protocol_path.read_text(encoding="utf-8"))[
        "method_parameters"
    ]
    model = StrataSCAN()
    gamma = model.gamma_config
    extraction = model.optimization_config
    assert parameters["version"] == stratascan.__version__
    assert parameters["k"] == model.k
    assert parameters["shell_ranks"] == list(gamma.ranks)
    assert parameters["criterion"] == f"{gamma.entropy_scope}_{gamma.criterion}"
    assert parameters["initial_component_bound"] == gamma.max_components
    assert parameters["hard_component_bound"] == gamma.hard_max_components
    assert parameters["background_distribution"] == gamma.background_distribution
    assert parameters["component_roles"] == gamma.component_roles
    assert parameters["initializations"] == gamma.n_init
    assert parameters["tolerance"] == gamma.tolerance
    assert parameters["confirmation_min_components"] == gamma.confirmation_min_components
    assert (
        parameters["confirmation_min_duplicate_profiles"]
        == gamma.confirmation_min_duplicate_profiles
    )
    assert parameters["core_rank"] == extraction.core_rank
    assert parameters["connectivity_rank"] == extraction.connectivity_rank
