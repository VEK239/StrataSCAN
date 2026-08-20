from __future__ import annotations

import numpy as np

from benchmarks import methods


class _FakeEstimator:
    last_gamma_config = None
    last_optimization_config = None

    def __init__(self, *args, gamma_config=None, optimization_config=None, **kwargs) -> None:
        type(self).last_gamma_config = gamma_config
        type(self).last_optimization_config = optimization_config
        self.profile_ = {"fake": True}

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.int64)


def test_public_benchmark_profile_records_patch_release(monkeypatch) -> None:
    monkeypatch.setattr(methods, "StrataSCAN", _FakeEstimator)
    result = methods.run_method("StrataSCAN", np.zeros((4, 2)), seed=7)
    assert result.parameters["version"] == "0.2.3"
    assert result.parameters["component_selection"] == "semantic_icl_with_adaptive_bound"
    assert result.parameters["hard_component_bound"] == 24


def test_oedm_snapshot_profile_and_search_are_explicitly_fixed(monkeypatch) -> None:
    monkeypatch.setattr(methods, "OptimizationStrataSCAN", _FakeEstimator)
    result = methods.run_method("StrataSCAN-Optimization", np.zeros((4, 2)), seed=7)
    gamma = _FakeEstimator.last_gamma_config
    assert gamma is not None
    assert gamma.max_components == 8
    assert gamma.hard_max_components == 8
    assert gamma.adaptive_components is False
    assert result.parameters["version"] == "0.2.0-dev10-oedm2026-evaluated"
    assert result.parameters["component_selection"] == "basis_icl_fixed_candidate_range"
    assert "adaptive" not in result.parameters["stratification_model"]
    assert "adaptive" not in result.parameters["background_model"]


def test_component_ablation_changes_only_declared_gamma_control(monkeypatch) -> None:
    monkeypatch.setattr(methods, "OptimizationStrataSCAN", _FakeEstimator)
    result = methods.run_method("StrataSCAN-FixedComponentBound", np.zeros((4, 2)), seed=7)
    gamma = _FakeEstimator.last_gamma_config
    assert gamma is not None
    assert gamma.adaptive_components is False
    assert gamma.max_components == gamma.hard_max_components == 8
    assert result.parameters["ablation_of"] == "adaptive_component_bound"


def test_structural_ablations_change_exactly_one_declared_control(monkeypatch) -> None:
    monkeypatch.setattr(methods, "OptimizationStrataSCAN", _FakeEstimator)
    defaults = methods.OptimizationStrictCoreConfig()
    expected = {
        "StrataSCAN-NoDensityStratification": ("signal_strata", "single_layer"),
        "StrataSCAN-FixedCoreScale": ("core_threshold_rule", "fixed_quantile"),
        "StrataSCAN-NoBackgroundRecovery": ("background_recovery", False),
    }
    fields = (
        "signal_strata",
        "core_threshold_rule",
        "fixed_core_quantile",
        "background_recovery",
        "core_rank",
        "connectivity_rank",
    )
    for method, (changed_field, expected_value) in expected.items():
        result = methods.run_method(method, np.zeros((4, 2)), seed=7)
        config = _FakeEstimator.last_optimization_config
        assert config is not None
        changed = {
            field for field in fields
            if getattr(config, field) != getattr(defaults, field)
        }
        assert changed == {changed_field}
        assert getattr(config, changed_field) == expected_value
        assert result.parameters["version"] == "0.2.4-structural-ablation-v1"


def test_dpc_dispatch_preserves_transparent_hybrid_metadata(monkeypatch) -> None:
    expected = methods.run_dpc_knn_2016(
        np.array([[0.0], [1.0], [4.0]]), block_size=2
    )
    monkeypatch.setattr(methods, "run_dpc_knn_2016", lambda X: expected)
    result = methods.run_method(
        methods.DPC_KNN_METHOD_ID, np.zeros((3, 1)), seed=7
    )
    assert result.parameters["canonical_core_doi"] == "10.1016/j.knosys.2016.02.001"
    assert result.parameters["center_selector_doi"] == "10.1016/j.knosys.2020.106350"
    assert result.parameters["all_points_assigned"] is True
