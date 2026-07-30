from __future__ import annotations

import numpy as np

from benchmarks import methods


class _FakeEstimator:
    last_gamma_config = None

    def __init__(self, *args, gamma_config=None, **kwargs) -> None:
        type(self).last_gamma_config = gamma_config
        self.profile_ = {"fake": True}

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.int64)


def test_public_benchmark_profile_records_patch_release(monkeypatch) -> None:
    monkeypatch.setattr(methods, "StrataSCAN", _FakeEstimator)
    result = methods.run_method("StrataSCAN", np.zeros((4, 2)), seed=7)
    assert result.parameters["version"] == "0.2.2"
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
