from __future__ import annotations

import numpy as np

from stratascan.baselines.dpc_knn import (
    METHOD_ID,
    _fit_dpc_knn_2016,
    _gb_auto_centers,
    _knn_density,
    run_dpc_knn_2016,
)


def test_du_density_matches_published_equation() -> None:
    X = np.array([[0.0], [1.0], [3.0], [10.0]])
    rho = _knn_density(X, k=2, block_size=2)
    log_rho = np.array([-5.0, -2.5, -6.5, -65.0])
    expected = np.exp(log_rho - np.max(log_rho))
    np.testing.assert_allclose(rho, expected)


def test_gb_selector_uses_last_above_mean_gap() -> None:
    rho = np.array([10.0, 9.0, 8.0, 1.0, 1.0])
    delta = np.array([10.0, 9.0, 8.0, 1.0, 1.0])
    centers, mean_gap, fallback = _gb_auto_centers(rho, delta)
    np.testing.assert_array_equal(centers, np.array([0, 1]))
    assert mean_gap == 12.0
    assert not fallback


def test_exact_parents_precede_children_and_all_points_are_assigned() -> None:
    X = np.array(
        [
            [-3.1, 0.0],
            [-3.0, 0.1],
            [-2.9, -0.1],
            [2.8, 0.0],
            [3.0, 0.1],
            [3.2, -0.1],
        ]
    )
    state = _fit_dpc_knn_2016(X, block_size=2)
    position = np.empty(X.shape[0], dtype=np.int64)
    position[state.density_order] = np.arange(X.shape[0])
    non_root = state.parent >= 0
    assert np.all(position[state.parent[non_root]] < position[np.flatnonzero(non_root)])
    assert np.all(state.labels >= 0)
    np.testing.assert_array_equal(
        state.labels, _fit_dpc_knn_2016(X, block_size=3).labels
    )


def test_public_result_records_hybrid_provenance_and_profile() -> None:
    result = run_dpc_knn_2016(np.array([[0.0], [1.0], [4.0]]), block_size=2)
    assert result.metadata["method_id"] == METHOD_ID
    assert result.metadata["neighbour_fraction"] == 0.02
    assert result.metadata["k"] == 1
    assert result.metadata["all_points_assigned"] is True
    assert result.metadata["noise_model"] == "none"
    assert result.profile["n_centers"] == result.metadata["n_centers"]
    assert result.labels.dtype == np.int64
    assert np.all(result.labels >= 0)
