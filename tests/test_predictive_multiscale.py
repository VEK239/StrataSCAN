import numpy as np
from scipy.special import gammaln, logsumexp

from stratascan import build_knn_graph
from stratascan.predictive import (
    PredictiveMultiscaleConfig,
    _fit_rate_profiles,
    _gamma_workspace,
    _log_probabilities,
    _responsibilities,
    estimate_predictive_multiscale_stratification,
)


def test_constrained_rate_profiles_are_log_linear_in_rank() -> None:
    config = PredictiveMultiscaleConfig(validation_repeats=2)
    shapes = np.array([4.0, 4.0, 8.0, 16.0])
    ranks = np.array([4.0, 8.0, 16.0, 32.0])
    weighted_shells = np.array([[2.0, 4.0, 9.0, 21.0], [8.0, 7.0, 6.0, 5.0]])
    effective = np.array([20.0, 15.0])
    mu, slopes, rates = _fit_rate_profiles(
        weighted_shells,
        effective,
        shapes=shapes,
        log_ranks=np.log(ranks),
        config=config,
    )
    expected = mu[:, None] + slopes[:, None] * np.log(ranks)[None, :]
    np.testing.assert_allclose(np.log(rates), expected, rtol=1e-10, atol=1e-10)
    scaled = weighted_shells * np.exp(slopes[:, None] * np.log(ranks)[None, :])
    weighted_log_rank = np.sum(scaled * np.log(ranks)[None, :], axis=1) / np.sum(
        scaled, axis=1
    )
    target = np.sum(shapes * np.log(ranks)) / np.sum(shapes)
    np.testing.assert_allclose(weighted_log_rank, target, rtol=1e-11, atol=1e-11)


def test_cached_gamma_likelihood_matches_direct_formula() -> None:
    shells = np.array([[0.2, 0.4, 0.9], [0.5, 0.8, 1.7]])
    shapes = np.array([4.0, 4.0, 8.0])
    weights = np.array([0.35, 0.65])
    rates = np.array([[3.0, 2.0, 1.0], [1.5, 1.0, 0.5]])
    direct = (
        np.log(weights)[None, :]
        + np.sum(
            shapes[None, None, :] * np.log(rates)[None, :, :]
            - gammaln(shapes)[None, None, :]
            + (shapes[None, None, :] - 1.0)
            * np.log(shells)[:, None, :]
            - shells[:, None, :] * rates[None, :, :],
            axis=2,
        )
    )
    cached = _log_probabilities(
        shells,
        shapes=shapes,
        weights=weights,
        rates=rates,
        workspace=_gamma_workspace(shells, shapes),
    )
    np.testing.assert_allclose(cached, direct, rtol=1e-12, atol=1e-12)


def test_gamma_responsibilities_match_stable_logsumexp() -> None:
    shells = np.array([[0.2, 0.4, 0.9], [0.5, 0.8, 1.7]])
    shapes = np.array([4.0, 4.0, 8.0])
    weights = np.array([0.35, 0.65])
    rates = np.array([[3.0, 2.0, 1.0], [1.5, 1.0, 0.5]])
    workspace = _gamma_workspace(shells, shapes)
    log_probability = _log_probabilities(
        shells, shapes=shapes, weights=weights, rates=rates, workspace=workspace
    )
    actual, actual_norm = _responsibilities(
        shells, shapes=shapes, weights=weights, rates=rates, workspace=workspace
    )
    expected_norm = logsumexp(log_probability, axis=1)
    expected = np.exp(log_probability - expected_norm[:, None])
    np.testing.assert_allclose(actual_norm, expected_norm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_predictive_selection_returns_aligned_stratification() -> None:
    rng = np.random.default_rng(42)
    dense = rng.normal(0.0, 0.15, size=(300, 2))
    background = rng.uniform(-4.0, 4.0, size=(500, 2))
    X = np.asarray(np.vstack([dense, background]), dtype=np.float32)
    graph, _ = build_knn_graph(X, k=32, backend="kd_tree", n_jobs=1)
    result = estimate_predictive_multiscale_stratification(
        graph,
        ambient_dimension=2.0,
        config=PredictiveMultiscaleConfig(
            min_components=2,
            max_components=4,
            validation_repeats=2,
            final_n_init=2,
            max_iter=100,
        ),
    )
    diagnostics = result.diagnostics
    assert result.groups.shape == (X.shape[0],)
    assert result.background_probability.shape == (X.shape[0],)
    assert 2 <= diagnostics["predictive_components"] <= 4
    assert len(diagnostics["predictive_validation_log_likelihood"]) == 3
    assert len(diagnostics["predictive_assignment_stability"]) == 3
    eligible = np.flatnonzero(diagnostics["predictive_within_one_se"])
    assert eligible.size
    assert diagnostics["predictive_components"] == int(
        diagnostics["predictive_candidate_components"][eligible[0]]
    )
    assert np.all(np.isfinite(diagnostics["predictive_rates"]))


def test_coarse_predictive_search_evaluates_fewer_models() -> None:
    rng = np.random.default_rng(7)
    X = np.asarray(
        np.vstack(
            [rng.normal(0.0, 0.2, size=(180, 2)), rng.uniform(-3.0, 3.0, size=(320, 2))]
        ),
        dtype=np.float32,
    )
    graph, _ = build_knn_graph(X, k=32, backend="kd_tree", n_jobs=1)
    outputs = {}
    for mode in ("full", "coarse_refine", "coarse_refine_warm"):
        outputs[mode] = estimate_predictive_multiscale_stratification(
            graph,
            ambient_dimension=2.0,
            config=PredictiveMultiscaleConfig(
                min_components=2,
                max_components=6,
                validation_repeats=2,
                final_n_init=1,
                max_iter=60,
                search_mode=mode,
            ),
        )
        diagnostics = outputs[mode].diagnostics
        candidates = diagnostics["predictive_candidate_components"]
        assert diagnostics["predictive_components"] in candidates
        assert len(candidates) == len(
            diagnostics["predictive_validation_log_likelihood"]
        )
        assert diagnostics["predictive_search_mode"] == mode

    full_count = outputs["full"].diagnostics["predictive_fit_count"]
    assert outputs["coarse_refine"].diagnostics["predictive_fit_count"] < full_count
    assert (
        outputs["coarse_refine_warm"].diagnostics["predictive_fit_count"] < full_count
    )
