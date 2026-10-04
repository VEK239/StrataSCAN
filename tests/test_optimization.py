from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import stratascan.optimization as optimization_module

from stratascan import build_knn_graph
from stratascan.optimization import (
    GammaMDLConfig,
    OptimizationStrataSCAN,
    OptimizationStrictCoreConfig,
    _best_threshold_event_sweep,
    _background_log_density,
    _density_ordered_labels_from_core,
    _fit_gamma_candidate,
    _information_score,
    _inverse_gamma_rate_background_log_density,
    _lognormal_rate_background_density_and_score,
    _lognormal_rate_background_log_density,
    _search_needs_expansion,
    _signal_log_odds,
    _stratum_events,
    _topology_log_bayes_factor,
)
from stratascan.types import KNNGraph


def _brute_force_threshold(
    graph,
    rows: np.ndarray,
    core_distance: np.ndarray,
    unary_gain: np.ndarray,
    rank: int,
    *,
    n_reference: int | None = None,
) -> tuple[int, float]:
    order = np.argsort(core_distance[rows], kind="stable")
    ordered = rows[order]
    best_size = 0
    best_gain = 0.0
    threshold_code = np.log(rows.size)
    for size in range(1, rows.size + 1):
        if (
            size < rows.size
            and core_distance[ordered[size]] == core_distance[ordered[size - 1]]
        ):
            continue
        candidate = np.zeros(graph.n_samples, dtype=bool)
        candidate[ordered[:size]] = True
        candidate_rows = ordered[:size]
        candidate_lookup = np.zeros(graph.n_samples, dtype=bool)
        candidate_lookup[candidate_rows] = True
        successes = int(
            np.sum(
                candidate_lookup[
                    graph.indices[candidate_rows, rank : 2 * rank]
                ]
            )
        )
        topology_gain = _topology_log_bayes_factor(
            size,
            successes,
            graph.n_samples if n_reference is None else n_reference,
            rank,
        )
        gain = float(
            np.sum(unary_gain[candidate_rows]) + topology_gain - threshold_code
        )
        if gain > best_gain:
            best_size = size
            best_gain = gain
    return best_size, best_gain


def test_event_sweep_matches_brute_force_nested_thresholds() -> None:
    rng = np.random.default_rng(11)
    X = np.vstack(
        [
            rng.normal([-2.0, 0.0], 0.15, size=(12, 2)),
            rng.normal([2.0, 0.0], 0.15, size=(12, 2)),
        ]
    ).astype(np.float32)
    graph, _ = build_knn_graph(X, k=8, backend="kd_tree", n_jobs=1)
    rows = np.arange(graph.n_samples, dtype=np.int64)
    core_distance = graph.distances[:, 3]
    background = np.full(graph.n_samples, 0.20, dtype=float)
    unary_gain = _signal_log_odds(background)
    order = np.argsort(core_distance[rows], kind="stable")
    events = _stratum_events(graph, rows, order, rank=4)
    actual = _best_threshold_event_sweep(
        unary_gain[rows[order]], core_distance[rows[order]], *events, rows.size, 4
    )
    expected = _brute_force_threshold(graph, rows, core_distance, unary_gain, 4)
    assert actual[0] == expected[0]
    assert np.isclose(actual[1], expected[1], rtol=1e-10, atol=1e-10)


def test_event_sweep_only_scores_realizable_thresholds_across_exact_ties() -> None:
    unary_gain = np.array([3.0, 3.0, 3.0, -20.0])
    ordering_score = np.array([1.0, 1.0, 1.0, 2.0])
    event_steps = np.array([1, 2, 2], dtype=np.int64)
    event_left = np.array([0, 0, 1], dtype=np.int64)
    event_right = np.array([1, 2, 2], dtype=np.int64)

    best_size, best_gain = _best_threshold_event_sweep(
        unary_gain,
        ordering_score,
        event_steps,
        event_left,
        event_right,
        10,
        1,
    )

    assert best_size == 3
    assert best_gain > 0.0
    assert best_size not in {1, 2}


def test_subset_stratum_uses_local_threshold_code_and_global_reference() -> None:
    unary_gain = np.full(3, 2.0)
    ordering_score = np.array([1.0, 2.0, 3.0])
    event_steps = np.array([1, 2, 2], dtype=np.int64)
    event_left = np.array([0, 0, 1], dtype=np.int64)
    event_right = np.array([1, 2, 2], dtype=np.int64)

    best_size, best_gain = _best_threshold_event_sweep(
        unary_gain,
        ordering_score,
        event_steps,
        event_left,
        event_right,
        10,
        1,
    )
    expected = (
        float(np.sum(unary_gain))
        + _topology_log_bayes_factor(3, 3, 10, 1)
        - math.log(3)
    )

    assert best_size == 3
    assert np.isclose(best_gain, expected)


def test_disabling_split_warm_start_forces_an_independent_candidate_fit(
    monkeypatch,
) -> None:
    calls = {"cold": 0, "split": 0}

    def fake_cold(*args, **kwargs):
        calls["cold"] += 1
        return SimpleNamespace(log_likelihood=2.0)

    def fake_split(*args, **kwargs):
        calls["split"] += 1
        return SimpleNamespace(log_likelihood=1.0)

    monkeypatch.setattr(optimization_module, "_fit_constrained_gamma", fake_cold)
    monkeypatch.setattr(optimization_module, "_fit_once", fake_split)
    previous = SimpleNamespace(responsibilities=np.ones((4, 1)))

    selected = _fit_gamma_candidate(
        np.ones((4, 2)),
        shapes=np.ones(2),
        components=2,
        config=GammaMDLConfig(
            split_warm_start=False,
            n_init=1,
            max_components=2,
            hard_max_components=2,
        ),
        numerical=None,
        workspace=None,
        previous=previous,
    )

    assert selected.log_likelihood == 2.0
    assert calls == {"cold": 1, "split": 0}


def test_optimization_estimator_is_deterministic_and_keeps_noise() -> None:
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal([-2.5, 0.0], 0.18, size=(120, 2)),
            rng.normal([2.5, 0.0], 0.18, size=(120, 2)),
            rng.uniform(-8.0, 8.0, size=(360, 2)),
        ]
    ).astype(np.float32)
    first = OptimizationStrataSCAN(backend="kd_tree", n_jobs=1).fit(X)
    second = OptimizationStrataSCAN(backend="kd_tree", n_jobs=1).fit(X)
    assert np.array_equal(first.labels_, second.labels_)
    assert first.profile_["algorithm_version"] == "0.2.3"
    assert first.profile_["mdl_gamma_criterion"] == "icl"
    assert first.profile_["mdl_gamma_fit_samples"] == X.shape[0]
    assert (
        first.profile_["mdl_gamma_assignment"]
        == "maximum_posterior_basis_then_semantic_role"
    )
    assert first.profile_["mdl_gamma_grouping"] == "largest_rate_gap"
    assert first.profile_["mdl_gamma_adaptive_search"] is True
    assert first.profile_["optimization_unified_background_scan"] is True
    assert first.profile_["optimization_sparse_memory"] is True
    assert first.n_clusters_ >= 1
    assert np.any(first.labels_ == -1)
    trace = first.profile_["optimization_objective_trace"]
    assert trace[-1] <= trace[0]


def test_icl_adds_classification_entropy_to_bic() -> None:
    crisp = np.array([[1.0, 0.0], [0.0, 1.0]])
    ambiguous = np.full((2, 2), 0.5)
    bic = _information_score(-10.0, crisp, 5, "bic")
    assert np.isclose(_information_score(-10.0, crisp, 5, "icl"), bic)
    assert _information_score(-10.0, ambiguous, 5, "icl") > bic


def test_optimization_configuration_has_no_inherited_tail_probe() -> None:
    gamma = GammaMDLConfig()
    extraction = OptimizationStrictCoreConfig()
    assert gamma.criterion == "icl"
    assert gamma.background_distribution == "gamma_components"
    assert gamma.component_roles == "largest_rate_gap"
    assert gamma.entropy_scope == "semantic"
    assert gamma.split_warm_start is True
    assert gamma.hard_max_components > gamma.max_components
    assert not hasattr(gamma, "validation_repeats")
    assert not hasattr(gamma, "max_fit_samples")
    assert not hasattr(gamma, "grouping")
    assert not hasattr(gamma, "background_cut")
    assert not hasattr(extraction, "inherit_0_1_2_tail_probe")
    assert not hasattr(extraction, "border_rule")


def test_single_gamma_component_represents_all_background() -> None:
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(120, 2)).astype(np.float32)
    model = OptimizationStrataSCAN(
        backend="kd_tree",
        n_jobs=1,
        gamma_config=GammaMDLConfig(
            min_components=1,
            max_components=1,
            hard_max_components=1,
            adaptive_components=False,
            n_init=1,
        ),
    ).fit(X)
    assert model.profile_["mdl_gamma_selected_components"] == 1
    assert model.profile_["mdl_gamma_signal_components"] == 0
    assert model.profile_["mdl_gamma_background_fraction"] == 1.0


def test_adaptive_component_search_expands_only_near_the_boundary() -> None:
    assert _search_needs_expansion(8, 8, 16, 1)
    assert _search_needs_expansion(7, 8, 16, 1)
    assert not _search_needs_expansion(6, 8, 16, 1)
    assert not _search_needs_expansion(16, 16, 16, 1)


def test_duplicate_profile_multiplicity_only_counts_selected_bases() -> None:
    from stratascan.optimization import _maximum_duplicate_profile_multiplicity

    mu = np.array([3.0, 2.0, 2.0, 2.0, 1.0])
    slopes = np.array([0.2, -0.1, -0.1, -0.1, 0.2])
    assert (
        _maximum_duplicate_profile_multiplicity(
            mu,
            slopes,
            np.array([1, 2, 3, 4]),
            tolerance=1e-8,
        )
        == 3
    )
    assert (
        _maximum_duplicate_profile_multiplicity(
            mu,
            slopes,
            np.array([0, 1, 4]),
            tolerance=1e-8,
        )
        == 1
    )


def test_gamma_rate_mixture_background_log_density_is_finite() -> None:
    shells = np.array([[0.2, 0.3], [1.0, 1.4], [4.0, 5.0]])
    shapes = np.array([4.0, 4.0])
    log_ranks = np.log(np.array([4.0, 8.0]))
    data_term = np.sum(
        (shapes[None, :] - 1.0) * np.log(shells)
        - np.array([math.lgamma(value) for value in shapes])[None, :],
        axis=1,
    )
    values = _background_log_density(
        shells,
        shapes=shapes,
        log_ranks=log_ranks,
        shape=3.0,
        rate=2.0,
        slope=0.0,
        data_term=data_term,
    )
    assert values.shape == (3,)
    assert np.all(np.isfinite(values))


def test_alternative_rate_background_log_densities_are_finite() -> None:
    shells = np.array([[0.2, 0.3], [1.0, 1.4], [4.0, 5.0]])
    shapes = np.array([4.0, 4.0])
    log_ranks = np.log(np.array([4.0, 8.0]))
    data_term = np.sum(
        (shapes[None, :] - 1.0) * np.log(shells)
        - np.array([math.lgamma(value) for value in shapes])[None, :],
        axis=1,
    )
    lognormal = _lognormal_rate_background_log_density(
        shells,
        shapes=shapes,
        log_ranks=log_ranks,
        location=0.0,
        scale=0.8,
        slope=0.0,
        data_term=data_term,
        quadrature_order=16,
    )
    inverse_gamma = _inverse_gamma_rate_background_log_density(
        shells,
        shapes=shapes,
        log_ranks=log_ranks,
        shape=3.0,
        scale=2.0,
        slope=0.0,
        data_term=data_term,
    )
    assert np.all(np.isfinite(lognormal))
    assert np.all(np.isfinite(inverse_gamma))


def test_lognormal_rate_background_score_matches_finite_differences() -> None:
    shells = np.array([[0.2, 0.3], [1.0, 1.4], [4.0, 5.0]])
    shapes = np.array([4.0, 4.0])
    log_ranks = np.log(np.array([4.0, 8.0]))
    data_term = np.sum(
        (shapes[None, :] - 1.0) * np.log(shells)
        - np.array([math.lgamma(value) for value in shapes])[None, :],
        axis=1,
    )
    parameters = np.array([0.1, 0.8, -0.2])

    def objective(values: np.ndarray) -> float:
        return float(
            np.sum(
                _lognormal_rate_background_log_density(
                    shells,
                    shapes=shapes,
                    log_ranks=log_ranks,
                    location=float(values[0]),
                    scale=float(values[1]),
                    slope=float(values[2]),
                    data_term=data_term,
                    quadrature_order=24,
                )
            )
        )

    _, score = _lognormal_rate_background_density_and_score(
        shells,
        shapes=shapes,
        log_ranks=log_ranks,
        location=float(parameters[0]),
        scale=float(parameters[1]),
        slope=float(parameters[2]),
        data_term=data_term,
        quadrature_order=24,
    )
    expected = np.sum(score, axis=0)
    actual = np.empty(3, dtype=float)
    step = 1e-5
    for index in range(3):
        lower = parameters.copy()
        upper = parameters.copy()
        lower[index] -= step
        upper[index] += step
        actual[index] = (objective(upper) - objective(lower)) / (2.0 * step)
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_continuous_rate_backgrounds_are_explicit_all_background_models() -> None:
    rng = np.random.default_rng(19)
    X = rng.uniform(-1.0, 1.0, size=(100, 2)).astype(np.float32)
    families = (
        "gamma_rate_mixture",
        "lognormal_rate_mixture",
        "inverse_gamma_rate_mixture",
    )
    for family in families:
        model = OptimizationStrataSCAN(
            backend="kd_tree",
            n_jobs=1,
            ambient_dimension=2.0,
            gamma_config=GammaMDLConfig(
                max_components=1,
                hard_max_components=1,
                adaptive_components=False,
                background_distribution=family,
                n_init=1,
            ),
        ).fit(X)
        assert model.profile_["mdl_gamma_background_distribution"] == family
        assert model.profile_["mdl_gamma_signal_components"] == 0
        assert model.profile_["mdl_gamma_background_components"] == 1
        assert model.profile_["mdl_gamma_mean_background_probability"] == 1.0


def test_coded_background_roles_can_avoid_signal_strata_on_uniform_null() -> None:
    rng = np.random.default_rng(17)
    X = rng.uniform(-1.0, 1.0, size=(120, 2)).astype(np.float32)
    model = OptimizationStrataSCAN(
        backend="kd_tree",
        n_jobs=1,
        ambient_dimension=2.0,
        gamma_config=GammaMDLConfig(
            max_components=4,
            hard_max_components=8,
            n_init=1,
            component_roles="background_scan_mdl",
        ),
    ).fit(X)
    assert model.profile_["mdl_gamma_selected_components"] >= 1
    assert model.profile_["mdl_gamma_signal_components"] == 0
    assert model.profile_["mdl_gamma_background_fraction"] == 1.0
    assert model.n_clusters_ == 0


def test_density_ordered_growth_does_not_merge_existing_clusters() -> None:
    graph = KNNGraph(
        indices=np.array(
            [
                [1, 2],
                [0, 2],
                [1, 3],
                [4, 2],
                [5, 3],
                [4, 3],
            ],
            dtype=np.int32,
        ),
        distances=np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 1.5],
                [1.0, 1.5],
                [1.0, 2.0],
                [1.0, 2.0],
            ],
            dtype=np.float32,
        ),
    )
    groups = np.array([0, 0, 1, 1, 0, 0], dtype=np.int32)
    labels, count = _density_ordered_labels_from_core(
        graph, np.ones(6, dtype=bool), np.full(6, 2.0), groups
    )
    assert count == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[5]
