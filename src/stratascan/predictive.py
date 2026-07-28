from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.metrics import adjusted_rand_score

from .core import StrataSCAN
from .multiscale import (
    _quantile_preserving_subsample,
    _shell_volumes,
    multiscale_density_signatures,
)
from .neighbors import Backend, HNSWConfig, build_knn_graph
from .stratification import StratificationResult
from .types import KNNGraph


@dataclass(slots=True)
class PredictiveMultiscaleConfig:
    """Constrained multiscale Gamma mixture selected by predictive likelihood."""

    ranks: tuple[int, ...] = (4, 8, 16, 32)
    min_components: int = 2
    max_components: int = 8
    validation_fraction: float = 0.20
    validation_repeats: int = 3
    selection_n_init: int = 1
    final_n_init: int = 3
    min_component_mass: float = 0.005
    min_assignment_stability: float = 0.80
    min_tail_mass: float = 0.20
    min_dense_mass: float = 0.01
    background_posterior_threshold: float = 0.50
    max_fit_samples: int = 10_000
    max_iter: int = 300
    tolerance: float = 1e-7
    selection_max_iter: int | None = None
    selection_tolerance: float | None = None
    min_rate: float = 1e-14
    max_rate: float = 1e14
    random_state: int = 42
    search_mode: Literal["full", "coarse_refine", "coarse_refine_warm"] = "full"
    coarse_step: int = 2


@dataclass(slots=True)
class _ConstrainedGammaFit:
    log_likelihood: float
    weights: np.ndarray
    mu: np.ndarray
    slopes: np.ndarray
    rates: np.ndarray
    responsibilities: np.ndarray
    iterations: int
    converged: bool


@dataclass(slots=True)
class _GammaWorkspace:
    shells: np.ndarray
    data_term: np.ndarray


def _gamma_workspace(shells: np.ndarray, shapes: np.ndarray) -> _GammaWorkspace:
    shells = np.asarray(shells, dtype=float)
    log_shells = np.log(np.maximum(shells, np.finfo(float).tiny))
    data_term = np.sum(
        (shapes[None, :] - 1.0) * log_shells - gammaln(shapes)[None, :],
        axis=1,
    )
    return _GammaWorkspace(shells=shells, data_term=data_term)


def _validate_config(config: PredictiveMultiscaleConfig) -> None:
    if not 2 <= config.min_components <= config.max_components:
        raise ValueError("predictive component bounds are invalid")
    if not 0.0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    if config.validation_repeats < 2:
        raise ValueError("validation_repeats must be at least 2")
    if config.selection_n_init < 1 or config.final_n_init < 1:
        raise ValueError("numbers of initializations must be positive")
    if not 0.0 < config.min_component_mass < 1.0:
        raise ValueError("min_component_mass must lie in (0, 1)")
    if not 0.0 <= config.min_assignment_stability <= 1.0:
        raise ValueError("min_assignment_stability must lie in [0, 1]")
    if not 0.0 < config.min_tail_mass < 1.0:
        raise ValueError("min_tail_mass must lie in (0, 1)")
    if not 0.0 < config.min_dense_mass < 1.0:
        raise ValueError("min_dense_mass must lie in (0, 1)")
    if not 0.0 < config.background_posterior_threshold < 1.0:
        raise ValueError("background_posterior_threshold must lie in (0, 1)")
    if config.max_fit_samples < 100:
        raise ValueError("max_fit_samples must be at least 100")
    if config.search_mode not in {"full", "coarse_refine", "coarse_refine_warm"}:
        raise ValueError("unknown predictive component search mode")
    if config.coarse_step < 2:
        raise ValueError("coarse_step must be at least 2")
    if config.selection_max_iter is not None and config.selection_max_iter < 1:
        raise ValueError("selection_max_iter must be positive")
    if config.selection_tolerance is not None and config.selection_tolerance <= 0.0:
        raise ValueError("selection_tolerance must be positive")


def _log_probabilities(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    weights: np.ndarray,
    rates: np.ndarray,
    workspace: _GammaWorkspace | None = None,
) -> np.ndarray:
    cached = workspace or _gamma_workspace(shells, shapes)
    if cached.shells.shape != np.asarray(shells).shape:
        raise ValueError("Gamma workspace does not match shells")
    component_term = (
        np.log(np.maximum(weights, np.finfo(float).tiny))
        + np.sum(shapes[None, :] * np.log(rates), axis=1)
    )
    return (
        cached.data_term[:, None]
        + component_term[None, :]
        - cached.shells @ rates.T
    )


def _responsibilities(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    weights: np.ndarray,
    rates: np.ndarray,
    workspace: _GammaWorkspace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    log_probability = _log_probabilities(
        shells,
        shapes=shapes,
        weights=weights,
        rates=rates,
        workspace=workspace,
    )
    log_norm = logsumexp(log_probability, axis=1)
    return np.exp(log_probability - log_norm[:, None]), log_norm


def _fit_rate_profiles(
    weighted_shells: np.ndarray,
    effective: np.ndarray,
    *,
    shapes: np.ndarray,
    log_ranks: np.ndarray,
    config: PredictiveMultiscaleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact M-step for log(rate_sj) = mu_s + slope_s log(rank_j)."""

    target = float(np.sum(shapes * log_ranks) / np.sum(shapes))
    low = np.full(effective.size, -20.0, dtype=float)
    high = np.full(effective.size, 20.0, dtype=float)
    safe_shells = np.maximum(weighted_shells, np.finfo(float).tiny)
    for _ in range(60):
        midpoint = 0.5 * (low + high)
        scaled = safe_shells * np.exp(midpoint[:, None] * log_ranks[None, :])
        predicted = np.sum(scaled * log_ranks[None, :], axis=1) / np.sum(
            scaled, axis=1
        )
        move_right = predicted < target
        low = np.where(move_right, midpoint, low)
        high = np.where(move_right, high, midpoint)
    slopes = 0.5 * (low + high)
    normalizer = np.sum(
        safe_shells * np.exp(slopes[:, None] * log_ranks[None, :]), axis=1
    )
    mu = np.log(np.maximum(effective * np.sum(shapes), np.finfo(float).tiny)) - np.log(
        np.maximum(normalizer, np.finfo(float).tiny)
    )
    rates = np.exp(mu[:, None] + slopes[:, None] * log_ranks[None, :])
    rates = np.clip(rates, config.min_rate, config.max_rate)
    return mu, slopes, rates


def _initial_labels(
    shells: np.ndarray,
    components: int,
    restart: int,
    random_state: int,
) -> np.ndarray:
    features = np.log(np.maximum(shells, np.finfo(float).tiny))
    features = (features - np.mean(features, axis=0)) / np.maximum(
        np.std(features, axis=0), 1e-8
    )
    if restart == 0:
        score = np.log(np.sum(shells, axis=1))
    else:
        direction = np.random.default_rng(random_state + restart).normal(
            size=features.shape[1]
        )
        score = features @ direction
    order = np.argsort(score, kind="stable")
    labels = np.empty(shells.shape[0], dtype=np.int32)
    labels[order] = np.minimum(
        components - 1,
        np.floor(np.arange(shells.shape[0]) * components / shells.shape[0]).astype(
            np.int32
        ),
    )
    return labels


def _fit_once(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    ranks: tuple[int, ...],
    components: int,
    restart: int,
    config: PredictiveMultiscaleConfig,
    initial_responsibilities: np.ndarray | None = None,
    max_iter: int | None = None,
    tolerance: float | None = None,
    workspace: _GammaWorkspace | None = None,
) -> _ConstrainedGammaFit:
    if initial_responsibilities is None:
        labels = _initial_labels(shells, components, restart, config.random_state)
        responsibilities = np.eye(components, dtype=float)[labels]
    else:
        responsibilities = np.asarray(initial_responsibilities, dtype=float).copy()
        if responsibilities.shape != (shells.shape[0], components):
            raise ValueError("warm-start responsibilities have the wrong shape")
        row_sums = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities /= np.maximum(row_sums, np.finfo(float).tiny)
    log_ranks = np.log(np.asarray(ranks, dtype=float))
    previous = -np.inf
    converged = False
    log_likelihood = -np.inf
    iteration_limit = config.max_iter if max_iter is None else int(max_iter)
    convergence_tolerance = config.tolerance if tolerance is None else float(tolerance)
    cached = workspace or _gamma_workspace(shells, shapes)
    for iteration in range(iteration_limit):
        effective = np.maximum(
            np.sum(responsibilities, axis=0), np.finfo(float).tiny
        )
        weights = effective / shells.shape[0]
        weighted_shells = responsibilities.T @ shells
        mu, slopes, rates = _fit_rate_profiles(
            weighted_shells,
            effective,
            shapes=shapes,
            log_ranks=log_ranks,
            config=config,
        )
        responsibilities, log_norm = _responsibilities(
            shells,
            shapes=shapes,
            weights=weights,
            rates=rates,
            workspace=cached,
        )
        log_likelihood = float(np.sum(log_norm))
        if np.isfinite(previous) and abs(log_likelihood - previous) <= convergence_tolerance * (
            1.0 + abs(previous)
        ):
            converged = True
            break
        previous = log_likelihood

    order = np.argsort(rates[:, 0], kind="stable")[::-1]
    return _ConstrainedGammaFit(
        log_likelihood=log_likelihood,
        weights=weights[order],
        mu=mu[order],
        slopes=slopes[order],
        rates=rates[order],
        responsibilities=responsibilities[:, order],
        iterations=iteration + 1,
        converged=converged,
    )


def _fit_constrained_gamma(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    ranks: tuple[int, ...],
    components: int,
    n_init: int,
    config: PredictiveMultiscaleConfig,
    max_iter: int | None = None,
    tolerance: float | None = None,
    workspace: _GammaWorkspace | None = None,
) -> _ConstrainedGammaFit:
    fits = [
        _fit_once(
            shells,
            shapes=shapes,
            ranks=ranks,
            components=components,
            restart=restart,
            config=config,
            max_iter=max_iter,
            tolerance=tolerance,
            workspace=workspace,
        )
        for restart in range(n_init)
    ]
    return max(fits, key=lambda fit: fit.log_likelihood)


def _split_responsibilities(
    responsibilities: np.ndarray,
    shells: np.ndarray,
    target_components: int,
) -> np.ndarray:
    """Grow a fitted mixture by deterministically splitting heterogeneous parts."""

    grown = np.asarray(responsibilities, dtype=float).copy()
    score = np.log(np.maximum(np.sum(shells, axis=1), np.finfo(float).tiny))
    while grown.shape[1] < target_components:
        effective = np.sum(grown, axis=0)
        means = np.sum(grown * score[:, None], axis=0) / np.maximum(
            effective, np.finfo(float).tiny
        )
        variance = np.sum(
            grown * (score[:, None] - means[None, :]) ** 2, axis=0
        ) / np.maximum(effective, np.finfo(float).tiny)
        component = int(np.argmax(effective * variance))
        members = grown[:, component]
        order = np.argsort(score, kind="stable")
        cumulative = np.cumsum(members[order])
        midpoint = 0.5 * effective[component]
        split_at = int(np.searchsorted(cumulative, midpoint, side="left"))
        threshold = score[order[min(split_at, order.size - 1)]]
        fraction = np.where(score <= threshold, 1.0, 0.0)
        left = members * fraction
        right = members - left
        if min(np.sum(left), np.sum(right)) <= np.finfo(float).eps:
            fraction = np.arange(score.size) % 2 == 0
            left = members * fraction
            right = members - left
        grown[:, component] = left
        grown = np.column_stack([grown, right])
    return grown


def _validation_splits(
    n_samples: int,
    repeats: int,
    fraction: float,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    validation_size = max(1, int(round(fraction * n_samples)))
    splits = []
    for repeat in range(repeats):
        order = np.random.default_rng(random_state + 10_000 + repeat).permutation(n_samples)
        validation = np.sort(order[:validation_size])
        training = np.sort(order[validation_size:])
        splits.append((training, validation))
    return splits


def _mean_pairwise_stability(assignments: list[np.ndarray]) -> float:
    pairs = list(combinations(assignments, 2))
    if not pairs:
        return 1.0
    return float(np.mean([adjusted_rand_score(left, right) for left, right in pairs]))


def estimate_predictive_multiscale_stratification(
    graph: KNNGraph,
    *,
    ambient_dimension: float,
    config: PredictiveMultiscaleConfig | None = None,
) -> StratificationResult:
    """Select a constrained multiscale Gamma mixture by held-out likelihood."""

    cfg = config or PredictiveMultiscaleConfig()
    _validate_config(cfg)
    dimension = float(ambient_dimension)
    if not np.isfinite(dimension) or dimension <= 0.0:
        raise ValueError("ambient_dimension must be positive and finite")

    started = perf_counter()
    signatures = multiscale_density_signatures(graph, cfg.ranks)
    shells, shapes = _shell_volumes(graph, cfg.ranks, dimension)
    signature_seconds = perf_counter() - started
    fit_indices = _quantile_preserving_subsample(
        signatures, cfg.max_fit_samples, cfg.random_state
    )
    fit_shells = shells[fit_indices]
    splits = _validation_splits(
        fit_shells.shape[0],
        cfg.validation_repeats,
        cfg.validation_fraction,
        cfg.random_state,
    )
    fit_workspace = _gamma_workspace(fit_shells, shapes)
    training_workspaces = [
        _gamma_workspace(fit_shells[training], shapes) for training, _ in splits
    ]
    validation_workspaces = [
        _gamma_workspace(fit_shells[validation], shapes) for _, validation in splits
    ]

    fit_started = perf_counter()
    all_components = np.arange(cfg.min_components, cfg.max_components + 1)
    fold_fits: list[dict[int, _ConstrainedGammaFit]] = [dict() for _ in splits]
    candidate_records: dict[int, dict[str, object]] = {}
    fit_count = 0

    def evaluate_candidate(components: int) -> None:
        nonlocal fit_count
        scores: list[float] = []
        assignments: list[np.ndarray] = []
        masses: list[float] = []
        iterations: list[int] = []
        for fold, (training, validation) in enumerate(splits):
            warm = cfg.search_mode == "coarse_refine_warm"
            lower = [value for value in fold_fits[fold] if value < components]
            if warm and lower:
                source = fold_fits[fold][max(lower)]
                initial = _split_responsibilities(
                    source.responsibilities,
                    fit_shells[training],
                    components,
                )
                fit = _fit_once(
                    fit_shells[training],
                    shapes=shapes,
                    ranks=cfg.ranks,
                    components=components,
                    restart=0,
                    config=cfg,
                    initial_responsibilities=initial,
                    max_iter=cfg.selection_max_iter,
                    tolerance=cfg.selection_tolerance,
                    workspace=training_workspaces[fold],
                )
                fit_count += 1
            else:
                fit = _fit_constrained_gamma(
                    fit_shells[training],
                    shapes=shapes,
                    ranks=cfg.ranks,
                    components=components,
                    n_init=cfg.selection_n_init,
                    config=cfg,
                    max_iter=cfg.selection_max_iter,
                    tolerance=cfg.selection_tolerance,
                    workspace=training_workspaces[fold],
                )
                fit_count += cfg.selection_n_init
            fold_fits[fold][components] = fit
            _, validation_log_norm = _responsibilities(
                fit_shells[validation],
                shapes=shapes,
                weights=fit.weights,
                rates=fit.rates,
                workspace=validation_workspaces[fold],
            )
            full_responsibilities, _ = _responsibilities(
                fit_shells,
                shapes=shapes,
                weights=fit.weights,
                rates=fit.rates,
                workspace=fit_workspace,
            )
            scores.append(float(np.mean(validation_log_norm)))
            assignments.append(np.argmax(full_responsibilities, axis=1))
            masses.append(float(np.min(fit.weights)))
            iterations.append(int(fit.iterations))
        candidate_records[components] = {
            "mean": float(np.mean(scores)),
            "error": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
            "stability": _mean_pairwise_stability(assignments),
            "minimum_mass": float(np.min(masses)),
            "valid": bool(np.min(masses) >= cfg.min_component_mass),
            "iterations": iterations,
        }

    if cfg.search_mode == "full":
        initial_grid = all_components.tolist()
    else:
        initial_grid = list(
            range(cfg.min_components, cfg.max_components + 1, cfg.coarse_step)
        )
        if initial_grid[-1] != cfg.max_components:
            initial_grid.append(cfg.max_components)
    for components in initial_grid:
        evaluate_candidate(int(components))

    if cfg.search_mode != "full":
        coarse_means = np.array(
            [candidate_records[value]["mean"] for value in initial_grid], dtype=float
        )
        coarse_errors = np.array(
            [candidate_records[value]["error"] for value in initial_grid], dtype=float
        )
        coarse_valid = np.array(
            [candidate_records[value]["valid"] for value in initial_grid], dtype=bool
        )
        coarse_stable = np.array(
            [
                candidate_records[value]["stability"]
                >= cfg.min_assignment_stability
                for value in initial_grid
            ],
            dtype=bool,
        )
        coarse_eligible = coarse_valid & coarse_stable
        if not np.any(coarse_eligible):
            coarse_eligible = coarse_valid
        if not np.any(coarse_eligible):
            coarse_eligible = np.ones(len(initial_grid), dtype=bool)
        coarse_best_index = int(
            np.flatnonzero(coarse_eligible)[np.argmax(coarse_means[coarse_eligible])]
        )
        coarse_best = initial_grid[coarse_best_index]
        coarse_threshold = coarse_means[coarse_best_index] - coarse_errors[coarse_best_index]
        anchors = [
            value
            for index, value in enumerate(initial_grid)
            if coarse_eligible[index] and coarse_means[index] >= coarse_threshold
        ]
        if not anchors:
            anchors = [coarse_best]
        refinements = sorted(
            {
                value
                for anchor in anchors
                for value in (anchor - 1, anchor + 1)
                if cfg.min_components <= value <= cfg.max_components
                and value not in candidate_records
            }
        )
        for components in refinements:
            evaluate_candidate(int(components))

    components_grid = np.array(sorted(candidate_records), dtype=int)
    mean_scores = [candidate_records[value]["mean"] for value in components_grid]
    standard_errors = [candidate_records[value]["error"] for value in components_grid]
    stabilities = [candidate_records[value]["stability"] for value in components_grid]
    minimum_masses = [candidate_records[value]["minimum_mass"] for value in components_grid]
    valid_candidates = [candidate_records[value]["valid"] for value in components_grid]
    fold_iterations = [candidate_records[value]["iterations"] for value in components_grid]

    means = np.asarray(mean_scores)
    errors = np.asarray(standard_errors)
    stable = np.asarray(stabilities) >= cfg.min_assignment_stability
    selection_eligible = np.asarray(valid_candidates) & stable
    valid = selection_eligible.copy()
    if not np.any(valid):
        valid = np.asarray(valid_candidates)
    if not np.any(valid):
        valid = np.ones(components_grid.size, dtype=bool)
    best_index = int(np.flatnonzero(valid)[np.argmax(means[valid])])
    threshold = means[best_index] - errors[best_index]
    within_one_se = valid & (means >= threshold)
    selected_index = int(np.flatnonzero(within_one_se)[0])
    selected_components = int(components_grid[selected_index])

    if cfg.search_mode == "coarse_refine_warm":
        source = max(
            (fits[selected_components] for fits in fold_fits),
            key=lambda fit: fit.log_likelihood,
        )
        initial, _ = _responsibilities(
            fit_shells,
            shapes=shapes,
            weights=source.weights,
            rates=source.rates,
            workspace=fit_workspace,
        )
        selected = _fit_once(
            fit_shells,
            shapes=shapes,
            ranks=cfg.ranks,
            components=selected_components,
            restart=0,
            config=cfg,
            initial_responsibilities=initial,
            workspace=fit_workspace,
        )
        fit_count += 1
    else:
        selected = _fit_constrained_gamma(
            fit_shells,
            shapes=shapes,
            ranks=cfg.ranks,
            components=selected_components,
            n_init=cfg.final_n_init,
            config=cfg,
            workspace=fit_workspace,
        )
        fit_count += cfg.final_n_init
    if fit_indices.size == graph.n_samples:
        responsibilities = selected.responsibilities
    else:
        responsibilities, _ = _responsibilities(
            shells,
            shapes=shapes,
            weights=selected.weights,
            rates=selected.rates,
            workspace=_gamma_workspace(shells, shapes),
        )

    density_rates = selected.rates[:, 0]
    gap_ratios = density_rates[:-1] / np.maximum(density_rates[1:], cfg.min_rate)
    tail_masses = np.asarray(
        [np.sum(selected.weights[index + 1 :]) for index in range(selected_components - 1)]
    )
    dense_masses = 1.0 - tail_masses
    eligible = np.flatnonzero(
        (tail_masses >= cfg.min_tail_mass) & (dense_masses >= cfg.min_dense_mass)
    )
    if eligible.size:
        cut = int(eligible[np.argmax(gap_ratios[eligible])] + 1)
    else:
        cut = int(max(1, selected_components - 1))

    background_probability = np.sum(responsibilities[:, cut:], axis=1)
    active = background_probability < cfg.background_posterior_threshold
    dense_group = np.argmax(responsibilities[:, :cut], axis=1).astype(np.int32)
    groups = np.where(active, dense_group, cut).astype(np.int32)
    entropy = -np.sum(
        responsibilities * np.log(np.maximum(responsibilities, np.finfo(float).tiny)),
        axis=1,
    )
    fit_seconds = perf_counter() - fit_started
    parameter_count = (selected_components - 1) + 2 * selected_components
    bic = -2.0 * selected.log_likelihood + parameter_count * np.log(fit_shells.shape[0])
    diagnostics = {
        "predictive_ranks": list(map(int, cfg.ranks)),
        "predictive_search_mode": cfg.search_mode,
        "predictive_fit_count": int(fit_count),
        "predictive_coarse_step": int(cfg.coarse_step),
        "predictive_cached_gamma_workspace": True,
        "predictive_shell_shapes": shapes.astype(int).tolist(),
        "predictive_components": selected_components,
        "predictive_supported_components": cut,
        "predictive_background_components": selected_components - cut,
        "predictive_background_fraction": float(np.mean(~active)),
        "predictive_weights": selected.weights.tolist(),
        "predictive_mu": selected.mu.tolist(),
        "predictive_slopes": selected.slopes.tolist(),
        "predictive_rates": selected.rates.tolist(),
        "predictive_density_gap_ratios": gap_ratios.tolist(),
        "predictive_tail_masses": tail_masses.tolist(),
        "predictive_candidate_components": components_grid.tolist(),
        "predictive_validation_log_likelihood": means.tolist(),
        "predictive_validation_standard_errors": errors.tolist(),
        "predictive_assignment_stability": stabilities,
        "predictive_minimum_component_masses": minimum_masses,
        "predictive_valid_candidates": valid_candidates,
        "predictive_stable_candidates": stable.tolist(),
        "predictive_selection_eligible": selection_eligible.tolist(),
        "predictive_within_one_se": within_one_se.tolist(),
        "predictive_best_components": int(components_grid[best_index]),
        "predictive_fold_iterations": fold_iterations,
        "predictive_final_iterations": int(selected.iterations),
        "predictive_final_converged": bool(selected.converged),
        "predictive_mean_posterior_entropy": float(np.mean(entropy)),
        "predictive_fit_samples": int(fit_indices.size),
    }
    return StratificationResult(
        signatures=signatures,
        groups=groups,
        eps_by_group=np.zeros(cut + 1, dtype=np.float32),
        point_eps=np.zeros(graph.n_samples, dtype=np.float32),
        supported_groups=np.arange(cut, dtype=np.int32),
        selected_components=cut + 1,
        bic=float(bic),
        fit_indices=fit_indices,
        timings={
            "density_signatures": signature_seconds,
            "predictive_multiscale_gamma": fit_seconds,
        },
        gap_ratios=gap_ratios[:cut].astype(np.float32),
        selection_method="repeated_holdout_one_se_constrained_multiscale_gamma",
        diagnostics=diagnostics,
        mode="predictive_constrained_multiscale_gamma_uniform_tail",
        background_probability=background_probability.astype(np.float32),
    )


class PredictiveMultiscaleStrataSCAN(StrataSCAN):
    """StrataSCAN 0.1.2 with predictive selection of density strata."""

    def __init__(
        self,
        *,
        predictive_config: PredictiveMultiscaleConfig | None = None,
        k: int = 32,
        backend: Backend = "auto",
        n_jobs: int = 1,
        ambient_dimension: float | None = None,
        hnsw_config: HNSWConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            k=k,
            backend=backend,
            n_jobs=n_jobs,
            ambient_dimension=ambient_dimension,
            hnsw_config=hnsw_config,
            **kwargs,
        )
        self.predictive_config = predictive_config or PredictiveMultiscaleConfig()

    def fit_from_graph(
        self,
        graph: KNNGraph,
        *,
        ambient_dimension: float | None = None,
    ) -> "PredictiveMultiscaleStrataSCAN":
        from .strict_core import gamma_strict_core_from_graph

        dimension = self.ambient_dimension if ambient_dimension is None else ambient_dimension
        if dimension is None:
            dimension = 1.0
        stratification = estimate_predictive_multiscale_stratification(
            graph,
            ambient_dimension=float(dimension),
            config=self.predictive_config,
        )
        result = gamma_strict_core_from_graph(
            graph,
            ambient_dimension=float(dimension),
            config=self._config(),
            stratification=stratification,
        )
        self.labels_ = result.labels
        self.core_sample_indices_ = np.flatnonzero(result.core_mask)
        self.n_clusters_ = int(np.unique(result.labels[result.labels >= 0]).size)
        self.stratification_ = stratification
        self.profile_ = {
            "algorithm_version": "0.1.2",
            "algorithm": "PredictiveMultiscaleStrataSCAN",
            "stratification": "predictive-constrained-multiscale-density-v1",
            **result.profile,
        }
        self.graph_ = graph
        return self

    def fit(self, X, y=None) -> "PredictiveMultiscaleStrataSCAN":
        del y
        values = np.asarray(X, dtype=np.float32, order="C")
        if values.ndim != 2:
            raise ValueError("X must be a 2-D array")
        graph, graph_seconds = build_knn_graph(
            values,
            k=self.k,
            backend=self.backend,
            n_jobs=self.n_jobs,
            hnsw_config=self.hnsw_config,
        )
        dimension = self.ambient_dimension
        if dimension is None:
            dimension = float(values.shape[1])
        self.fit_from_graph(graph, ambient_dimension=dimension)
        self.profile_ = {"graph_seconds": float(graph_seconds), **self.profile_}
        return self
