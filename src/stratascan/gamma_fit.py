from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Literal
from numba import njit
import numpy as np
from scipy.special import gammaln
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
    slopes = np.zeros(effective.size, dtype=float)
    # The score is monotone in the slope.  Safeguarded Newton updates normally
    # converge in a handful of steps; retaining the bracket gives bisection's
    # robustness when a proposal is numerically unsuitable.
    for _ in range(12):
        scaled = safe_shells * np.exp(slopes[:, None] * log_ranks[None, :])
        totals = np.sum(scaled, axis=1)
        predicted = np.sum(scaled * log_ranks[None, :], axis=1) / totals
        if np.all(np.abs(predicted - target) <= 1e-12):
            break
        move_right = predicted < target
        low = np.where(move_right, slopes, low)
        high = np.where(move_right, high, slopes)
        derivative = np.sum(
            scaled * (log_ranks[None, :] - predicted[:, None]) ** 2,
            axis=1,
        ) / totals
        proposal = slopes - (predicted - target) / np.maximum(
            derivative, np.finfo(float).tiny
        )
        midpoint = 0.5 * (low + high)
        valid = np.isfinite(proposal) & (proposal > low) & (proposal < high)
        slopes = np.where(valid, proposal, midpoint)
    normalizer = np.sum(
        safe_shells * np.exp(slopes[:, None] * log_ranks[None, :]), axis=1
    )
    mu = np.log(np.maximum(effective * np.sum(shapes), np.finfo(float).tiny)) - np.log(
        np.maximum(normalizer, np.finfo(float).tiny)
    )
    rates = np.exp(mu[:, None] + slopes[:, None] * log_ranks[None, :])
    rates = np.clip(rates, config.min_rate, config.max_rate)
    return mu, slopes, rates
@njit(cache=True)
def _fit_once_compiled(
    shells: np.ndarray,
    data_term: np.ndarray,
    shapes: np.ndarray,
    log_ranks: np.ndarray,
    responsibilities: np.ndarray,
    iteration_limit: int,
    convergence_tolerance: float,
    min_rate: float,
    max_rate: float,
) -> tuple:
    """Run constrained-Gamma EM without allocating per-iteration matrices."""

    n_samples, n_shells = shells.shape
    components = responsibilities.shape[1]
    tiny = np.finfo(np.float64).tiny
    total_shape = 0.0
    target_numerator = 0.0
    for shell in range(n_shells):
        total_shape += shapes[shell]
        target_numerator += shapes[shell] * log_ranks[shell]
    target = target_numerator / total_shape

    effective = np.empty(components, dtype=np.float64)
    weights = np.empty(components, dtype=np.float64)
    weighted_shells = np.empty((components, n_shells), dtype=np.float64)
    mu = np.empty(components, dtype=np.float64)
    slopes = np.empty(components, dtype=np.float64)
    rates = np.empty((components, n_shells), dtype=np.float64)
    component_term = np.empty(components, dtype=np.float64)
    previous = -np.inf
    log_likelihood = -np.inf
    converged = False

    for iteration in range(iteration_limit):
        for component in range(components):
            effective[component] = 0.0
            for shell in range(n_shells):
                weighted_shells[component, shell] = 0.0
        for row in range(n_samples):
            for component in range(components):
                probability = responsibilities[row, component]
                effective[component] += probability
                for shell in range(n_shells):
                    weighted_shells[component, shell] += probability * shells[row, shell]

        for component in range(components):
            if effective[component] < tiny:
                effective[component] = tiny
            weights[component] = effective[component] / n_samples
            low = -20.0
            high = 20.0
            slope = 0.0
            for _ in range(12):
                total = 0.0
                predicted_numerator = 0.0
                for shell in range(n_shells):
                    safe_shell = max(weighted_shells[component, shell], tiny)
                    scaled = safe_shell * math.exp(slope * log_ranks[shell])
                    total += scaled
                    predicted_numerator += scaled * log_ranks[shell]
                predicted = predicted_numerator / total
                if abs(predicted - target) <= 1e-12:
                    break
                if predicted < target:
                    low = slope
                else:
                    high = slope
                derivative_numerator = 0.0
                for shell in range(n_shells):
                    safe_shell = max(weighted_shells[component, shell], tiny)
                    scaled = safe_shell * math.exp(slope * log_ranks[shell])
                    difference = log_ranks[shell] - predicted
                    derivative_numerator += scaled * difference * difference
                derivative = max(derivative_numerator / total, tiny)
                proposal = slope - (predicted - target) / derivative
                if math.isfinite(proposal) and proposal > low and proposal < high:
                    slope = proposal
                else:
                    slope = 0.5 * (low + high)
            slopes[component] = slope
            normalizer = 0.0
            for shell in range(n_shells):
                safe_shell = max(weighted_shells[component, shell], tiny)
                normalizer += safe_shell * math.exp(slope * log_ranks[shell])
            value_mu = math.log(max(effective[component] * total_shape, tiny)) - math.log(
                max(normalizer, tiny)
            )
            mu[component] = value_mu
            rate_log_term = 0.0
            for shell in range(n_shells):
                rate = math.exp(value_mu + slope * log_ranks[shell])
                rate = min(max(rate, min_rate), max_rate)
                rates[component, shell] = rate
                rate_log_term += shapes[shell] * math.log(rate)
            component_term[component] = math.log(max(weights[component], tiny)) + rate_log_term

        log_likelihood = 0.0
        for row in range(n_samples):
            maximum = -np.inf
            for component in range(components):
                log_probability = data_term[row] + component_term[component]
                for shell in range(n_shells):
                    log_probability -= shells[row, shell] * rates[component, shell]
                responsibilities[row, component] = log_probability
                maximum = max(maximum, log_probability)
            normalizer = 0.0
            for component in range(components):
                probability = math.exp(responsibilities[row, component] - maximum)
                responsibilities[row, component] = probability
                normalizer += probability
            log_likelihood += maximum + math.log(normalizer)
            for component in range(components):
                responsibilities[row, component] /= normalizer

        if math.isfinite(previous) and abs(log_likelihood - previous) <= convergence_tolerance * (
            1.0 + abs(previous)
        ):
            converged = True
            break
        previous = log_likelihood

    return (
        log_likelihood,
        weights,
        mu,
        slopes,
        rates,
        responsibilities,
        iteration + 1,
        converged,
    )
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
    iteration_limit = config.max_iter if max_iter is None else int(max_iter)
    convergence_tolerance = config.tolerance if tolerance is None else float(tolerance)
    cached = workspace or _gamma_workspace(shells, shapes)
    (
        log_likelihood,
        weights,
        mu,
        slopes,
        rates,
        responsibilities,
        iterations,
        converged,
    ) = _fit_once_compiled(
        np.asarray(cached.shells, dtype=np.float64),
        np.asarray(cached.data_term, dtype=np.float64),
        np.asarray(shapes, dtype=np.float64),
        log_ranks,
        np.asarray(responsibilities, dtype=np.float64),
        iteration_limit,
        convergence_tolerance,
        config.min_rate,
        config.max_rate,
    )
    order = np.argsort(rates[:, 0], kind="stable")[::-1]
    return _ConstrainedGammaFit(
        log_likelihood=log_likelihood,
        weights=weights[order],
        mu=mu[order],
        slopes=slopes[order],
        rates=rates[order],
        responsibilities=responsibilities[:, order],
        iterations=iterations,
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
