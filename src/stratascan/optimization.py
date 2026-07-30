from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.special import betainc, digamma, gammainc, gammaln, kve, logsumexp

from .core import StrataSCAN as GraphStrataSCAN
from .multiscale import _shell_volumes, multiscale_density_signatures
from .neighbors import Backend, HNSWConfig
from .predictive import (
    PredictiveMultiscaleConfig,
    _fit_constrained_gamma,
    _fit_once,
    _fit_rate_profiles,
    _gamma_workspace,
    _split_responsibilities,
)
from .stratification import StratificationResult
from .types import KNNGraph

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*_args, **_kwargs):
        def decorate(function):
            return function

        return decorate


@dataclass(frozen=True, slots=True)
class GammaMDLConfig:
    """Full-data multiscale Gamma model selection with explicit background.

    Neighbour ranks define the representation.  Component bounds are search
    limits, while the information criterion decides how many density bases are
    retained.  Background bases are aggregated into one semantic state rather
    than reported as scientific strata.  Experimental continuous-rate
    background families can replace those discrete background bases while the
    signal states retain the same Gamma shell model.  Every observation
    participates in every candidate fit.
    """

    ranks: tuple[int, ...] = (4, 8, 16, 32)
    min_components: int = 1
    max_components: int = 8
    hard_max_components: int = 24
    adaptive_components: bool = True
    expansion_step: int = 4
    boundary_margin: int = 1
    criterion: Literal["bic", "aicc", "icl"] = "icl"
    entropy_scope: Literal["semantic", "basis"] = "semantic"
    background_distribution: Literal[
        "gamma_rate_mixture",
        "lognormal_rate_mixture",
        "inverse_gamma_rate_mixture",
        "gamma_components",
    ] = "gamma_components"
    split_warm_start: bool = True
    component_roles: Literal[
        "validated_rate_prefix",
        "background_scan_mdl",
        "rate_changepoint_mdl",
        "topology_mdl",
        "largest_rate_gap",
    ] = "largest_rate_gap"
    role_rank: int = 4
    n_init: int = 3
    max_iter: int = 300
    tolerance: float = 1e-7
    min_rate: float = 1e-14
    max_rate: float = 1e14
    min_background_shape: float = 1e-3
    max_background_shape: float = 1e6
    min_background_scale: float = 1e-3
    max_background_scale: float = 3.0
    max_background_abs_slope: float = 4.0
    background_quadrature_order: int = 24
    background_m_step_max_iter: int = 40
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class OptimizationStrictCoreConfig:
    """Structural controls for unified Gamma-stratum StrictCore extraction."""

    core_rank: int = 4
    connectivity_rank: int = 4


@dataclass(frozen=True, slots=True)
class OptimizationStrictCoreResult:
    labels: np.ndarray
    core_mask: np.ndarray
    profile: dict[str, object]


def _validate_gamma_config(graph: KNNGraph, config: GammaMDLConfig) -> None:
    ranks = np.asarray(config.ranks, dtype=np.int64)
    if ranks.ndim != 1 or ranks.size < 2:
        raise ValueError("at least two Gamma shell ranks are required")
    if ranks[0] < 1 or ranks[-1] > graph.k or np.any(np.diff(ranks) <= 0):
        raise ValueError("Gamma shell ranks must be strictly increasing within the graph")
    if not 1 <= config.min_components <= config.max_components:
        raise ValueError("Gamma component search bounds are invalid")
    if config.hard_max_components < config.max_components:
        raise ValueError("hard_max_components must not be below max_components")
    if config.expansion_step < 1 or config.boundary_margin < 0:
        raise ValueError("adaptive Gamma search controls are invalid")
    if config.criterion not in {"bic", "aicc", "icl"}:
        raise ValueError("unknown Gamma information criterion")
    if config.entropy_scope not in {"semantic", "basis"}:
        raise ValueError("unknown Gamma entropy scope")
    if config.background_distribution not in {
        "gamma_rate_mixture",
        "lognormal_rate_mixture",
        "inverse_gamma_rate_mixture",
        "gamma_components",
    }:
        raise ValueError("unknown Gamma background distribution")
    if config.component_roles not in {
        "validated_rate_prefix",
        "background_scan_mdl",
        "rate_changepoint_mdl",
        "topology_mdl",
        "largest_rate_gap",
    }:
        raise ValueError("unknown Gamma component-role model")
    if config.role_rank < 1 or (
        config.component_roles == "topology_mdl" and 3 * config.role_rank > graph.k
    ):
        raise ValueError("role_rank needs three disjoint windows in the k-NN graph")
    if config.n_init < 1 or config.max_iter < 1 or config.tolerance <= 0.0:
        raise ValueError("Gamma solver controls must be positive")
    if not 0.0 < config.min_background_shape < config.max_background_shape:
        raise ValueError("background shape bounds are invalid")
    if not 0.0 < config.min_background_scale < config.max_background_scale:
        raise ValueError("background scale bounds are invalid")
    if config.max_background_abs_slope <= 0.0:
        raise ValueError("background slope bound must be positive")
    if config.background_quadrature_order < 4:
        raise ValueError("background quadrature order must be at least four")
    if config.background_m_step_max_iter < 1:
        raise ValueError("background M-step iteration limit must be positive")


def _information_score(
    log_likelihood: float,
    responsibilities: np.ndarray,
    parameter_count: int,
    criterion: str,
) -> float:
    """Return a minimization score in nats for one fitted mixture."""

    n = int(responsibilities.shape[0])
    deviance = -2.0 * float(log_likelihood)
    if criterion == "bic":
        return deviance + parameter_count * math.log(n)
    if criterion == "aicc":
        denominator = n - parameter_count - 1
        if denominator <= 0:
            return math.inf
        return (
            deviance
            + 2.0 * parameter_count
            + 2.0 * parameter_count * (parameter_count + 1) / denominator
        )
    entropy = -float(
        np.sum(
            responsibilities
            * np.log(np.maximum(responsibilities, np.finfo(float).tiny))
        )
    )
    return deviance + parameter_count * math.log(n) + 2.0 * entropy


@dataclass(slots=True)
class _HeterogeneousBackgroundGammaFit:
    """Gamma signal strata plus one continuous-rate background state."""

    log_likelihood: float
    weights: np.ndarray
    mu: np.ndarray
    slopes: np.ndarray
    rates: np.ndarray
    background_parameters: tuple[float, float, float]
    responsibilities: np.ndarray
    iterations: int
    converged: bool


def _background_log_density(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    log_ranks: np.ndarray,
    shape: float,
    rate: float,
    slope: float,
    data_term: np.ndarray,
) -> np.ndarray:
    """Log density after integrating a Gamma-distributed local Poisson rate.

    Conditional on a point-specific base rate ``lambda``, shell ``j`` remains
    Gamma with the exact Poisson shell shape and rate
    ``lambda * rank_j**slope``.  Integrating ``lambda`` against a Gamma law
    gives one heavy-tailed background state that can represent a continuum of
    local intensities without turning each intensity band into a stratum.
    """

    total_shape = float(np.sum(shapes))
    rank_scale = np.exp(np.clip(slope * log_ranks, -50.0, 50.0))
    exposure = shells @ rank_scale
    normalizer = np.maximum(rate + exposure, np.finfo(float).tiny)
    return (
        data_term
        + slope * float(np.sum(shapes * log_ranks))
        + shape * math.log(rate)
        - float(gammaln(shape))
        + float(gammaln(shape + total_shape))
        - (shape + total_shape) * np.log(normalizer)
    )


def _initial_background_parameters(
    shells: np.ndarray,
    responsibilities: np.ndarray,
    shapes: np.ndarray,
    config: GammaMDLConfig,
) -> tuple[float, float, float]:
    total_shape = float(np.sum(shapes))
    local_rate = total_shape / np.maximum(
        np.sum(shells, axis=1), np.finfo(float).tiny
    )
    weights = np.asarray(responsibilities, dtype=float)
    effective = max(float(np.sum(weights)), np.finfo(float).tiny)
    mean = float(np.sum(weights * local_rate) / effective)
    variance = float(np.sum(weights * (local_rate - mean) ** 2) / effective)
    shape = mean * mean / max(variance, mean * mean / config.max_background_shape)
    shape = float(
        np.clip(shape, config.min_background_shape, config.max_background_shape)
    )
    rate = float(
        np.clip(
            shape / max(mean, config.min_rate),
            config.min_rate,
            1.0 / config.min_rate,
        )
    )
    return shape, rate, 0.0


def _fit_background_parameters(
    shells: np.ndarray,
    responsibilities: np.ndarray,
    *,
    shapes: np.ndarray,
    log_ranks: np.ndarray,
    initial: tuple[float, float, float],
    config: GammaMDLConfig,
) -> tuple[float, float, float]:
    """Weighted M-step for the continuous-rate background distribution."""

    weights = np.asarray(responsibilities, dtype=float)
    if float(np.sum(weights)) <= np.finfo(float).eps:
        return initial
    total_shape = float(np.sum(shapes))
    rank_moment = float(np.sum(shapes * log_ranks))
    min_beta = config.min_background_shape / config.max_rate
    max_beta = config.max_background_shape / config.min_rate

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        shape = float(np.exp(theta[0]))
        rate = float(np.exp(theta[1]))
        slope = float(theta[2])
        rank_scale = np.exp(np.clip(slope * log_ranks, -50.0, 50.0))
        exposure = shells @ rank_scale
        exposure_gradient = shells @ (rank_scale * log_ranks)
        normalizer = np.maximum(rate + exposure, np.finfo(float).tiny)
        log_density = (
            slope * rank_moment
            + shape * math.log(rate)
            - float(gammaln(shape))
            + float(gammaln(shape + total_shape))
            - (shape + total_shape) * np.log(normalizer)
        )
        derivative_shape = (
            math.log(rate)
            - float(digamma(shape))
            + float(digamma(shape + total_shape))
            - np.log(normalizer)
        )
        derivative_rate = shape / rate - (shape + total_shape) / normalizer
        derivative_slope = (
            rank_moment
            - (shape + total_shape) * exposure_gradient / normalizer
        )
        value = -float(np.sum(weights * log_density))
        gradient = -np.array(
            [
                np.sum(weights * derivative_shape) * shape,
                np.sum(weights * derivative_rate) * rate,
                np.sum(weights * derivative_slope),
            ],
            dtype=float,
        )
        return value, gradient

    initial_vector = np.array(
        [math.log(initial[0]), math.log(initial[1]), initial[2]], dtype=float
    )
    result = minimize(
        objective,
        initial_vector,
        method="L-BFGS-B",
        jac=True,
        bounds=[
            (
                math.log(config.min_background_shape),
                math.log(config.max_background_shape),
            ),
            (math.log(min_beta), math.log(max_beta)),
            (-config.max_background_abs_slope, config.max_background_abs_slope),
        ],
        options={"maxiter": config.background_m_step_max_iter, "ftol": 1e-10},
    )
    vector = result.x if np.all(np.isfinite(result.x)) else initial_vector
    return float(np.exp(vector[0])), float(np.exp(vector[1])), float(vector[2])


def _lognormal_rate_background_log_density(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    log_ranks: np.ndarray,
    location: float,
    scale: float,
    slope: float,
    data_term: np.ndarray,
    quadrature_order: int,
) -> np.ndarray:
    """Integrate the conditional Gamma shells over a lognormal local rate."""

    density, _ = _lognormal_rate_background_density_and_score(
        shells,
        shapes=shapes,
        log_ranks=log_ranks,
        location=location,
        scale=scale,
        slope=slope,
        data_term=data_term,
        quadrature_order=quadrature_order,
    )
    return density


def _lognormal_rate_background_density_and_score(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    log_ranks: np.ndarray,
    location: float,
    scale: float,
    slope: float,
    data_term: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return marginal density and derivatives for lognormal parameters."""

    nodes, weights = _gauss_hermite_rule(quadrature_order)
    standard_normal = np.sqrt(2.0) * nodes
    log_weights = np.log(weights) - 0.5 * math.log(math.pi)
    log_lambda = location + scale * standard_normal
    local_rate = np.exp(np.clip(log_lambda, -700.0, 700.0))
    total_shape = float(np.sum(shapes))
    rank_scale = np.exp(np.clip(slope * log_ranks, -50.0, 50.0))
    exposure = shells @ rank_scale
    log_integrand = (
        log_weights[None, :]
        + total_shape * log_lambda[None, :]
        - exposure[:, None] * local_rate[None, :]
    )
    integrated = logsumexp(log_integrand, axis=1)
    posterior = np.exp(log_integrand - integrated[:, None])
    base_score = total_shape - exposure[:, None] * local_rate[None, :]
    expected_rate = posterior @ local_rate
    score = np.column_stack(
        [
            np.sum(posterior * base_score, axis=1),
            np.sum(
                posterior * base_score * standard_normal[None, :], axis=1
            ),
            float(np.sum(shapes * log_ranks))
            - (shells @ (rank_scale * log_ranks)) * expected_rate,
        ]
    )
    density = (
        data_term
        + slope * float(np.sum(shapes * log_ranks))
        + integrated
    )
    return density, score


@lru_cache(maxsize=None)
def _gauss_hermite_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.hermite.hermgauss(order)


def _inverse_gamma_rate_background_log_density(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    log_ranks: np.ndarray,
    shape: float,
    scale: float,
    slope: float,
    data_term: np.ndarray,
) -> np.ndarray:
    """Integrate the conditional Gamma shells over an inverse-Gamma rate."""

    total_shape = float(np.sum(shapes))
    rank_scale = np.exp(np.clip(slope * log_ranks, -50.0, 50.0))
    exposure = np.maximum(shells @ rank_scale, np.finfo(float).tiny)
    order = abs(total_shape - shape)
    argument = 2.0 * np.sqrt(scale * exposure)
    log_bessel = np.log(
        np.maximum(kve(order, argument), np.finfo(float).tiny)
    ) - argument
    integrated = (
        shape * math.log(scale)
        - float(gammaln(shape))
        + math.log(2.0)
        + 0.5
        * (total_shape - shape)
        * (math.log(scale) - np.log(exposure))
        + log_bessel
    )
    return (
        data_term
        + slope * float(np.sum(shapes * log_ranks))
        + integrated
    )


def _initial_background_model_parameters(
    shells: np.ndarray,
    responsibilities: np.ndarray,
    shapes: np.ndarray,
    config: GammaMDLConfig,
) -> tuple[float, float, float]:
    family = config.background_distribution
    if family == "gamma_rate_mixture":
        return _initial_background_parameters(
            shells, responsibilities, shapes, config
        )
    total_shape = float(np.sum(shapes))
    local_rate = total_shape / np.maximum(
        np.sum(shells, axis=1), np.finfo(float).tiny
    )
    weights = np.asarray(responsibilities, dtype=float)
    effective = max(float(np.sum(weights)), np.finfo(float).tiny)
    if family == "lognormal_rate_mixture":
        log_rate = np.log(np.maximum(local_rate, config.min_rate))
        location = float(np.sum(weights * log_rate) / effective)
        variance = float(
            np.sum(weights * (log_rate - location) ** 2) / effective
        )
        scale = float(
            np.clip(
                math.sqrt(max(variance, config.min_background_scale**2)),
                config.min_background_scale,
                config.max_background_scale,
            )
        )
        return location, scale, 0.0
    if family == "inverse_gamma_rate_mixture":
        mean = float(np.sum(weights * local_rate) / effective)
        variance = float(np.sum(weights * (local_rate - mean) ** 2) / effective)
        shape = 2.0 + mean * mean / max(
            variance, mean * mean / config.max_background_shape
        )
        shape = float(
            np.clip(shape, config.min_background_shape, config.max_background_shape)
        )
        scale = float(
            np.clip(
                mean * max(shape - 1.0, config.min_background_shape),
                config.min_rate,
                1.0 / config.min_rate,
            )
        )
        return shape, scale, 0.0
    raise ValueError("explicit background parameters require a continuous-rate family")


def _background_model_log_density(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    log_ranks: np.ndarray,
    parameters: tuple[float, float, float],
    data_term: np.ndarray,
    config: GammaMDLConfig,
) -> np.ndarray:
    first, second, slope = parameters
    if config.background_distribution == "gamma_rate_mixture":
        return _background_log_density(
            shells,
            shapes=shapes,
            log_ranks=log_ranks,
            shape=first,
            rate=second,
            slope=slope,
            data_term=data_term,
        )
    if config.background_distribution == "lognormal_rate_mixture":
        return _lognormal_rate_background_log_density(
            shells,
            shapes=shapes,
            log_ranks=log_ranks,
            location=first,
            scale=second,
            slope=slope,
            data_term=data_term,
            quadrature_order=config.background_quadrature_order,
        )
    if config.background_distribution == "inverse_gamma_rate_mixture":
        return _inverse_gamma_rate_background_log_density(
            shells,
            shapes=shapes,
            log_ranks=log_ranks,
            shape=first,
            scale=second,
            slope=slope,
            data_term=data_term,
        )
    raise ValueError("background log density requires a continuous-rate family")


def _fit_background_model_parameters(
    shells: np.ndarray,
    responsibilities: np.ndarray,
    *,
    shapes: np.ndarray,
    log_ranks: np.ndarray,
    initial: tuple[float, float, float],
    data_term: np.ndarray,
    config: GammaMDLConfig,
) -> tuple[float, float, float]:
    if config.background_distribution == "gamma_rate_mixture":
        return _fit_background_parameters(
            shells,
            responsibilities,
            shapes=shapes,
            log_ranks=log_ranks,
            initial=initial,
            config=config,
        )
    weights = np.asarray(responsibilities, dtype=float)
    if float(np.sum(weights)) <= np.finfo(float).eps:
        return initial
    if config.background_distribution == "lognormal_rate_mixture":
        initial_vector = np.array(
            [initial[0], math.log(initial[1]), initial[2]], dtype=float
        )
        bounds = [
            (math.log(config.min_rate), math.log(config.max_rate)),
            (
                math.log(config.min_background_scale),
                math.log(config.max_background_scale),
            ),
            (-config.max_background_abs_slope, config.max_background_abs_slope),
        ]

        def unpack(theta: np.ndarray) -> tuple[float, float, float]:
            return float(theta[0]), float(np.exp(theta[1])), float(theta[2])

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            location, scale, slope = unpack(theta)
            density, score = _lognormal_rate_background_density_and_score(
                shells,
                shapes=shapes,
                log_ranks=log_ranks,
                location=location,
                scale=scale,
                slope=slope,
                data_term=data_term,
                quadrature_order=config.background_quadrature_order,
            )
            if not np.all(np.isfinite(density)) or not np.all(np.isfinite(score)):
                return 1e100, np.zeros(3, dtype=float)
            gradient = -np.array(
                [
                    np.sum(weights * score[:, 0]),
                    np.sum(weights * score[:, 1]) * scale,
                    np.sum(weights * score[:, 2]),
                ],
                dtype=float,
            )
            return -float(np.sum(weights * density)), gradient

        use_jacobian = True

    elif config.background_distribution == "inverse_gamma_rate_mixture":
        initial_vector = np.array(
            [math.log(initial[0]), math.log(initial[1]), initial[2]], dtype=float
        )
        bounds = [
            (
                math.log(max(config.min_background_shape, 1.0 + 1e-3)),
                math.log(config.max_background_shape),
            ),
            (math.log(config.min_rate), math.log(1.0 / config.min_rate)),
            (-config.max_background_abs_slope, config.max_background_abs_slope),
        ]

        def unpack(theta: np.ndarray) -> tuple[float, float, float]:
            return float(np.exp(theta[0])), float(np.exp(theta[1])), float(theta[2])

        def objective(theta: np.ndarray) -> float:
            parameters = unpack(theta)
            density = _background_model_log_density(
                shells,
                shapes=shapes,
                log_ranks=log_ranks,
                parameters=parameters,
                data_term=data_term,
                config=config,
            )
            if not np.all(np.isfinite(density)):
                return 1e100
            return -float(np.sum(weights * density))

        use_jacobian = False

    else:
        raise ValueError("background M-step requires a continuous-rate family")

    result = minimize(
        objective,
        initial_vector,
        method="L-BFGS-B",
        jac=use_jacobian,
        bounds=bounds,
        options={"maxiter": config.background_m_step_max_iter, "ftol": 1e-9},
    )
    initial_result = objective(initial_vector)
    initial_value = (
        float(initial_result[0]) if isinstance(initial_result, tuple) else initial_result
    )
    candidate_result = objective(result.x) if np.all(np.isfinite(result.x)) else math.inf
    candidate_value = (
        float(candidate_result[0])
        if isinstance(candidate_result, tuple)
        else candidate_result
    )
    vector = result.x if candidate_value <= initial_value else initial_vector
    return unpack(vector)


def _background_parameter_diagnostics(
    parameters: tuple[float, float, float],
    config: GammaMDLConfig,
) -> dict[str, object]:
    first, second, slope = parameters
    family = config.background_distribution
    at_bounds: list[str] = []
    if np.isclose(
        abs(slope), config.max_background_abs_slope, rtol=1e-5, atol=1e-8
    ):
        at_bounds.append("slope")
    if family == "gamma_rate_mixture":
        mean = first / second
        names = {"shape": first, "rate": second, "slope": slope}
        if np.isclose(first, config.min_background_shape, rtol=1e-5):
            at_bounds.append("shape_min")
        if np.isclose(first, config.max_background_shape, rtol=1e-5):
            at_bounds.append("shape_max")
    elif family == "lognormal_rate_mixture":
        mean = math.exp(min(first + 0.5 * second * second, 700.0))
        names = {"log_location": first, "log_scale": second, "slope": slope}
        if np.isclose(second, config.min_background_scale, rtol=1e-5):
            at_bounds.append("log_scale_min")
        if np.isclose(second, config.max_background_scale, rtol=1e-5):
            at_bounds.append("log_scale_max")
    elif family == "inverse_gamma_rate_mixture":
        mean = second / (first - 1.0) if first > 1.0 else math.inf
        names = {"shape": first, "scale": second, "slope": slope}
        if np.isclose(
            first,
            max(config.min_background_shape, 1.0 + 1e-3),
            rtol=1e-5,
        ):
            at_bounds.append("shape_min")
        if np.isclose(first, config.max_background_shape, rtol=1e-5):
            at_bounds.append("shape_max")
    else:
        raise ValueError("unknown continuous-rate background family")
    diagnostics: dict[str, object] = {
        "mdl_gamma_background_parameters": names,
        "mdl_gamma_background_parameters_at_bounds": at_bounds,
        "mdl_gamma_background_rate_mean": mean,
    }
    diagnostics.update(
        {f"mdl_gamma_background_{name}": value for name, value in names.items()}
    )
    return diagnostics


def _cold_semantic_responsibilities(
    shells: np.ndarray,
    components: int,
    restart: int,
    n_init: int,
    random_state: int,
) -> np.ndarray:
    """Initialize signal strata while reserving a broad sparse background."""

    n = shells.shape[0]
    if components == 1:
        return np.ones((n, 1), dtype=float)
    signal_components = components - 1
    density_score = np.log(
        np.maximum(np.sum(shells, axis=1), np.finfo(float).tiny)
    )
    background_fraction = (restart + 1.0) / (n_init + 1.0)
    background_size = int(np.clip(round(background_fraction * n), 1, n - 1))
    density_order = np.argsort(density_score, kind="stable")
    background_rows = density_order[-background_size:]
    signal_rows = density_order[:-background_size]
    if restart == 0 or signal_components == 1:
        signal_score = density_score[signal_rows]
    else:
        features = np.log(np.maximum(shells, np.finfo(float).tiny))
        features = (features - np.mean(features, axis=0)) / np.maximum(
            np.std(features, axis=0), 1e-8
        )
        direction = np.random.default_rng(random_state + restart).normal(
            size=features.shape[1]
        )
        signal_score = features[signal_rows] @ direction
    signal_order = signal_rows[np.argsort(signal_score, kind="stable")]
    labels = np.full(n, signal_components, dtype=np.int32)
    labels[signal_order] = np.minimum(
        signal_components - 1,
        np.floor(
            np.arange(signal_order.size) * signal_components / signal_order.size
        ).astype(np.int32),
    )
    labels[background_rows] = signal_components
    return np.eye(components, dtype=float)[labels]


def _split_semantic_responsibilities(
    responsibilities: np.ndarray,
    shells: np.ndarray,
) -> np.ndarray:
    """Add one signal state without splitting the semantic background label."""

    current = np.asarray(responsibilities, dtype=float)
    score = np.log(np.maximum(np.sum(shells, axis=1), np.finfo(float).tiny))
    effective = np.sum(current, axis=0)
    means = np.sum(current * score[:, None], axis=0) / np.maximum(
        effective, np.finfo(float).tiny
    )
    variance = np.sum(
        current * (score[:, None] - means[None, :]) ** 2, axis=0
    ) / np.maximum(effective, np.finfo(float).tiny)
    component = int(np.argmax(effective * variance))
    members = current[:, component]
    order = np.argsort(score, kind="stable")
    split_at = int(
        np.searchsorted(np.cumsum(members[order]), 0.5 * effective[component])
    )
    threshold = score[order[min(split_at, order.size - 1)]]
    dense = members * (score <= threshold)
    sparse = members - dense
    if min(float(np.sum(dense)), float(np.sum(sparse))) <= np.finfo(float).eps:
        dense = members * (np.arange(score.size) % 2 == 0)
        sparse = members - dense
    background = current[:, -1]
    signals = current[:, :-1].copy()
    if component == current.shape[1] - 1:
        return np.column_stack([signals, dense, sparse])
    signals[:, component] = dense
    return np.column_stack([signals, sparse, background])


def _fit_heterogeneous_background_once(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    ranks: tuple[int, ...],
    components: int,
    config: GammaMDLConfig,
    initial_responsibilities: np.ndarray,
    workspace: object,
) -> _HeterogeneousBackgroundGammaFit:
    responsibilities = np.asarray(initial_responsibilities, dtype=float).copy()
    if responsibilities.shape != (shells.shape[0], components):
        raise ValueError("background warm start has the wrong shape")
    responsibilities /= np.maximum(
        np.sum(responsibilities, axis=1, keepdims=True), np.finfo(float).tiny
    )
    signal_components = components - 1
    log_ranks = np.log(np.asarray(ranks, dtype=float))
    background_parameters = _initial_background_model_parameters(
        shells, responsibilities[:, -1], shapes, config
    )
    previous = -math.inf
    converged = False
    log_likelihood = -math.inf
    mu = np.empty(0, dtype=float)
    slopes = np.empty(0, dtype=float)
    rates = np.empty((0, len(ranks)), dtype=float)
    cached = workspace
    for iteration in range(config.max_iter):
        effective = np.maximum(
            np.sum(responsibilities, axis=0), np.finfo(float).tiny
        )
        weights = effective / shells.shape[0]
        if signal_components:
            weighted_shells = responsibilities[:, :signal_components].T @ shells
            mu, slopes, rates = _fit_rate_profiles(
                weighted_shells,
                effective[:signal_components],
                shapes=shapes,
                log_ranks=log_ranks,
                config=PredictiveMultiscaleConfig(
                    ranks=ranks,
                    min_rate=config.min_rate,
                    max_rate=config.max_rate,
                ),
            )
        background_parameters = _fit_background_model_parameters(
            shells,
            responsibilities[:, -1],
            shapes=shapes,
            log_ranks=log_ranks,
            initial=background_parameters,
            data_term=cached.data_term,
            config=config,
        )
        log_probability = np.empty((shells.shape[0], components), dtype=float)
        if signal_components:
            component_term = (
                np.log(np.maximum(weights[:signal_components], np.finfo(float).tiny))
                + np.sum(shapes[None, :] * np.log(rates), axis=1)
            )
            log_probability[:, :signal_components] = (
                cached.data_term[:, None]
                + component_term[None, :]
                - shells @ rates.T
            )
        log_probability[:, -1] = np.log(
            max(float(weights[-1]), np.finfo(float).tiny)
        ) + _background_model_log_density(
            shells,
            shapes=shapes,
            log_ranks=log_ranks,
            parameters=background_parameters,
            data_term=cached.data_term,
            config=config,
        )
        log_norm = logsumexp(log_probability, axis=1)
        responsibilities = np.exp(log_probability - log_norm[:, None])
        log_likelihood = float(np.sum(log_norm))
        if np.isfinite(previous) and abs(log_likelihood - previous) <= config.tolerance * (
            1.0 + abs(previous)
        ):
            converged = True
            break
        previous = log_likelihood

    if signal_components:
        order = np.argsort(rates[:, 0], kind="stable")[::-1]
        weights = np.concatenate([weights[order], weights[-1:]])
        mu = mu[order]
        slopes = slopes[order]
        rates = rates[order]
        responsibilities = np.column_stack(
            [responsibilities[:, order], responsibilities[:, -1]]
        )
    return _HeterogeneousBackgroundGammaFit(
        log_likelihood=log_likelihood,
        weights=weights,
        mu=mu,
        slopes=slopes,
        rates=rates,
        background_parameters=background_parameters,
        responsibilities=responsibilities,
        iterations=iteration + 1,
        converged=converged,
    )


def _fit_heterogeneous_background_gamma(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    ranks: tuple[int, ...],
    components: int,
    config: GammaMDLConfig,
    previous: _HeterogeneousBackgroundGammaFit | None,
    workspace: object,
) -> _HeterogeneousBackgroundGammaFit:
    starts: list[np.ndarray] = []
    if (
        config.split_warm_start
        and previous is not None
        and previous.responsibilities.shape[1] + 1 == components
    ):
        starts.append(
            _split_semantic_responsibilities(previous.responsibilities, shells)
        )
    cold_restart = 0
    while len(starts) < config.n_init:
        starts.append(
            _cold_semantic_responsibilities(
                shells,
                components,
                cold_restart,
                config.n_init,
                config.random_state,
            )
        )
        cold_restart += 1
    fits = [
        _fit_heterogeneous_background_once(
            shells,
            shapes=shapes,
            ranks=ranks,
            components=components,
            config=config,
            initial_responsibilities=initial,
            workspace=workspace,
        )
        for initial in starts
    ]
    return max(fits, key=lambda fit: fit.log_likelihood)


def _search_needs_expansion(
    selected_components: int,
    evaluated_max: int,
    hard_max: int,
    boundary_margin: int,
) -> bool:
    return (
        evaluated_max < hard_max
        and selected_components >= evaluated_max - boundary_margin
    )


def _fit_gamma_candidate(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    components: int,
    config: GammaMDLConfig,
    numerical: PredictiveMultiscaleConfig,
    workspace: object,
    previous: object | None,
) -> object:
    """Fit one Gamma basis size with both split-warm and independent starts."""

    starts = config.n_init
    fits = []
    if previous is not None and previous.responsibilities.shape[1] + 1 == components:
        initial = _split_responsibilities(
            previous.responsibilities, shells, components
        )
        fits.append(
            _fit_once(
                shells,
                shapes=shapes,
                ranks=config.ranks,
                components=components,
                restart=0,
                config=numerical,
                initial_responsibilities=initial,
                workspace=workspace,
            )
        )
        starts -= 1
    if starts:
        fits.append(
            _fit_constrained_gamma(
                shells,
                shapes=shapes,
                ranks=config.ranks,
                components=components,
                n_init=starts,
                config=numerical,
                workspace=workspace,
            )
        )
    return max(fits, key=lambda fit: fit.log_likelihood)


def _background_scan_supported_components(
    graph: KNNGraph,
    responsibilities: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, list[float], list[int]]:
    """Promote Gamma bases only when they beat the coded background scan."""

    components = responsibilities.shape[1]
    assignment = np.argmax(responsibilities, axis=1)
    core_distance = np.asarray(graph.distances[:, rank - 1], dtype=np.float64)
    gains = [0.0] * components
    sizes = [0] * components
    supported: list[int] = []
    # Retain the sparsest density basis as the explicit background anchor. Any
    # finite clusters inside it are still recoverable by the downstream
    # background scan.
    for component in range(max(components - 1, 0)):
        mask = assignment == component
        if int(np.sum(mask)) < 2 * rank:
            continue
        selected, _, profile = _adaptive_background_core(
            graph,
            mask,
            core_distance,
            rank=rank,
        )
        gain = float(profile["total_gain"])
        gains[component] = gain
        sizes[component] = int(np.sum(selected))
        if gain > 0.0 and np.any(selected):
            supported.append(component)
    return np.asarray(supported, dtype=np.int32), gains, sizes


def _rate_changepoint_supported_components(
    rates: np.ndarray,
    weights: np.ndarray,
    *,
    n_samples: int,
    total_shell_shape: float,
) -> tuple[np.ndarray, list[float], list[int]]:
    """Find a dense-prefix/background change point with an MDL evidence test.

    Adjacent first-shell log-rate gaps are standardized by the approximate
    Fisher variance of their fitted Gamma rates.  A split must repay both the
    extra binary boundary and the search over all adjacent boundaries.  Unlike
    the former largest-gap rule, a homogeneous continuum may therefore retain
    the all-background solution.
    """

    components = int(weights.size)
    gains = [0.0] * components
    sizes = [0] * components
    if components <= 1:
        return np.empty(0, dtype=np.int32), gains, sizes
    effective = np.maximum(n_samples * np.asarray(weights, dtype=float), 1.0)
    log_rate = np.log(np.maximum(rates[:, 0], np.finfo(float).tiny))
    gaps = log_rate[:-1] - log_rate[1:]
    variance = (
        1.0 / (total_shell_shape * effective[:-1])
        + 1.0 / (total_shell_shape * effective[1:])
    )
    evidence = (
        0.5 * gaps * gaps / np.maximum(variance, np.finfo(float).tiny)
        - 0.5 * math.log(n_samples)
        - math.log(components - 1)
    )
    boundary = int(np.argmax(evidence))
    for index, value in enumerate(evidence):
        gains[index] = float(value)
    if evidence[boundary] <= 0.0:
        return np.empty(0, dtype=np.int32), gains, sizes
    cut = boundary + 1
    sizes[boundary] = int(round(float(np.sum(effective[:cut]))))
    return np.arange(cut, dtype=np.int32), gains, sizes


def _validated_rate_prefix_components(
    graph: KNNGraph,
    rates: np.ndarray,
    weights: np.ndarray,
    responsibilities: np.ndarray,
    *,
    rank: int,
    total_shell_shape: float,
) -> tuple[np.ndarray, list[float], list[int]]:
    """Require a density-rate prefix to beat the coded background process."""

    proposed, rate_gains, sizes = _rate_changepoint_supported_components(
        rates,
        weights,
        n_samples=graph.n_samples,
        total_shell_shape=total_shell_shape,
    )
    if proposed.size == 0:
        return proposed, rate_gains, sizes
    assignment = np.argmax(responsibilities, axis=1)
    mask = np.isin(assignment, proposed)
    core_distance = np.asarray(graph.distances[:, rank - 1], dtype=np.float64)
    from .strict_core import _rank_window_component_sizes

    rows = np.flatnonzero(mask).astype(np.int64)
    widths: list[int] = []
    width = rank
    while 2 * width <= rows.size:
        widths.append(width)
        width *= 2
    validation_gain = -math.inf
    selected_size = 0
    if widths:
        order = rows[np.argsort(core_distance[rows], kind="stable")]
        for candidate_width in widths:
            component_sizes = _rank_window_component_sizes(
                np.asarray(graph.indices, dtype=np.int32),
                np.asarray(order, dtype=np.int32),
                int(candidate_width),
                int(rank),
            )
            if component_sizes.size < 2:
                continue
            observed = int(component_sizes[0])
            pvalue = (
                1.0 + float(np.sum(component_sizes[1:] >= observed))
            ) / component_sizes.size
            gain = -math.log(pvalue) - math.log(len(widths)) - math.log(2.0)
            if gain > validation_gain:
                validation_gain = gain
                selected_size = observed
    boundary = int(proposed[-1])
    rate_gains[boundary] += validation_gain
    sizes[boundary] = selected_size
    if validation_gain <= 0.0 or selected_size == 0:
        return np.empty(0, dtype=np.int32), rate_gains, sizes
    return proposed, rate_gains, sizes


def _topology_supported_gamma_components(
    graph: KNNGraph,
    responsibilities: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, list[float], list[int]]:
    """Assign semantic signal roles using graph evidence disjoint from extraction.

    The first neighbour window constructs StrictCore later, the second scores
    its extraction thresholds, and this third window selects whether a Gamma
    basis component deserves a signal-stratum role.  The sparsest basis remains
    part of the explicit background so the all-background solution is always
    available.
    """

    components = responsibilities.shape[1]
    assignment = np.argmax(responsibilities, axis=1)
    core_distance = np.asarray(graph.distances[:, rank - 1], dtype=np.float64)
    gains = [0.0] * components
    threshold_sizes = [0] * components
    supported: list[int] = []
    tiny = np.finfo(np.float64).eps
    for component in range(max(components - 1, 0)):
        rows = np.flatnonzero(assignment == component).astype(np.int64)
        if rows.size < 2:
            continue
        probability = np.clip(responsibilities[:, component], tiny, 1.0 - tiny)
        unary_gain = np.log(probability) - np.log1p(-probability)
        order = np.argsort(core_distance[rows], kind="stable")
        event_steps, event_left, event_right = _stratum_events(
            graph,
            rows,
            order,
            rank,
            edge_offset=2 * rank,
        )
        best_size, best_gain = _best_threshold_event_sweep(
            unary_gain[rows[order]],
            event_steps,
            event_left,
            event_right,
            graph.n_samples,
            rank,
        )
        gains[component] = float(best_gain)
        threshold_sizes[component] = int(best_size)
        if best_size > 0 and best_gain > 0.0:
            supported.append(component)
    return np.asarray(supported, dtype=np.int32), gains, threshold_sizes


def _semantic_responsibilities(
    responsibilities: np.ndarray,
    supported: np.ndarray,
) -> np.ndarray:
    components = responsibilities.shape[1]
    supported_set = set(map(int, supported))
    background = [index for index in range(components) if index not in supported_set]
    columns = [responsibilities[:, int(index)] for index in supported]
    columns.append(np.sum(responsibilities[:, background], axis=1))
    return np.column_stack(columns)


def _estimate_gamma_basis_stratification(
    graph: KNNGraph,
    *,
    signatures: np.ndarray,
    shells: np.ndarray,
    shapes: np.ndarray,
    signature_seconds: float,
    config: GammaMDLConfig,
) -> StratificationResult:
    """Select Gamma density bases and infer signal roles separately."""

    workspace = _gamma_workspace(shells, shapes)
    numerical = PredictiveMultiscaleConfig(
        ranks=config.ranks,
        min_components=config.min_components,
        max_components=config.hard_max_components,
        final_n_init=config.n_init,
        max_iter=config.max_iter,
        tolerance=config.tolerance,
        min_rate=config.min_rate,
        max_rate=config.max_rate,
        random_state=config.random_state,
    )
    fit_started = perf_counter()
    fits = []
    component_grid: list[int] = []
    scores: list[float] = []
    log_likelihoods: list[float] = []
    entropies: list[float] = []
    iterations: list[int] = []
    converged: list[bool] = []
    supported_by_candidate: list[np.ndarray] = []
    role_gains_by_candidate: list[list[float]] = []
    role_sizes_by_candidate: list[list[int]] = []
    search_caps: list[int] = []
    cap = config.max_components
    next_components = config.min_components
    previous = None
    while True:
        for components in range(next_components, cap + 1):
            fit = _fit_gamma_candidate(
                shells,
                shapes=shapes,
                components=components,
                config=config,
                numerical=numerical,
                workspace=workspace,
                previous=previous,
            )
            if config.component_roles == "validated_rate_prefix":
                supported, role_gains, role_sizes = (
                    _validated_rate_prefix_components(
                        graph,
                        fit.rates,
                        fit.weights,
                        fit.responsibilities,
                        rank=config.role_rank,
                        total_shell_shape=float(np.sum(shapes)),
                    )
                )
            elif config.component_roles == "background_scan_mdl":
                supported, role_gains, role_sizes = (
                    _background_scan_supported_components(
                        graph, fit.responsibilities, rank=config.role_rank
                    )
                )
            elif config.component_roles == "rate_changepoint_mdl":
                supported, role_gains, role_sizes = (
                    _rate_changepoint_supported_components(
                        fit.rates,
                        fit.weights,
                        n_samples=graph.n_samples,
                        total_shell_shape=float(np.sum(shapes)),
                    )
                )
            elif config.component_roles == "topology_mdl":
                supported, role_gains, role_sizes = (
                    _topology_supported_gamma_components(
                        graph, fit.responsibilities, rank=config.role_rank
                    )
                )
            elif components == 1:
                supported = np.empty(0, dtype=np.int32)
                role_gains = [0.0]
                role_sizes = [0]
            else:
                density_rates = fit.rates[:, 0]
                gaps = (
                    np.log(np.maximum(density_rates[:-1], config.min_rate))
                    - np.log(np.maximum(density_rates[1:], config.min_rate))
                )
                cut = int(np.argmax(gaps) + 1)
                supported = np.arange(cut, dtype=np.int32)
                role_gains = [0.0] * components
                role_sizes = [0] * components
            semantic = (
                _semantic_responsibilities(fit.responsibilities, supported)
                if config.entropy_scope == "semantic"
                else fit.responsibilities
            )
            parameter_count = (components - 1) + 2 * components
            entropy = -float(
                np.sum(
                    semantic
                    * np.log(np.maximum(semantic, np.finfo(float).tiny))
                )
            )
            score = _information_score(
                fit.log_likelihood,
                semantic,
                parameter_count,
                config.criterion,
            )
            fits.append(fit)
            component_grid.append(components)
            scores.append(float(score))
            log_likelihoods.append(float(fit.log_likelihood))
            entropies.append(entropy)
            iterations.append(int(fit.iterations))
            converged.append(bool(fit.converged))
            supported_by_candidate.append(supported)
            role_gains_by_candidate.append(role_gains)
            role_sizes_by_candidate.append(role_sizes)
            previous = fit
        search_caps.append(cap)
        selected_index = int(np.argmin(scores))
        selected_components = component_grid[selected_index]
        if not config.adaptive_components or not _search_needs_expansion(
            selected_components,
            cap,
            config.hard_max_components,
            config.boundary_margin,
        ):
            break
        next_components = cap + 1
        cap = min(config.hard_max_components, cap + config.expansion_step)

    selected = fits[selected_index]
    supported_basis = supported_by_candidate[selected_index]
    supported_set = set(map(int, supported_basis))
    background_basis = np.asarray(
        [
            component
            for component in range(selected_components)
            if component not in supported_set
        ],
        dtype=np.int32,
    )
    signal_components = int(supported_basis.size)
    background_group = signal_components
    assignment = np.argmax(selected.responsibilities, axis=1)
    basis_to_group = np.full(selected_components, background_group, dtype=np.int32)
    basis_to_group[supported_basis] = np.arange(signal_components, dtype=np.int32)
    groups = basis_to_group[assignment]
    background_probability = np.sum(
        selected.responsibilities[:, background_basis], axis=1
    )
    signal_rates = selected.rates[supported_basis, 0]
    log_rate_gaps = (
        np.log(np.maximum(signal_rates[:-1], config.min_rate))
        - np.log(np.maximum(signal_rates[1:], config.min_rate))
        if signal_components > 1
        else np.empty(0)
    )
    selected_parameter_count = (selected_components - 1) + 2 * selected_components
    bic = -2.0 * selected.log_likelihood + selected_parameter_count * math.log(
        graph.n_samples
    )
    fit_seconds = perf_counter() - fit_started
    hit_hard_max = bool(
        selected_components >= config.hard_max_components - config.boundary_margin
    )
    return StratificationResult(
        signatures=signatures,
        groups=groups,
        eps_by_group=np.zeros(signal_components + 1, dtype=np.float32),
        point_eps=np.zeros(graph.n_samples, dtype=np.float32),
        supported_groups=np.arange(signal_components, dtype=np.int32),
        selected_components=signal_components + 1,
        bic=float(bic),
        fit_indices=np.arange(graph.n_samples, dtype=np.int64),
        timings={
            "density_signatures": signature_seconds,
            "full_data_gamma_mdl": fit_seconds,
        },
        gap_ratios=np.exp(log_rate_gaps).astype(np.float32),
        selection_method=(
            f"adaptive_full_data_{config.entropy_scope}_{config.criterion}_constrained_"
            "multiscale_gamma"
        ),
        diagnostics={
            "mdl_gamma_ranks": list(map(int, config.ranks)),
            "mdl_gamma_criterion": config.criterion,
            "mdl_gamma_entropy_scope": config.entropy_scope,
            "mdl_gamma_candidate_components": component_grid,
            "mdl_gamma_candidate_scores": scores,
            "mdl_gamma_candidate_log_likelihoods": log_likelihoods,
            "mdl_gamma_candidate_entropies": entropies,
            "mdl_gamma_candidate_iterations": iterations,
            "mdl_gamma_candidate_converged": converged,
            "mdl_gamma_candidate_signal_components": [
                values.astype(int).tolist() for values in supported_by_candidate
            ],
            "mdl_gamma_candidate_role_gains_nats": role_gains_by_candidate,
            "mdl_gamma_candidate_role_threshold_sizes": role_sizes_by_candidate,
            "mdl_gamma_selected_components": selected_components,
            "mdl_gamma_signal_components": signal_components,
            "mdl_gamma_signal_basis_components": supported_basis.astype(int).tolist(),
            "mdl_gamma_background_components": int(background_basis.size),
            "mdl_gamma_background_basis_components": background_basis.astype(int).tolist(),
            "mdl_gamma_grouping": config.component_roles,
            "mdl_gamma_background_group": background_group,
            "mdl_gamma_assignment": "maximum_posterior_basis_then_semantic_role",
            "mdl_gamma_background_boundary": config.component_roles,
            "mdl_gamma_background_distribution": "gamma_components",
            "mdl_gamma_role_evidence_neighbour_ranks": [
                int(2 * config.role_rank + 1),
                int(3 * config.role_rank),
            ]
            if config.component_roles == "topology_mdl"
            else [],
            "mdl_gamma_fit_samples": int(graph.n_samples),
            "mdl_gamma_weights": selected.weights.tolist(),
            "mdl_gamma_mu": selected.mu.tolist(),
            "mdl_gamma_slopes": selected.slopes.tolist(),
            "mdl_gamma_rates": selected.rates.tolist(),
            "mdl_gamma_log_rate_gaps": log_rate_gaps.tolist(),
            "mdl_gamma_background_fraction": float(
                np.mean(groups == background_group)
            ),
            "mdl_gamma_mean_background_probability": float(
                np.mean(background_probability)
            ),
            "mdl_gamma_mean_posterior_entropy": float(
                entropies[selected_index] / graph.n_samples
            ),
            "mdl_gamma_adaptive_search": bool(config.adaptive_components),
            "mdl_gamma_split_warm_start": bool(config.split_warm_start),
            "mdl_gamma_search_caps": search_caps,
            "mdl_gamma_search_initial_max": int(config.max_components),
            "mdl_gamma_search_hard_max": int(config.hard_max_components),
            "mdl_gamma_search_hit_hard_max": hit_hard_max,
        },
        mode="semantic_information_criterion_gamma_basis",
        background_probability=background_probability.astype(np.float32),
    )


def _estimate_heterogeneous_background_stratification(
    graph: KNNGraph,
    *,
    signatures: np.ndarray,
    shells: np.ndarray,
    shapes: np.ndarray,
    signature_seconds: float,
    config: GammaMDLConfig,
) -> StratificationResult:
    workspace = _gamma_workspace(shells, shapes)
    fit_started = perf_counter()
    fits: list[_HeterogeneousBackgroundGammaFit] = []
    component_grid: list[int] = []
    scores: list[float] = []
    log_likelihoods: list[float] = []
    entropies: list[float] = []
    iterations: list[int] = []
    converged: list[bool] = []
    search_caps: list[int] = []
    cap = config.max_components
    next_components = config.min_components
    previous: _HeterogeneousBackgroundGammaFit | None = None
    while True:
        for components in range(next_components, cap + 1):
            fit = _fit_heterogeneous_background_gamma(
                shells,
                shapes=shapes,
                ranks=config.ranks,
                components=components,
                config=config,
                previous=previous,
                workspace=workspace,
            )
            # One background shape/rate/slope triple, two parameters per signal
            # rate profile, and one free weight per signal state.
            parameter_count = 3 + 3 * (components - 1)
            entropy = -float(
                np.sum(
                    fit.responsibilities
                    * np.log(
                        np.maximum(fit.responsibilities, np.finfo(float).tiny)
                    )
                )
            )
            score = _information_score(
                fit.log_likelihood,
                fit.responsibilities,
                parameter_count,
                config.criterion,
            )
            fits.append(fit)
            component_grid.append(components)
            scores.append(float(score))
            log_likelihoods.append(float(fit.log_likelihood))
            entropies.append(entropy)
            iterations.append(int(fit.iterations))
            converged.append(bool(fit.converged))
            previous = fit
        search_caps.append(cap)
        selected_index = int(np.argmin(scores))
        selected_components = component_grid[selected_index]
        if not config.adaptive_components or not _search_needs_expansion(
            selected_components,
            cap,
            config.hard_max_components,
            config.boundary_margin,
        ):
            break
        next_components = cap + 1
        cap = min(config.hard_max_components, cap + config.expansion_step)

    selected = fits[selected_index]
    signal_components = selected_components - 1
    background_group = signal_components
    component_assignment = np.argmax(selected.responsibilities, axis=1)
    groups = component_assignment.astype(np.int32, copy=False)
    background_probability = selected.responsibilities[:, -1]
    signal_rates = selected.rates[:, 0] if signal_components else np.empty(0)
    log_rate_gaps = (
        np.log(np.maximum(signal_rates[:-1], config.min_rate))
        - np.log(np.maximum(signal_rates[1:], config.min_rate))
        if signal_components > 1
        else np.empty(0)
    )
    selected_parameter_count = 3 + 3 * signal_components
    bic = -2.0 * selected.log_likelihood + selected_parameter_count * math.log(
        graph.n_samples
    )
    fit_seconds = perf_counter() - fit_started
    hit_hard_max = bool(
        selected_components >= config.hard_max_components - config.boundary_margin
    )
    return StratificationResult(
        signatures=signatures,
        groups=groups,
        eps_by_group=np.zeros(selected_components, dtype=np.float32),
        point_eps=np.zeros(graph.n_samples, dtype=np.float32),
        supported_groups=np.arange(signal_components, dtype=np.int32),
        selected_components=selected_components,
        bic=float(bic),
        fit_indices=np.arange(graph.n_samples, dtype=np.int64),
        timings={
            "density_signatures": signature_seconds,
            "full_data_gamma_mdl": fit_seconds,
        },
        gap_ratios=np.exp(log_rate_gaps).astype(np.float32),
        selection_method=(
            f"adaptive_full_data_{config.criterion}_gamma_signals_with_"
            f"{config.background_distribution}_background"
        ),
        diagnostics={
            "mdl_gamma_ranks": list(map(int, config.ranks)),
            "mdl_gamma_criterion": config.criterion,
            "mdl_gamma_entropy_scope": "semantic",
            "mdl_gamma_candidate_components": component_grid,
            "mdl_gamma_candidate_scores": scores,
            "mdl_gamma_candidate_log_likelihoods": log_likelihoods,
            "mdl_gamma_candidate_entropies": entropies,
            "mdl_gamma_candidate_iterations": iterations,
            "mdl_gamma_candidate_converged": converged,
            "mdl_gamma_selected_components": selected_components,
            "mdl_gamma_signal_components": signal_components,
            "mdl_gamma_background_components": 1,
            "mdl_gamma_grouping": "explicit_semantic_background",
            "mdl_gamma_background_group": background_group,
            "mdl_gamma_assignment": "maximum_posterior_semantic_state",
            "mdl_gamma_background_boundary": "explicit_posterior",
            "mdl_gamma_background_distribution": config.background_distribution,
            **_background_parameter_diagnostics(
                selected.background_parameters, config
            ),
            "mdl_gamma_fit_samples": int(graph.n_samples),
            "mdl_gamma_weights": selected.weights.tolist(),
            "mdl_gamma_mu": selected.mu.tolist(),
            "mdl_gamma_slopes": selected.slopes.tolist(),
            "mdl_gamma_rates": selected.rates.tolist(),
            "mdl_gamma_log_rate_gaps": log_rate_gaps.tolist(),
            "mdl_gamma_background_fraction": float(
                np.mean(groups == background_group)
            ),
            "mdl_gamma_mean_background_probability": float(
                np.mean(background_probability)
            ),
            "mdl_gamma_mean_posterior_entropy": float(
                entropies[selected_index] / graph.n_samples
            ),
            "mdl_gamma_adaptive_search": bool(config.adaptive_components),
            "mdl_gamma_search_caps": search_caps,
            "mdl_gamma_search_initial_max": int(config.max_components),
            "mdl_gamma_search_hard_max": int(config.hard_max_components),
            "mdl_gamma_search_hit_hard_max": hit_hard_max,
        },
        mode="explicit_heterogeneous_background_information_criterion_gamma",
        background_probability=background_probability.astype(np.float32),
    )


def estimate_mdl_multiscale_stratification(
    graph: KNNGraph,
    *,
    ambient_dimension: float,
    config: GammaMDLConfig | None = None,
) -> StratificationResult:
    """Fit full-data Gamma candidates and select one information criterion."""

    cfg = config or GammaMDLConfig()
    _validate_gamma_config(graph, cfg)
    dimension = float(ambient_dimension)
    if not np.isfinite(dimension) or dimension <= 0.0:
        raise ValueError("ambient_dimension must be positive and finite")

    started = perf_counter()
    signatures = multiscale_density_signatures(graph, cfg.ranks)
    shells, shapes = _shell_volumes(graph, cfg.ranks, dimension)
    signature_seconds = perf_counter() - started
    if cfg.background_distribution != "gamma_components":
        return _estimate_heterogeneous_background_stratification(
            graph,
            signatures=signatures,
            shells=shells,
            shapes=shapes,
            signature_seconds=signature_seconds,
            config=cfg,
        )

    return _estimate_gamma_basis_stratification(
        graph,
        signatures=signatures,
        shells=shells,
        shapes=shapes,
        signature_seconds=signature_seconds,
        config=cfg,
    )


def _validate_config(graph: KNNGraph, config: OptimizationStrictCoreConfig) -> None:
    if config.core_rank < 1 or config.core_rank > graph.k:
        raise ValueError("core_rank must lie within the available k-NN graph")
    if config.connectivity_rank < 1 or config.connectivity_rank > graph.k:
        raise ValueError("connectivity_rank must lie within the available k-NN graph")
    if 2 * config.connectivity_rank > graph.k:
        raise ValueError(
            "optimization needs a disjoint connectivity-sized edge window "
            "for topology evidence"
        )


def _signal_log_odds(background_probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(background_probability, dtype=np.float64)
    tiny = np.finfo(np.float64).eps
    probability = np.clip(probability, tiny, 1.0 - tiny)
    return np.log1p(-probability) - np.log(probability)


def _uniform_lower_tail_log_bayes_factor(probabilities: np.ndarray) -> float:
    """Integrated evidence that uniform null probabilities concentrate near zero.

    Under the alternative, transformed probabilities follow ``Beta(q, 1)``
    with ``q`` integrated uniformly over ``[0, 1]``.  The null is Uniform(0,1).
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.size == 0:
        return -math.inf
    values = np.clip(values, np.finfo(float).tiny, 1.0)
    total = -float(np.sum(np.log(values)))
    n = int(values.size)
    if total <= np.finfo(float).eps:
        return -math.log(n + 1.0)
    regularized = float(gammainc(n + 1.0, total))
    if regularized <= 0.0:
        return -math.inf
    return (
        total
        + float(gammaln(n + 1.0))
        + math.log(regularized)
        - (n + 1.0) * math.log(total)
    )


def _background_boundary_gains(
    graph: KNNGraph,
    component_labels: np.ndarray,
    component_count: int,
    *,
    ambient_dimension: float,
    inner_rank: int,
) -> np.ndarray:
    """Code finite-component boundary evidence against a homogeneous Poisson null."""

    inner = graph.distances[:, inner_rank - 1].astype(np.float64, copy=False)
    outer = np.maximum(
        graph.distances[:, -1].astype(np.float64, copy=False),
        np.finfo(np.float64).tiny,
    )
    radius_fraction = np.clip(inner / outer, 0.0, 1.0)
    volume_fraction = np.power(radius_fraction, float(ambient_dimension))
    null_probability = betainc(
        float(inner_rank),
        float(graph.k - inner_rank),
        volume_fraction,
    )
    gains = np.full(component_count, -math.inf, dtype=np.float64)
    for component in range(component_count):
        rows = component_labels == component
        gains[component] = _uniform_lower_tail_log_bayes_factor(
            null_probability[rows]
        )
    return gains


def _intrinsic_dimension_mle(graph: KNNGraph, rows: np.ndarray) -> float:
    """Robust kNN maximum-likelihood intrinsic dimension on selected rows."""

    outer = np.maximum(
        graph.distances[rows, -1, None].astype(np.float64, copy=False),
        np.finfo(np.float64).tiny,
    )
    inner = np.maximum(
        graph.distances[rows, :-1].astype(np.float64, copy=False),
        np.finfo(np.float64).tiny,
    )
    denominator = np.sum(np.log(outer / inner), axis=1)
    local = (graph.k - 1.0) / np.maximum(denominator, np.finfo(float).tiny)
    finite = local[np.isfinite(local) & (local > 0.0)]
    return float(np.median(finite)) if finite.size else 1.0


def _largest_prefix_component(
    graph: KNNGraph,
    ordered_rows: np.ndarray,
    width: int,
    rank: int,
) -> np.ndarray:
    prefix = np.asarray(ordered_rows[:width], dtype=np.int64)
    local = np.full(graph.n_samples, -1, dtype=np.int64)
    local[prefix] = np.arange(prefix.size, dtype=np.int64)
    source = np.repeat(np.arange(prefix.size, dtype=np.int64), rank)
    target = local[graph.indices[prefix, :rank].reshape(-1)]
    keep = target >= 0
    adjacency = coo_matrix(
        (
            np.ones(int(np.sum(keep)), dtype=np.uint8),
            (source[keep], target[keep]),
        ),
        shape=(prefix.size, prefix.size),
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False, return_labels=True)
    chosen = int(np.argmax(np.bincount(labels)))
    return prefix[labels == chosen]


def _adaptive_background_core(
    graph: KNNGraph,
    background: np.ndarray,
    core_distance: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Select finite background components without a fixed rank percentage.

    Dyadic prefix sizes are compared with equal-width rank windows from the
    same background stratum.  A component must repay the code for choosing a
    width, activating a cluster, identifying a root, and following graph
    neighbours.  Boundary evidence uses the homogeneous-Poisson Beta law with
    intrinsic dimension estimated from the background graph itself.
    """

    from .strict_core import _rank_window_component_sizes

    original_rows = np.flatnonzero(background).astype(np.int64)
    selected = np.zeros(graph.n_samples, dtype=bool)
    point_gain = np.full(graph.n_samples, -np.inf, dtype=np.float64)
    if original_rows.size < 2 * rank:
        return selected, point_gain, {
            "intrinsic_dimension": 1.0,
            "events": [],
            "total_gain": 0.0,
        }

    intrinsic_dimension = _intrinsic_dimension_mle(graph, original_rows)
    active = background.copy()
    events: list[dict[str, object]] = []
    total_gain = 0.0
    while True:
        rows = np.flatnonzero(active).astype(np.int64)
        widths: list[int] = []
        width = rank
        while 2 * width <= rows.size:
            widths.append(width)
            width *= 2
        if not widths:
            break
        order = rows[np.argsort(core_distance[rows], kind="stable")]
        best: tuple[float, int, float, int, int] | None = None
        for candidate_width in widths:
            sizes = _rank_window_component_sizes(
                np.asarray(graph.indices, dtype=np.int32),
                np.asarray(order, dtype=np.int32),
                int(candidate_width),
                int(rank),
            )
            if sizes.size < 2:
                continue
            observed = int(sizes[0])
            pvalue = (1.0 + float(np.sum(sizes[1:] >= observed))) / sizes.size
            window_gain = -math.log(pvalue) - math.log(len(widths))
            candidate = (
                window_gain,
                int(candidate_width),
                float(pvalue),
                observed,
                int(sizes.size),
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            break

        window_gain, width, pvalue, observed, windows = best
        activation_gain = window_gain - math.log(2.0)
        if activation_gain <= 0.0:
            break
        component = _largest_prefix_component(graph, order, width, rank)
        component_labels = np.full(graph.n_samples, -1, dtype=np.int32)
        component_labels[component] = 0
        boundary_gain = float(
            _background_boundary_gains(
                graph,
                component_labels,
                1,
                ambient_dimension=intrinsic_dimension,
                inner_rank=rank,
            )[0]
        )
        graph_code = (
            math.log(rows.size)
            + max(component.size - 1, 0) * math.log(rank)
            + math.log(len(widths))
        )
        gain = activation_gain + boundary_gain - graph_code
        if not np.isfinite(gain) or gain <= 0.0:
            break

        selected[component] = True
        active[component] = False
        point_gain[component] = gain / component.size
        total_gain += gain
        events.append(
            {
                "width": width,
                "component_size": int(component.size),
                "window_pvalue": pvalue,
                "window_count": windows,
                "window_gain_nats": float(window_gain),
                "boundary_gain_nats": boundary_gain,
                "graph_code_nats": float(graph_code),
                "description_gain_nats": float(gain),
            }
        )

    return selected, point_gain, {
        "intrinsic_dimension": intrinsic_dimension,
        "events": events,
        "total_gain": float(total_gain),
    }


@njit(cache=True)
def _find(parent: np.ndarray, item: int) -> int:
    root = item
    while parent[root] != root:
        root = parent[root]
    while parent[item] != item:
        next_item = parent[item]
        parent[item] = root
        item = next_item
    return root


@njit(cache=True)
def _topology_log_bayes_factor(
    size: int,
    successes: int,
    n_reference: int,
    rank: int,
) -> float:
    if size < 2 or successes < 1 or n_reference < 2:
        return -math.inf
    trials = size * rank
    if successes > trials:
        return -math.inf
    null_probability = (size - 1.0) / (n_reference - 1.0)
    observed_probability = successes / trials
    if observed_probability <= null_probability:
        return -math.inf
    failures = trials - successes
    log_alternative = (
        math.lgamma(successes + 0.5)
        + math.lgamma(failures + 0.5)
        - math.lgamma(trials + 1.0)
        - (math.lgamma(0.5) + math.lgamma(0.5) - math.lgamma(1.0))
    )
    if null_probability <= 0.0:
        log_null = 0.0 if successes == 0 else -math.inf
    elif null_probability >= 1.0:
        log_null = 0.0 if failures == 0 else -math.inf
    else:
        log_null = (
            successes * math.log(null_probability)
            + failures * math.log1p(-null_probability)
        )
    return log_alternative - log_null


@njit(cache=True)
def _component_gain_scalar(
    size: int,
    successes: int,
    unary_gain: float,
    n_reference: int,
    rank: int,
) -> float:
    log_bayes_factor = _topology_log_bayes_factor(
        size, successes, n_reference, rank
    )
    if not math.isfinite(log_bayes_factor):
        return -math.inf
    # A retained component is identified by one representative vertex. The
    # corresponding uniform root code costs log(n), which prevents a threshold
    # from manufacturing many unsupported micro-components without introducing
    # a minimum cluster size.
    return unary_gain + log_bayes_factor - math.log(n_reference)


@njit(cache=True)
def _best_threshold_event_sweep(
    unary_gain: np.ndarray,
    event_steps: np.ndarray,
    event_left: np.ndarray,
    event_right: np.ndarray,
    n_reference: int,
    rank: int,
) -> tuple[int, float]:
    """Optimize one stratum exactly over its nested core-distance prefixes."""

    n = unary_gain.size
    if n == 0:
        return 0, 0.0
    successes = 0
    unary = 0.0
    best_gain = 0.0
    best_size = 0
    event = 0
    threshold_code = math.log(n)

    for step in range(n):
        unary += unary_gain[step]
        while event < event_steps.size and event_steps[event] == step:
            successes += 1
            event += 1

        topology_gain = _topology_log_bayes_factor(
            step + 1, successes, n_reference, rank
        )
        candidate_gain = unary + topology_gain - threshold_code
        if candidate_gain > best_gain:
            best_gain = candidate_gain
            best_size = step + 1

    return best_size, best_gain


def _stratum_events(
    graph: KNNGraph,
    rows: np.ndarray,
    order: np.ndarray,
    rank: int,
    *,
    edge_offset: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directed within-stratum edges ordered by activation event."""

    n_total = graph.n_samples
    offset = rank if edge_offset is None else int(edge_offset)
    if offset < 0 or offset + rank > graph.k:
        raise ValueError("stratum evidence window lies outside the k-NN graph")
    local = np.full(n_total, -1, dtype=np.int64)
    local[rows[order]] = np.arange(rows.size, dtype=np.int64)
    # The first ``rank`` neighbours define StrictCore connectivity.  Use the
    # next equally sized window as held-out topology evidence so that a
    # component is not rewarded merely for the edges that constructed it.
    source = np.repeat(np.arange(rows.size, dtype=np.int64), rank)
    target = local[graph.indices[rows[order], offset : offset + rank].reshape(-1)]
    keep = target >= 0
    source = source[keep]
    target = target[keep]
    steps = np.maximum(source, target)
    event_order = np.argsort(steps, kind="stable")
    return steps[event_order], source[event_order], target[event_order]


def _component_statistics(
    graph: KNNGraph,
    candidate: np.ndarray,
    unary_gain: np.ndarray,
    rank: int,
    n_reference: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label candidate components and calculate their MDL gains."""

    rows = np.flatnonzero(candidate).astype(np.int64)
    labels = np.full(graph.n_samples, -1, dtype=np.int32)
    if rows.size == 0:
        return labels, np.empty(0), np.empty(0)
    local = np.full(graph.n_samples, -1, dtype=np.int64)
    local[rows] = np.arange(rows.size, dtype=np.int64)
    source = np.repeat(np.arange(rows.size, dtype=np.int64), rank)
    target = local[graph.indices[rows, :rank].reshape(-1)]
    keep = target >= 0
    adjacency = coo_matrix(
        (
            np.ones(int(np.sum(keep)), dtype=np.uint8),
            (source[keep], target[keep]),
        ),
        shape=(rows.size, rows.size),
    ).tocsr()
    count, local_labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    labels[rows] = local_labels.astype(np.int32, copy=False)
    sizes = np.bincount(local_labels, minlength=count).astype(np.int64)
    # Score a component on a disjoint edge window.  Reusing its construction
    # edges would make connectedness tautological and systematically reward
    # micro-components.
    neighbour_labels = labels[graph.indices[rows, rank : 2 * rank]]
    successes = np.bincount(
        local_labels,
        weights=np.sum(neighbour_labels == local_labels[:, None], axis=1),
        minlength=count,
    ).astype(np.int64)
    unary = np.bincount(
        local_labels, weights=unary_gain[rows], minlength=count
    ).astype(np.float64)
    gains = np.asarray(
        [
            _component_gain_scalar(
                int(size), int(success), float(prize), n_reference, rank
            )
            for size, success, prize in zip(sizes, successes, unary, strict=True)
        ],
        dtype=np.float64,
    )
    return labels, gains, sizes.astype(np.float64)


def _density_ordered_labels_from_core(
    graph: KNNGraph,
    core: np.ndarray,
    core_radius: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Grow lower-density core components without merging denser clusters.

    Gamma groups are ordered from high to low rate.  A component with no
    contact to an earlier group starts a cluster.  Otherwise graph-distance
    Voronoi growth assigns it to the denser clusters that reach it, preserving
    density valleys instead of letting a lower-density bridge merge clusters.
    """

    labels = np.full(graph.n_samples, -1, dtype=np.int64)
    next_label = 0
    for group in map(int, np.unique(groups[core])):
        rows = np.flatnonzero(core & (groups == group)).astype(np.int64)
        if rows.size == 0:
            continue
        local = np.full(graph.n_samples, -1, dtype=np.int64)
        local[rows] = np.arange(rows.size, dtype=np.int64)
        source = np.repeat(np.arange(rows.size, dtype=np.int64), graph.k)
        source_rows = np.repeat(rows, graph.k)
        neighbour_rows = graph.indices[rows].reshape(-1)
        target = local[neighbour_rows]
        edge_distance = graph.distances[rows].reshape(-1).astype(
            np.float64, copy=False
        )
        keep = (target >= 0) & (edge_distance <= core_radius[source_rows])
        adjacency = coo_matrix(
            (edge_distance[keep], (source[keep], target[keep])),
            shape=(rows.size, rows.size),
        ).tocsr()
        count, components = connected_components(
            adjacency, directed=False, return_labels=True
        )
        for component in range(count):
            members = np.flatnonzero(components == component).astype(np.int64)
            component_rows = rows[members]
            neighbours = graph.indices[component_rows]
            neighbour_labels = labels[neighbours]
            eligible = (neighbour_labels >= 0) & (
                graph.distances[component_rows] <= core_radius[neighbours]
            )
            seeds = np.flatnonzero(np.any(eligible, axis=1)).astype(np.int64)
            if seeds.size == 0:
                labels[component_rows] = next_label
                next_label += 1
                continue

            distances = np.where(
                eligible,
                graph.distances[component_rows],
                np.inf,
            )
            nearest = np.argmin(distances[seeds], axis=1)
            seed_labels = neighbour_labels[seeds, nearest]
            subgraph = adjacency[members][:, members]
            _, _, sources = dijkstra(
                subgraph,
                directed=False,
                indices=seeds,
                return_predecessors=True,
                min_only=True,
            )
            seed_label_by_row = np.full(members.size, -1, dtype=np.int64)
            seed_label_by_row[seeds] = seed_labels
            assigned = seed_label_by_row[np.asarray(sources, dtype=np.int64)]
            labels[component_rows] = assigned
    return labels, next_label


def optimize_strict_core_from_graph(
    graph: KNNGraph,
    stratification: StratificationResult,
    *,
    ambient_dimension: float = 1.0,
    config: OptimizationStrictCoreConfig | None = None,
) -> OptimizationStrictCoreResult:
    """Optimize Gamma-stratum StrictCore radii and border/noise decisions."""

    cfg = config or OptimizationStrictCoreConfig()
    _validate_config(graph, cfg)
    if stratification.groups.shape != (graph.n_samples,):
        raise ValueError("stratification groups must align with the graph")
    if stratification.background_probability is None:
        raise ValueError("optimization requires Gamma background probabilities")

    started = perf_counter()
    groups = np.asarray(stratification.groups, dtype=np.int32)
    background_probability = np.asarray(
        stratification.background_probability, dtype=np.float64
    )
    unary_gain = _signal_log_odds(background_probability)
    core_distance = np.asarray(
        graph.distances[:, cfg.core_rank - 1], dtype=np.float64
    )
    core = np.zeros(graph.n_samples, dtype=bool)
    core_radius = core_distance.copy()

    background_value = stratification.diagnostics.get("mdl_gamma_background_group")
    background_group = int(background_value) if background_value is not None else None
    candidate_groups = np.unique(groups)
    reported_groups = candidate_groups.astype(int).tolist()

    thresholds: list[float] = []
    threshold_sizes: list[int] = []
    group_sizes: list[int] = []
    group_gains: list[float] = []
    selected_components: list[int] = []
    selected_groups: list[int] = []
    ordering_modes: list[str] = []
    selected_component_gains: list[float] = []
    boundary_component_gains: list[list[float]] = []
    background_scan_profile: dict[str, object] = {}

    for group in map(int, candidate_groups):
        rows = np.flatnonzero(groups == group).astype(np.int64)
        group_sizes.append(int(rows.size))
        if rows.size == 0:
            thresholds.append(0.0)
            threshold_sizes.append(0)
            group_gains.append(0.0)
            selected_components.append(0)
            continue
        is_background = background_group is not None and group == background_group
        if is_background:
            ordering_modes.append("adaptive_d4_rank_windows_with_beta_boundary_code")
            background_core, _, background_scan_profile = (
                _adaptive_background_core(
                    graph,
                    groups == group,
                    core_distance,
                    rank=cfg.connectivity_rank,
                )
            )
            events = list(background_scan_profile["events"])
            threshold_sizes.append(int(np.sum(background_core)))
            thresholds.append(
                float(np.max(core_distance[background_core]))
                if np.any(background_core)
                else 0.0
            )
            group_gains.append(float(background_scan_profile["total_gain"]))
            selected_components.append(len(events))
            boundary_component_gains.append(
                [float(event["boundary_gain_nats"]) for event in events]
            )
            if np.any(background_core):
                core |= background_core
                selected_groups.append(group)
                selected_component_gains.extend(
                    float(event["description_gain_nats"]) for event in events
                )
            continue
        ordering_score = core_distance
        ordering_modes.append("d4")
        group_unary_gain = unary_gain
        order = np.argsort(ordering_score[rows], kind="stable")
        ordered_rows = rows[order]
        event_steps, event_left, event_right = _stratum_events(
            graph, rows, order, cfg.connectivity_rank
        )
        reference_size = graph.n_samples
        best_size, best_gain = _best_threshold_event_sweep(
            group_unary_gain[ordered_rows],
            event_steps,
            event_left,
            event_right,
            reference_size,
            cfg.connectivity_rank,
        )
        threshold_sizes.append(int(best_size))
        group_gains.append(float(best_gain))
        if best_size == 0:
            thresholds.append(0.0)
            selected_components.append(0)
            continue

        threshold = float(ordering_score[ordered_rows[best_size - 1]])
        thresholds.append(threshold)
        candidate = (groups == group) & (ordering_score <= threshold)
        local_labels, gains, _ = _component_statistics(
            graph,
            candidate,
            group_unary_gain,
            cfg.connectivity_rank,
            reference_size,
        )
        boundary_gains = np.zeros(gains.size, dtype=np.float64)
        boundary_component_gains.append(boundary_gains.astype(float).tolist())
        keep_components = np.flatnonzero(gains > 0.0)
        keep = np.isin(local_labels, keep_components) & (local_labels >= 0)
        if not np.any(keep):
            selected_components.append(0)
            continue
        core |= keep
        core_radius[keep] = threshold
        selected_components.append(int(keep_components.size))
        selected_groups.append(group)
        selected_component_gains.extend(gains[keep_components].astype(float).tolist())

    labels, initial_clusters = _density_ordered_labels_from_core(
        graph, core, core_radius, groups
    )
    border_started = perf_counter()
    border = np.flatnonzero(~core).astype(np.int64)
    attached = np.zeros(graph.n_samples, dtype=bool)
    attachment_gain = np.zeros(graph.n_samples, dtype=np.float64)
    border_model = "no_core_components"
    if border.size and initial_clusters:
        neighbours = graph.indices[border, : cfg.connectivity_rank]
        neighbour_labels = labels[neighbours]
        valid = neighbour_labels >= 0
        ratios = np.full(valid.shape, np.inf, dtype=np.float64)
        ratios[valid] = (
            graph.distances[border, : cfg.connectivity_rank]
            / np.maximum(
                core_radius[neighbours],
                np.finfo(np.float32).tiny,
            )
        )[valid]
        best_column = np.argmin(ratios, axis=1)
        best_ratio = ratios[np.arange(border.size), best_column]
        has_core_neighbour = np.isfinite(best_ratio)
        local_attach = has_core_neighbour & (best_ratio <= 1.0)
        chosen = neighbours[np.arange(border.size), best_column]
        attach = local_attach
        candidate_attachment_gain = -np.log(
            np.maximum(best_ratio, np.finfo(np.float64).tiny)
        )
        border_model = "strict_local_core_radius"
        labels[border[attach]] = labels[chosen[attach]]
        attached[border[attach]] = True
        attachment_gain[border[attach]] = candidate_attachment_gain[attach]

    total_gain = float(
        np.sum(selected_component_gains)
        + np.sum(attachment_gain)
    )
    elapsed = perf_counter() - started
    return OptimizationStrictCoreResult(
        labels=labels,
        core_mask=core,
        profile={
            "optimization_solver": "exact_stratum_event_sweep",
            "optimization_objective": "gamma_topology_component_and_attachment_description_gain",
            "optimization_topology_evidence_ranks": [
                int(cfg.connectivity_rank + 1),
                int(2 * cfg.connectivity_rank),
            ],
            "optimization_objective_trace": [0.0, -total_gain],
            "optimization_description_gain_nats": total_gain,
            "optimization_core_rank": int(cfg.core_rank),
            "optimization_connectivity_rank": int(cfg.connectivity_rank),
            "optimization_candidate_groups": reported_groups,
            "optimization_selected_groups": selected_groups,
            "optimization_group_sizes": group_sizes,
            "optimization_threshold_sizes": threshold_sizes,
            "optimization_thresholds": thresholds,
            "optimization_ordering_modes": ordering_modes,
            "optimization_group_gains_nats": group_gains,
            "optimization_selected_component_gains_nats": selected_component_gains,
            "optimization_boundary_component_gains_nats": boundary_component_gains,
            "optimization_selected_local_components": selected_components,
            "optimization_core_fraction": float(np.mean(core)),
            "optimization_border_fraction": float(np.mean(attached)),
            "optimization_border_model": border_model,
            "optimization_noise_fraction": float(np.mean(labels < 0)),
            "optimization_clusters": int(np.unique(labels[labels >= 0]).size),
            "optimization_threshold_seconds": float(border_started - started),
            "optimization_border_seconds": float(perf_counter() - border_started),
            "optimization_seconds": float(elapsed),
            "optimization_sparse_memory": True,
            "optimization_unified_background_scan": bool(
                background_group is not None
            ),
            "optimization_background_intrinsic_dimension": background_scan_profile.get(
                "intrinsic_dimension", float("nan")
            ),
            "optimization_background_events": background_scan_profile.get("events", []),
        },
    )


class OptimizationStrataSCAN(GraphStrataSCAN):
    """StrataSCAN 0.2.1 semantic-MDL Gamma plus unified StrictCore estimator."""

    def __init__(
        self,
        *,
        optimization_config: OptimizationStrictCoreConfig | None = None,
        gamma_config: GammaMDLConfig | None = None,
        k: int = 32,
        backend: Backend = "auto",
        n_jobs: int = 1,
        ambient_dimension: float | None = None,
        hnsw_config: HNSWConfig | None = None,
    ) -> None:
        super().__init__(
            k=k,
            backend=backend,
            n_jobs=n_jobs,
            ambient_dimension=ambient_dimension,
            hnsw_config=hnsw_config,
        )
        self.optimization_config = optimization_config or OptimizationStrictCoreConfig()
        self.gamma_config = gamma_config or GammaMDLConfig()

    def fit_from_graph(
        self,
        graph: KNNGraph,
        *,
        ambient_dimension: float | None = None,
    ) -> "OptimizationStrataSCAN":
        dimension = self.ambient_dimension if ambient_dimension is None else ambient_dimension
        if dimension is None:
            dimension = 1.0
        stratification = estimate_mdl_multiscale_stratification(
            graph,
            ambient_dimension=float(dimension),
            config=self.gamma_config,
        )
        result = optimize_strict_core_from_graph(
            graph,
            stratification,
            ambient_dimension=float(dimension),
            config=self.optimization_config,
        )
        self.labels_ = result.labels
        self.core_sample_indices_ = np.flatnonzero(result.core_mask)
        self.n_clusters_ = int(np.unique(result.labels[result.labels >= 0]).size)
        self.stratification_ = stratification
        self.profile_ = {
            "algorithm_version": "0.2.1",
            "algorithm": "OptimizationStrataSCAN",
            "stratification": "adaptive-semantic-information-criterion-gamma-v2",
            **stratification.diagnostics,
            **result.profile,
        }
        self.graph_ = graph
        return self
