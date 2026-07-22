from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.special import gammaln, logsumexp

from .stratification import StratificationResult, density_signatures
from .types import KNNGraph


@dataclass(slots=True)
class UniformTailConfig:
    """Configuration for a homogeneous-background local-volume mixture."""

    min_components: int = 2
    max_components: int = 8
    min_tail_mass: float = 0.20
    min_dense_mass: float = 0.01
    background_posterior_threshold: float = 0.50
    epsilon_quantile: float = 0.98
    max_iter: int = 300
    tolerance: float = 1e-7
    min_rate: float = 1e-14
    max_rate: float = 1e14
    max_fit_samples: int = 50_000
    random_state: int = 42


@dataclass(slots=True)
class _GammaMixtureFit:
    bic: float
    weights: np.ndarray
    rates: np.ndarray
    responsibilities: np.ndarray
    iterations: int


def _gamma_responsibilities(
    log_volume: np.ndarray,
    *,
    shape: float,
    weights: np.ndarray,
    rates: np.ndarray,
    config: UniformTailConfig,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.exp(np.clip(log_volume, -30.0, 30.0))
    log_probability = (
        np.log(np.maximum(weights, np.finfo(float).tiny))[None, :]
        + shape * np.log(np.maximum(rates, config.min_rate))[None, :]
        - gammaln(shape)
        + (shape - 1.0) * log_volume[:, None]
        - values[:, None] * rates[None, :]
    )
    log_norm = logsumexp(log_probability, axis=1)
    return np.exp(log_probability - log_norm[:, None]), log_norm


def estimate_intrinsic_dimension(graph: KNNGraph, *, rank: int = 16) -> float:
    """Estimate intrinsic dimension from neighbour-distance ratios."""
    k = min(max(3, rank), graph.k)
    distances = np.maximum(
        graph.distances[:, :k].astype(np.float64), np.finfo(np.float64).tiny
    )
    denominator = np.sum(np.log(distances[:, [-1]] / distances[:, :-1]), axis=1)
    valid = np.isfinite(denominator) & (denominator > 1e-12)
    if not np.any(valid):
        return 2.0
    estimates = (k - 1) / denominator[valid]
    return float(np.clip(np.median(estimates), 1.0, 64.0))


def _fit_gamma_mixture(
    log_volume: np.ndarray,
    *,
    components: int,
    shape: float,
    config: UniformTailConfig,
) -> _GammaMixtureFit:
    values = np.exp(np.clip(log_volume, -30.0, 30.0))
    edges = np.quantile(log_volume, np.linspace(0.0, 1.0, components + 1))
    labels = np.searchsorted(edges[1:-1], log_volume, side="right")
    weights = np.bincount(labels, minlength=components).astype(np.float64)
    weights /= weights.sum()
    rates = np.array([
        shape / max(float(np.mean(values[labels == group])), np.finfo(float).tiny)
        for group in range(components)
    ])
    previous = -np.inf
    log_likelihood = -np.inf
    responsibilities = np.empty((values.size, components), dtype=np.float64)
    for iteration in range(config.max_iter):
        responsibilities, log_norm = _gamma_responsibilities(
            log_volume,
            shape=shape,
            weights=weights,
            rates=rates,
            config=config,
        )
        log_likelihood = float(np.sum(log_norm))
        effective = np.maximum(
            np.sum(responsibilities, axis=0), np.finfo(float).tiny
        )
        weights = effective / values.size
        rates = shape * effective / np.maximum(
            responsibilities.T @ values, np.finfo(float).tiny
        )
        rates = np.clip(rates, config.min_rate, config.max_rate)
        if np.isfinite(previous) and (
            abs(log_likelihood - previous)
            <= config.tolerance * (1.0 + abs(previous))
        ):
            break
        previous = log_likelihood

    order = np.argsort(rates, kind="stable")[::-1]
    weights = weights[order]
    rates = rates[order]
    responsibilities = responsibilities[:, order]
    parameter_count = 2 * components - 1
    bic = -2.0 * log_likelihood + parameter_count * np.log(values.size)
    return _GammaMixtureFit(bic, weights, rates, responsibilities, iteration + 1)


def estimate_stratification_uniform_tail(
    graph: KNNGraph,
    *,
    density_ranks: tuple[int, ...] = (5, 10, 20, 32),
    eps_rank: int = 4,
    ambient_dimension: float | None = None,
    cached_signatures: np.ndarray | None = None,
    config: UniformTailConfig | None = None,
) -> StratificationResult:
    """Separate dense local-Poisson regimes from one aggregate uniform tail.

    For a homogeneous spatial Poisson process, the volume out to the k-th
    neighbour follows a Gamma distribution with shape k.  A mixture is fitted
    to ``d_k ** dimension``; all components beyond the strongest density-rate
    transition are aggregated into one background component instead of being
    retained as many operational strata.
    """
    cfg = config or UniformTailConfig()
    if eps_rank < 1 or eps_rank > graph.k:
        raise ValueError("eps_rank must be within [1, graph.k]")
    if not 2 <= cfg.min_components <= cfg.max_components:
        raise ValueError("uniform-tail component bounds are invalid")
    if not 0 < cfg.min_tail_mass < 1 or not 0 < cfg.min_dense_mass < 1:
        raise ValueError("uniform-tail mass bounds must lie in (0, 1)")
    if not 0 < cfg.background_posterior_threshold < 1:
        raise ValueError("background_posterior_threshold must lie in (0, 1)")
    if not 0 < cfg.epsilon_quantile <= 1:
        raise ValueError("epsilon_quantile must lie in (0, 1]")
    if cfg.max_fit_samples < 100:
        raise ValueError("max_fit_samples must be at least 100")

    started = perf_counter()
    signatures = (
        cached_signatures
        if cached_signatures is not None
        else density_signatures(graph, density_ranks)
    )
    dimension = (
        estimate_intrinsic_dimension(graph)
        if ambient_dimension is None
        else float(ambient_dimension)
    )
    if not np.isfinite(dimension) or dimension <= 0:
        raise ValueError("ambient_dimension must be positive and finite")
    kth = np.maximum(
        graph.distances[:, eps_rank - 1].astype(np.float64),
        np.finfo(np.float64).tiny,
    )
    log_distance = np.log(kth)
    log_volume = dimension * (log_distance - np.median(log_distance))
    if log_volume.size > cfg.max_fit_samples:
        fit_indices = np.sort(
            np.random.default_rng(cfg.random_state).choice(
                log_volume.size,
                size=cfg.max_fit_samples,
                replace=False,
            )
        )
    else:
        fit_indices = np.arange(log_volume.size, dtype=np.int64)
    fit_log_volume = log_volume[fit_indices]

    fit_started = perf_counter()
    fits = [
        _fit_gamma_mixture(
            fit_log_volume,
            components=components,
            shape=float(eps_rank),
            config=cfg,
        )
        for components in range(cfg.min_components, cfg.max_components + 1)
    ]
    selected = min(fits, key=lambda fit: fit.bic)
    rates = selected.rates
    weights = selected.weights
    if fit_indices.size == log_volume.size:
        responsibilities = selected.responsibilities
    else:
        responsibilities, _ = _gamma_responsibilities(
            log_volume,
            shape=float(eps_rank),
            weights=weights,
            rates=rates,
            config=cfg,
        )
    ratios = rates[:-1] / np.maximum(rates[1:], cfg.min_rate)
    tail_masses = np.array([
        float(np.sum(weights[index + 1:])) for index in range(weights.size - 1)
    ])
    dense_masses = 1.0 - tail_masses
    eligible = np.flatnonzero(
        (tail_masses >= cfg.min_tail_mass) & (dense_masses >= cfg.min_dense_mass)
    )
    if eligible.size:
        cut = int(eligible[np.argmax(ratios[eligible])] + 1)
    else:
        cut = int(max(1, weights.size - 1))

    background_probability = np.sum(responsibilities[:, cut:], axis=1)
    active = background_probability < cfg.background_posterior_threshold
    dense_responsibility = responsibilities[:, :cut]
    dense_group = np.argmax(dense_responsibility, axis=1).astype(np.int32)
    background_group = cut
    groups = np.where(active, dense_group, background_group).astype(np.int32)

    eps = np.zeros(cut + 1, dtype=np.float32)
    for group in range(cut):
        values = kth[active & (dense_group == group)]
        if values.size:
            eps[group] = np.quantile(values, cfg.epsilon_quantile)
    point_eps = eps[groups]
    supported = np.arange(cut, dtype=np.int32)
    timings = {
        "uniform_tail_mixture": perf_counter() - fit_started,
        "density_signatures": 0.0 if cached_signatures is not None else perf_counter() - started,
    }
    diagnostics = {
        "epsilon_estimator": "uniform_tail_quantile",
        "uniform_tail_dimension": float(dimension),
        "uniform_tail_components": int(weights.size),
        "uniform_tail_dense_components": int(cut),
        "uniform_tail_weights": weights.tolist(),
        "uniform_tail_rates": rates.tolist(),
        "uniform_tail_rate_ratios": ratios.tolist(),
        "uniform_tail_tail_masses": tail_masses.tolist(),
        "uniform_tail_background_fraction": float(np.mean(~active)),
        "uniform_tail_background_threshold": float(cfg.background_posterior_threshold),
        "uniform_tail_epsilon_quantile": float(cfg.epsilon_quantile),
        "uniform_tail_candidate_bics": [float(fit.bic) for fit in fits],
        "uniform_tail_iterations": [int(fit.iterations) for fit in fits],
        "uniform_tail_fit_samples": int(fit_indices.size),
    }
    return StratificationResult(
        signatures=signatures,
        groups=groups,
        eps_by_group=eps,
        point_eps=point_eps,
        supported_groups=supported,
        selected_components=cut + 1,
        bic=float(selected.bic),
        fit_indices=fit_indices,
        timings=timings,
        gap_ratios=ratios[:cut].astype(np.float32),
        selection_method="local_poisson_gamma_with_uniform_tail",
        diagnostics=diagnostics,
        mode="uniform_poisson_tail",
        background_probability=background_probability.astype(np.float32),
    )
