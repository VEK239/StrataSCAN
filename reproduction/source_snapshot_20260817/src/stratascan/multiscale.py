from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.special import gammaln, logsumexp

from .stratification import StratificationResult
from .types import KNNGraph


@dataclass(slots=True)
class MultiscaleConfig:
    """Configuration for a multi-shell local-Poisson Gamma mixture."""

    ranks: tuple[int, ...] = (4, 8, 16, 32)
    min_components: int = 2
    max_components: int = 8
    min_tail_mass: float = 0.20
    min_dense_mass: float = 0.01
    background_posterior_threshold: float = 0.50
    max_fit_samples: int = 10_000
    max_iter: int = 300
    tolerance: float = 1e-7
    min_rate: float = 1e-14
    max_rate: float = 1e14
    random_state: int = 42


@dataclass(slots=True)
class _MultiscaleGammaFit:
    bic: float
    weights: np.ndarray
    rates: np.ndarray
    responsibilities: np.ndarray
    iterations: int


def multiscale_density_signatures(
    graph: KNNGraph,
    ranks: tuple[int, ...] = (4, 8, 16, 32),
) -> np.ndarray:
    """Return log(d4) and anchored log distance-ratio diagnostics."""

    selected = np.asarray(ranks, dtype=np.int64)
    if selected.ndim != 1 or selected.size < 2:
        raise ValueError("at least two multiscale ranks are required")
    if selected[0] < 1 or selected[-1] > graph.k or np.any(np.diff(selected) <= 0):
        raise ValueError("multiscale ranks must be strictly increasing within the graph")
    tiny = np.finfo(np.float32).tiny
    log_distances = np.log(
        np.maximum(graph.distances[:, selected - 1], tiny).astype(np.float64)
    )
    return np.column_stack(
        [log_distances[:, 0], log_distances[:, [0]] - log_distances[:, 1:]]
    ).astype(np.float32)


def _shell_volumes(
    graph: KNNGraph,
    ranks: tuple[int, ...],
    dimension: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nested-volume increments and their fixed Gamma shapes.

    In a homogeneous Poisson process, volume increments between neighbour
    ranks are independent Gamma variables with shapes equal to rank increments.
    """

    selected = np.asarray(ranks, dtype=np.int64)
    distances = np.maximum(
        graph.distances[:, selected - 1].astype(np.float64),
        np.finfo(np.float64).tiny,
    )
    log_distances = np.log(distances)
    centered = dimension * (log_distances - np.median(log_distances[:, 0]))
    cumulative = np.exp(np.clip(centered, -60.0, 60.0))
    shells = np.diff(
        np.column_stack([np.zeros(cumulative.shape[0]), cumulative]), axis=1
    )
    shells = np.maximum(shells, np.finfo(np.float64).tiny)
    shapes = np.diff(np.concatenate([[0], selected])).astype(np.float64)
    return shells, shapes


def _quantile_preserving_subsample(
    signatures: np.ndarray,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    n = signatures.shape[0]
    if n <= max_samples:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(signatures[:, 0], kind="stable")
    offset = float(np.random.default_rng(random_state).random())
    positions = np.floor(
        (np.arange(max_samples, dtype=np.float64) + offset) * n / max_samples
    ).astype(np.int64)
    return np.sort(order[np.minimum(positions, n - 1)]).astype(np.int64)


def _responsibilities(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    weights: np.ndarray,
    rates: np.ndarray,
    config: MultiscaleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    log_shells = np.log(np.maximum(shells, np.finfo(np.float64).tiny))
    log_probability = np.log(np.maximum(weights, np.finfo(float).tiny))[None, :]
    log_probability = log_probability + np.sum(
        shapes[None, None, :] * np.log(np.maximum(rates, config.min_rate))[None, :, :]
        - gammaln(shapes)[None, None, :]
        + (shapes[None, None, :] - 1.0) * log_shells[:, None, :]
        - shells[:, None, :] * rates[None, :, :],
        axis=2,
    )
    log_norm = logsumexp(log_probability, axis=1)
    return np.exp(log_probability - log_norm[:, None]), log_norm


def _fit_multiscale_gamma(
    shells: np.ndarray,
    *,
    shapes: np.ndarray,
    components: int,
    config: MultiscaleConfig,
) -> _MultiscaleGammaFit:
    score = np.log(shells[:, 0])
    edges = np.quantile(score, np.linspace(0.0, 1.0, components + 1))
    labels = np.searchsorted(edges[1:-1], score, side="right")
    weights = np.bincount(labels, minlength=components).astype(np.float64)
    weights /= weights.sum()
    rates = np.empty((components, shells.shape[1]), dtype=np.float64)
    for group in range(components):
        group_shells = shells[labels == group]
        mean_shells = np.mean(group_shells, axis=0)
        rates[group] = shapes / np.maximum(mean_shells, np.finfo(float).tiny)

    previous = -np.inf
    log_likelihood = -np.inf
    responsibilities = np.empty((shells.shape[0], components), dtype=np.float64)
    for iteration in range(config.max_iter):
        responsibilities, log_norm = _responsibilities(
            shells,
            shapes=shapes,
            weights=weights,
            rates=rates,
            config=config,
        )
        log_likelihood = float(np.sum(log_norm))
        effective = np.maximum(np.sum(responsibilities, axis=0), np.finfo(float).tiny)
        weights = effective / shells.shape[0]
        weighted_shells = responsibilities.T @ shells
        rates = shapes[None, :] * effective[:, None] / np.maximum(
            weighted_shells, np.finfo(float).tiny
        )
        rates = np.clip(rates, config.min_rate, config.max_rate)
        if np.isfinite(previous) and (
            abs(log_likelihood - previous)
            <= config.tolerance * (1.0 + abs(previous))
        ):
            break
        previous = log_likelihood

    # d4 determines the working epsilon, so order strata by their first-shell
    # density rate while the full four-shell likelihood determines membership.
    order = np.argsort(rates[:, 0], kind="stable")[::-1]
    weights = weights[order]
    rates = rates[order]
    responsibilities = responsibilities[:, order]
    parameter_count = (components - 1) + components * shells.shape[1]
    bic = -2.0 * log_likelihood + parameter_count * np.log(shells.shape[0])
    return _MultiscaleGammaFit(
        bic, weights, rates, responsibilities, iteration + 1
    )


def estimate_multiscale_stratification(
    graph: KNNGraph,
    *,
    ambient_dimension: float,
    config: MultiscaleConfig | None = None,
) -> StratificationResult:
    """Fit a multi-shell Gamma mixture and retain the original tail logic."""

    cfg = config or MultiscaleConfig()
    if not 2 <= cfg.min_components <= cfg.max_components:
        raise ValueError("multiscale component bounds are invalid")
    if not 0.0 < cfg.min_tail_mass < 1.0 or not 0.0 < cfg.min_dense_mass < 1.0:
        raise ValueError("multiscale mass bounds must lie in (0, 1)")
    if not 0.0 < cfg.background_posterior_threshold < 1.0:
        raise ValueError("background_posterior_threshold must lie in (0, 1)")
    if cfg.max_fit_samples < 100:
        raise ValueError("max_fit_samples must be at least 100")
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

    fit_started = perf_counter()
    fits = [
        _fit_multiscale_gamma(
            fit_shells,
            shapes=shapes,
            components=components,
            config=cfg,
        )
        for components in range(cfg.min_components, cfg.max_components + 1)
    ]
    selected = min(fits, key=lambda fit: fit.bic)
    if fit_indices.size == graph.n_samples:
        responsibilities = selected.responsibilities
    else:
        responsibilities, _ = _responsibilities(
            shells,
            shapes=shapes,
            weights=selected.weights,
            rates=selected.rates,
            config=cfg,
        )

    density_rates = selected.rates[:, 0]
    gap_ratios = density_rates[:-1] / np.maximum(density_rates[1:], cfg.min_rate)
    tail_masses = np.array([
        float(np.sum(selected.weights[index + 1 :]))
        for index in range(selected.weights.size - 1)
    ])
    dense_masses = 1.0 - tail_masses
    eligible = np.flatnonzero(
        (tail_masses >= cfg.min_tail_mass) & (dense_masses >= cfg.min_dense_mass)
    )
    if eligible.size:
        cut = int(eligible[np.argmax(gap_ratios[eligible])] + 1)
    else:
        cut = int(max(1, selected.weights.size - 1))

    background_probability = np.sum(responsibilities[:, cut:], axis=1)
    active = background_probability < cfg.background_posterior_threshold
    dense_group = np.argmax(responsibilities[:, :cut], axis=1).astype(np.int32)
    groups = np.where(active, dense_group, cut).astype(np.int32)
    fit_seconds = perf_counter() - fit_started
    diagnostics = {
        "multiscale_ranks": list(map(int, cfg.ranks)),
        "multiscale_shell_shapes": shapes.astype(int).tolist(),
        "multiscale_features": [
            f"log_d{cfg.ranks[0]}",
            *[f"log_d{cfg.ranks[0]}_over_d{rank}" for rank in cfg.ranks[1:]],
        ],
        "multiscale_components": int(selected.weights.size),
        "multiscale_supported_components": cut,
        "multiscale_background_components": int(selected.weights.size - cut),
        "multiscale_background_fraction": float(np.mean(~active)),
        "multiscale_weights": selected.weights.tolist(),
        "multiscale_rates": selected.rates.tolist(),
        "multiscale_density_gap_ratios": gap_ratios.tolist(),
        "multiscale_tail_masses": tail_masses.tolist(),
        "multiscale_candidate_bics": [float(fit.bic) for fit in fits],
        "multiscale_candidate_iterations": [int(fit.iterations) for fit in fits],
        "multiscale_fit_samples": int(fit_indices.size),
    }
    return StratificationResult(
        signatures=signatures,
        groups=groups,
        eps_by_group=np.zeros(cut + 1, dtype=np.float32),
        point_eps=np.zeros(graph.n_samples, dtype=np.float32),
        supported_groups=np.arange(cut, dtype=np.int32),
        selected_components=cut + 1,
        bic=float(selected.bic),
        fit_indices=fit_indices,
        timings={
            "density_signatures": signature_seconds,
            "multiscale_gamma": fit_seconds,
        },
        gap_ratios=gap_ratios[:cut].astype(np.float32),
        selection_method="multishell_local_poisson_gamma_with_uniform_tail",
        diagnostics=diagnostics,
        mode="multiscale_gamma_uniform_tail",
        background_probability=background_probability.astype(np.float32),
    )
