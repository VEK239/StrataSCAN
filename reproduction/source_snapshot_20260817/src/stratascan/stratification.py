from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import KNNGraph


@dataclass(slots=True)
class StratificationResult:
    signatures: np.ndarray
    groups: np.ndarray
    eps_by_group: np.ndarray
    point_eps: np.ndarray
    supported_groups: np.ndarray
    selected_components: int
    bic: float
    fit_indices: np.ndarray
    timings: dict[str, float]
    gap_ratios: np.ndarray | None = None
    selection_method: str = "local_poisson_gamma_with_uniform_tail"
    diagnostics: dict = field(default_factory=dict)
    mode: str = "uniform_poisson_tail"
    background_probability: np.ndarray | None = None


def density_signatures(graph: KNNGraph, density_ranks: tuple[int, ...]) -> np.ndarray:
    ranks = np.asarray(density_ranks, dtype=np.int64)
    if np.any(ranks < 1) or np.any(ranks > graph.k):
        raise ValueError("density ranks must be within [1, graph.k]")
    return np.log(
        np.maximum(graph.distances[:, ranks - 1], np.finfo(np.float32).tiny)
    ).astype(np.float32)
