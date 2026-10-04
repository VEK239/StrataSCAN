from __future__ import annotations
import numpy as np
from .types import KNNGraph
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
