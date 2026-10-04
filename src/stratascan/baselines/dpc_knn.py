from __future__ import annotations

"""Controlled implementation of the Du--Ding--Jia (2016) DPC-kNN core.

The published method leaves cluster-centre selection to a manual decision
graph.  This benchmark variant combines that core with the automatic,
gap-based centre selector of Flores and Garza (2020).  It is therefore named
as a hybrid in its metadata and must not be described as the authors' code.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist


METHOD_ID = "DPC-kNN-2016+GB-auto-p2pct"
NEIGHBOUR_FRACTION = 0.02


@dataclass(slots=True)
class DPCkNNState:
    rho: np.ndarray
    delta: np.ndarray
    parent: np.ndarray
    density_order: np.ndarray
    centers: np.ndarray
    labels: np.ndarray
    gap_threshold: float | None
    center_fallback_used: bool


@dataclass(slots=True)
class DPCkNNResult:
    labels: np.ndarray
    metadata: dict[str, Any]
    profile: dict[str, Any]


def _validate_input(X: np.ndarray, block_size: int) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if values.shape[0] == 0:
        raise ValueError("X must contain at least one row")
    if not np.all(np.isfinite(values)):
        raise ValueError("X must contain only finite values")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    return values


def _knn_density(
    X: np.ndarray, *, k: int, block_size: int
) -> np.ndarray:
    """Equation (7) of Du et al., using exact Euclidean neighbours."""
    n = X.shape[0]
    log_rho = np.empty(n, dtype=np.float64)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        distances = cdist(X[start:stop], X, metric="euclidean")
        rows = np.arange(stop - start)
        distances[rows, np.arange(start, stop)] = np.inf
        nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
        log_rho[start:stop] = -np.mean(nearest * nearest, axis=1)

    # A common multiplicative rescaling of rho also rescales every gamma and
    # gap.  It preserves the decision graph ordering while preventing needless
    # underflow when coordinates have a large scale.
    log_rho -= float(np.max(log_rho))
    return np.exp(log_rho)


def _global_delta_and_parent(
    X: np.ndarray, rho: np.ndarray, *, block_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find exact global nearest-higher-density parents in bounded memory.

    Du et al. do not specify how equal densities are handled.  We establish a
    total order by decreasing rho and then increasing original row index.
    Earlier rows in that order are treated as denser for parent selection.
    """
    n = X.shape[0]
    indices = np.arange(n, dtype=np.int64)
    order = np.lexsort((indices, -rho)).astype(np.int64, copy=False)
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n, dtype=np.int64)
    delta = np.empty(n, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)

    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        distances = cdist(X[start:stop], X, metric="euclidean")
        for offset, row in enumerate(range(start, stop)):
            rank = int(position[row])
            if rank == 0:
                delta[row] = float(np.max(distances[offset]))
                continue
            candidates = order[:rank]
            local = int(np.argmin(distances[offset, candidates]))
            parent[row] = int(candidates[local])
            delta[row] = float(distances[offset, parent[row]])
    return delta, parent, order


def _gb_auto_centers(
    rho: np.ndarray, delta: np.ndarray
) -> tuple[np.ndarray, float | None, bool]:
    """Flores--Garza gap-based automatic centre selection.

    Candidate set P1 contains points above the respective mean rho and delta.
    Candidates are sorted by decreasing gamma=rho*delta.  The final adjacent
    gamma gap at least as large as the mean gap is the cut.  The paper divides
    the sum of the |P1|-1 gaps by |P1|; that convention is retained here.
    """
    gamma = rho * delta
    candidates = np.flatnonzero(
        (rho > float(np.mean(rho))) & (delta > float(np.mean(delta)))
    ).astype(np.int64)
    fallback = False
    if candidates.size == 0:
        candidates = np.array([int(np.argmax(gamma))], dtype=np.int64)
        fallback = True

    candidate_order = np.lexsort((candidates, -gamma[candidates]))
    ranked = candidates[candidate_order]
    if ranked.size == 1:
        return ranked, None, fallback

    gaps = np.abs(np.diff(gamma[ranked]))
    mean_gap = float(np.sum(gaps) / ranked.size)
    qualifying = np.flatnonzero(gaps >= mean_gap)
    if qualifying.size == 0:
        return ranked[:1], mean_gap, True
    last_gap = int(qualifying[-1])
    return ranked[: last_gap + 1], mean_gap, fallback


def _assign_labels(
    parent: np.ndarray, density_order: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    labels = np.full(parent.size, -1, dtype=np.int64)
    for label, center in enumerate(centers):
        labels[int(center)] = label
    for row_value in density_order:
        row = int(row_value)
        if labels[row] >= 0:
            continue
        ancestor = int(parent[row])
        if ancestor < 0 or labels[ancestor] < 0:
            raise RuntimeError("density parent must be assigned before its child")
        labels[row] = labels[ancestor]
    return labels


def _fit_dpc_knn_2016(X: np.ndarray, *, block_size: int = 256) -> DPCkNNState:
    values = _validate_input(X, block_size)
    n = values.shape[0]
    if n == 1:
        return DPCkNNState(
            rho=np.ones(1, dtype=np.float64),
            delta=np.zeros(1, dtype=np.float64),
            parent=np.full(1, -1, dtype=np.int64),
            density_order=np.zeros(1, dtype=np.int64),
            centers=np.zeros(1, dtype=np.int64),
            labels=np.zeros(1, dtype=np.int64),
            gap_threshold=None,
            center_fallback_used=False,
        )

    k = min(n - 1, max(1, int(np.ceil(NEIGHBOUR_FRACTION * n))))
    rho = _knn_density(values, k=k, block_size=block_size)
    delta, parent, density_order = _global_delta_and_parent(
        values, rho, block_size=block_size
    )
    centers, gap_threshold, fallback = _gb_auto_centers(rho, delta)
    labels = _assign_labels(parent, density_order, centers)
    return DPCkNNState(
        rho=rho,
        delta=delta,
        parent=parent,
        density_order=density_order,
        centers=centers,
        labels=labels,
        gap_threshold=gap_threshold,
        center_fallback_used=fallback,
    )


def run_dpc_knn_2016(X: np.ndarray, *, block_size: int = 256) -> DPCkNNResult:
    """Run the locked p=2% DPC-kNN core with GB automatic centres."""
    values = _validate_input(X, block_size)
    state = _fit_dpc_knn_2016(values, block_size=block_size)
    n = values.shape[0]
    k = 0 if n == 1 else min(
        n - 1, max(1, int(np.ceil(NEIGHBOUR_FRACTION * n)))
    )
    metadata: dict[str, Any] = {
        "method_id": METHOD_ID,
        "implementation": "controlled_exact_python_reimplementation",
        "canonical_core": "Du-Ding-Jia DPC-KNN (2016)",
        "canonical_core_doi": "10.1016/j.knosys.2016.02.001",
        "center_selector": "Flores-Garza GB-DPC P1 last-gap rule",
        "center_selector_doi": "10.1016/j.knosys.2020.106350",
        "distance": "euclidean",
        "neighbour_fraction": NEIGHBOUR_FRACTION,
        "k": k,
        "k_rounding": "ceil(p*n)_clipped_to_[1,n-1]",
        "knn_search": "exact_blockwise_all_pairs",
        "delta_search": "exact_global_blockwise_all_pairs",
        "density_tie_convention": "rho_descending_then_original_row_index_ascending",
        "all_points_assigned": True,
        "noise_model": "none",
        "block_size": int(block_size),
        "n_centers": int(state.centers.size),
        "center_fallback_used": bool(state.center_fallback_used),
    }
    profile = {
        "n_centers": int(state.centers.size),
        "rho_min": float(np.min(state.rho)),
        "rho_max": float(np.max(state.rho)),
        "delta_min": float(np.min(state.delta)),
        "delta_max": float(np.max(state.delta)),
        "gb_mean_gap": state.gap_threshold,
    }
    return DPCkNNResult(state.labels, metadata, profile)


__all__ = [
    "DPCkNNResult",
    "METHOD_ID",
    "NEIGHBOUR_FRACTION",
    "run_dpc_knn_2016",
]
