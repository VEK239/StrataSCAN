from __future__ import annotations

"""Controlled baseline implementations used by the unified benchmark.

The goal of this module is reproducibility under one runner, not to claim that
these serial Python ports match the system-level performance of every authors'
production implementation. Library implementations are used for DBSCAN,
HDBSCAN, and OPTICS. kNN-DBSCAN follows the corrected paper semantics from the
v0.3.2 benchmark. SNN-DBSCAN and VDBSCAN-2007 are controlled reimplementations.
AMD-DBSCAN is an official-compatible, deliberately dense compatibility path.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS

from stratascan.neighbors import build_knn_graph
from stratascan.types import KNNGraph


@dataclass(slots=True)
class BaselineResult:
    labels: np.ndarray
    metadata: dict[str, Any]


def _build_graph(X: np.ndarray, *, k: int, profile: str) -> KNNGraph:
    backend = "kd_tree" if profile == "low_dim" else "faiss_hnsw"
    graph, _ = build_knn_graph(X, k=k, backend=backend, n_jobs=1)
    return graph


def _polynomial_knee(values: np.ndarray, *, degree: int = 7) -> float:
    """Stable polynomial knee for a monotone increasing k-distance curve."""
    y = np.sort(np.asarray(values, dtype=np.float64))
    n = y.size
    if n < 8:
        return float(np.quantile(y, 0.9))
    x = np.linspace(0.0, 1.0, n)
    degree = min(degree, max(2, n - 1))
    coeff = np.polyfit(x, y, degree)
    smooth = np.polyval(coeff, x)
    # Distance to the chord between endpoints. For a convex increasing curve,
    # the largest downward chord deviation is the elbow before the sparse tail.
    chord = smooth[0] + x * (smooth[-1] - smooth[0])
    idx = int(np.argmax(chord - smooth))
    idx = max(0, min(n - 1, idx))
    return float(y[idx])


def _eps_from_rank_quantile(
    X: np.ndarray, *, rank: int, quantile: float, profile: str
) -> tuple[float, KNNGraph]:
    graph = _build_graph(X, k=rank, profile=profile)
    eps = float(np.quantile(graph.distances[:, rank - 1], quantile))
    return eps, graph


def run_dbscan(X: np.ndarray, profile: str) -> BaselineResult:
    if profile == "low_dim":
        graph = _build_graph(X, k=4, profile=profile)
        base_eps = _polynomial_knee(graph.distances[:, 3])
        eps = 0.7 * base_eps
        params = {"min_samples": 5, "eps_multiplier": 0.7, "base_eps": base_eps, "eps": eps}
    else:
        eps, _ = _eps_from_rank_quantile(X, rank=4, quantile=0.05, profile=profile)
        params = {"min_samples": 5, "eps_quantile": 0.05, "eps": eps}
    labels = DBSCAN(eps=eps, min_samples=5, n_jobs=1).fit_predict(X).astype(np.int64, copy=False)
    return BaselineResult(labels, params)


def run_hdbscan(X: np.ndarray, profile: str) -> BaselineResult:
    min_cluster_size = 40 if profile == "low_dim" else 8
    model = HDBSCAN(
        min_samples=5,
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        algorithm="auto",
        n_jobs=1,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    )
    labels = model.fit_predict(X).astype(np.int64, copy=False)
    return BaselineResult(labels, {"min_samples": 5, "min_cluster_size": min_cluster_size})


def run_optics(X: np.ndarray, profile: str) -> BaselineResult:
    if profile == "low_dim":
        min_cluster_size, xi, q = 40, 0.1, 0.98
    else:
        min_cluster_size, xi, q = 8, 0.03, 0.1
    eps, _ = _eps_from_rank_quantile(X, rank=4, quantile=q, profile=profile)
    model = OPTICS(
        min_samples=5,
        min_cluster_size=min_cluster_size,
        xi=xi,
        max_eps=eps,
        metric="euclidean",
        algorithm="auto",
        n_jobs=1,
    )
    labels = model.fit_predict(X).astype(np.int64, copy=False)
    return BaselineResult(
        labels,
        {"min_samples": 5, "min_cluster_size": min_cluster_size, "xi": xi, "max_eps_quantile": q, "max_eps": eps},
    )


def _connected_labels_from_edges(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    core: np.ndarray,
    *,
    min_cluster_size: int = 1,
) -> np.ndarray:
    core_idx = np.flatnonzero(core).astype(np.int32)
    labels = np.full(n, -1, dtype=np.int64)
    if core_idx.size == 0:
        return labels
    local = np.full(n, -1, dtype=np.int32)
    local[core_idx] = np.arange(core_idx.size, dtype=np.int32)
    keep = core[rows] & core[cols]
    lr = local[rows[keep]]
    lc = local[cols[keep]]
    data = np.ones(lr.size, dtype=np.uint8)
    adj = coo_matrix((data, (lr, lc)), shape=(core_idx.size, core_idx.size)).tocsr()
    _, component = connected_components(adj, directed=False, return_labels=True)
    sizes = np.bincount(component)
    valid = sizes >= min_cluster_size
    remap = np.full(sizes.size, -1, dtype=np.int64)
    remap[np.flatnonzero(valid)] = np.arange(np.sum(valid), dtype=np.int64)
    labels[core_idx] = remap[component]
    return labels


def knn_dbscan_from_graph(graph: KNNGraph, *, eps: float, min_samples: int = 5) -> np.ndarray:
    """Corrected Chen--Ruys--Biros semantics using exactly M-1 non-self neighbours."""
    m = min_samples - 1
    if graph.k < m:
        raise ValueError("graph too narrow for min_samples")
    neigh = graph.indices[:, :m]
    dist = graph.distances[:, :m]
    core = dist[:, -1] <= eps
    rows = np.repeat(np.arange(graph.n_samples, dtype=np.int32), m)
    cols = neigh.reshape(-1)
    labels = _connected_labels_from_edges(graph.n_samples, rows, cols, core)
    border = np.flatnonzero(~core)
    if border.size:
        b_neigh = neigh[border]
        b_dist = dist[border]
        valid = core[b_neigh] & (b_dist <= eps) & (labels[b_neigh] >= 0)
        has = np.any(valid, axis=1)
        first = np.argmax(valid, axis=1)
        chosen = b_neigh[np.arange(border.size), first]
        labels[border[has]] = labels[chosen[has]]
    return labels


def run_knn_dbscan(X: np.ndarray, profile: str) -> BaselineResult:
    q = 0.25 if profile == "low_dim" else 0.05
    graph = _build_graph(X, k=4, profile=profile)
    eps = float(np.quantile(graph.distances[:, 3], q))
    labels = knn_dbscan_from_graph(graph, eps=eps, min_samples=5)
    return BaselineResult(labels, {"k": 4, "min_samples": 5, "eps_quantile": q, "eps": eps})


def warm_numba_kernels() -> None:
    from baselines.snn_kernel import warm
    warm()


def _shared_counts_for_edges(sorted_neigh: np.ndarray, edge_neigh: np.ndarray) -> np.ndarray:
    from baselines.snn_kernel import shared_counts_numba
    return shared_counts_numba(sorted_neigh, edge_neigh)

def run_snn_dbscan(X: np.ndarray, profile: str) -> BaselineResult:
    k, eps_snn, min_samples = 20, 13, 5
    graph = _build_graph(X, k=k, profile=profile)
    sorted_neigh = np.sort(graph.indices.astype(np.int32, copy=False), axis=1)
    shared = _shared_counts_for_edges(sorted_neigh, graph.indices)
    valid = shared >= eps_snn
    counts = np.sum(valid, axis=1)
    core = counts >= min_samples
    rows = np.repeat(np.arange(graph.n_samples, dtype=np.int32), k)[valid.reshape(-1)]
    cols = graph.indices.reshape(-1)[valid.reshape(-1)]
    labels = _connected_labels_from_edges(graph.n_samples, rows, cols, core)
    border = np.flatnonzero(~core)
    if border.size:
        bn = graph.indices[border]
        bv = valid[border]
        eligible = bv & core[bn] & (labels[bn] >= 0)
        has = np.any(eligible, axis=1)
        first = np.argmax(eligible, axis=1)
        chosen = bn[np.arange(border.size), first]
        labels[border[has]] = labels[chosen[has]]
    return BaselineResult(labels, {"k": k, "eps": eps_snn, "min_samples": min_samples})


def _vdbscan_eps_values(kdist: np.ndarray, *, max_values: int, prominence: float) -> list[float]:
    y = np.sort(np.asarray(kdist, dtype=np.float64))
    n = y.size
    if n < 16:
        return [float(np.quantile(y, 0.9))]
    # Smooth the monotone curve, then identify changes in its slope. The
    # prominence is expressed on a [0, 1] normalized derivative, matching the
    # controlled v0.3.x reimplementation parameterization.
    window = min(n // 2 * 2 - 1, 501)
    window = max(9, window)
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window % 2 == 0:
        window -= 1
    smooth = savgol_filter(y, window_length=window, polyorder=min(3, window - 2), mode="interp")
    slope = np.gradient(smooth)
    lo, hi = float(np.min(slope)), float(np.max(slope))
    normalized = (slope - lo) / max(hi - lo, 1e-12)
    peaks, props = find_peaks(normalized, prominence=prominence, distance=max(4, n // 100))
    candidates = peaks.tolist()
    # Always include the global polynomial knee as a robust primary scale.
    knee = _polynomial_knee(y)
    knee_idx = int(np.searchsorted(y, knee, side="left"))
    candidates.append(knee_idx)
    candidates = sorted(set(max(0, min(n - 1, int(i))) for i in candidates))
    # Prefer separated scales with highest slope prominence, while preserving
    # ascending epsilon order for the sequential VDBSCAN extraction.
    vals = sorted(set(float(y[i]) for i in candidates))
    if len(vals) > max_values:
        # Preserve endpoints and use approximately even quantiles among detected knees.
        idx = np.unique(np.linspace(0, len(vals) - 1, max_values).round().astype(int))
        vals = [vals[int(i)] for i in idx]
    return vals or [float(np.quantile(y, 0.9))]


def _sequential_dbscan(X: np.ndarray, eps_values: list[float], *, min_samples: int) -> np.ndarray:
    labels = np.full(X.shape[0], -1, dtype=np.int64)
    next_label = 0
    remaining = np.arange(X.shape[0], dtype=np.int64)
    for eps in sorted(eps_values):
        if remaining.size < min_samples:
            break
        local = DBSCAN(eps=float(eps), min_samples=min_samples, n_jobs=1).fit_predict(X[remaining])
        for c in np.unique(local[local >= 0]):
            mask = local == c
            labels[remaining[mask]] = next_label
            next_label += 1
        remaining = remaining[local < 0]
    return labels


def run_vdbscan_2007(X: np.ndarray, profile: str) -> BaselineResult:
    max_values = 3 if profile == "low_dim" else 7
    prominence = 0.03 if profile == "low_dim" else 0.08
    graph = _build_graph(X, k=4, profile=profile)
    eps_values = _vdbscan_eps_values(graph.distances[:, 3], max_values=max_values, prominence=prominence)
    labels = _sequential_dbscan(X, eps_values, min_samples=5)
    return BaselineResult(
        labels,
        {"min_samples": 5, "max_eps_values": max_values, "knee_prominence": prominence, "eps_values": eps_values},
    )


def run_amd_dbscan(X: np.ndarray, profile: str) -> BaselineResult:
    """Dense official-compatible AMD path retained to expose its O(n^2) frontier.

    The public reference procedure uses a dense pairwise-distance stage. This
    compatibility port deliberately preserves that memory behavior and derives
    three density levels before sequential DBSCAN extraction.
    """
    from scipy.spatial.distance import cdist

    n, d = X.shape
    distances = cdist(X, X, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    if d > 8:
        adaptive_rank = max(5, n - 10)
    else:
        adaptive_rank = max(5, int(round(np.sqrt(n) / 2.0)))
    adaptive_rank = min(adaptive_rank, n - 1)
    kth = np.partition(distances, adaptive_rank - 1, axis=1)[:, adaptive_rank - 1]
    if d > 8:
        # In the official-compatible high-dimensional path the adaptive rank
        # approaches n, yielding broad scales and often one giant component.
        qs = (0.25, 0.65, 0.90)
    else:
        qs = (0.08, 0.55, 0.85)
    eps_values = [float(np.quantile(kth, q)) for q in qs]
    # Release the dense matrix before DBSCAN, while peak RSS still records it.
    del distances
    labels = _sequential_dbscan(X, eps_values, min_samples=5)
    return BaselineResult(
        labels,
        {"n_density_levels": 3, "balance_rounds": 3, "adaptive_rank": adaptive_rank, "eps_values": eps_values},
    )


def run_knn_leiden(X: np.ndarray, profile: str, *, seed: int = 42) -> BaselineResult:
    import igraph as ig
    import leidenalg as la

    k, edge_rank, resolution = 20, 15, 0.5
    graph = _build_graph(X, k=k, profile=profile)
    n = graph.n_samples
    rows = np.repeat(np.arange(n, dtype=np.int32), edge_rank)
    cols = graph.indices[:, :edge_rank].reshape(-1)
    dists = graph.distances[:, :edge_rank].reshape(-1)
    lo = np.minimum(rows, cols).astype(np.int64)
    hi = np.maximum(rows, cols).astype(np.int64)
    key = lo * np.int64(n) + hi
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    first = np.r_[True, key_sorted[1:] != key_sorted[:-1]]
    chosen = order[first]
    edges = list(zip(lo[chosen].tolist(), hi[chosen].tolist(), strict=True))
    weights = (1.0 / (1.0 + dists[chosen])).astype(float).tolist()
    g = ig.Graph(n=n, edges=edges, directed=False)
    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed,
    )
    labels = np.asarray(partition.membership, dtype=np.int64)
    return BaselineResult(
        labels,
        {"k": k, "edge_rank": edge_rank, "resolution": resolution, "symmetrization": "union", "weights": "1/(1+d)"},
    )


def dispatch_baseline(method: str, X: np.ndarray, profile: str, *, seed: int = 42) -> BaselineResult:
    if profile not in {"low_dim", "high_dim"}:
        raise ValueError("baseline profile must be 'low_dim' or 'high_dim'")
    if method == "DBSCAN":
        return run_dbscan(X, profile)
    if method == "HDBSCAN":
        return run_hdbscan(X, profile)
    if method == "OPTICS":
        return run_optics(X, profile)
    if method == "SNN-DBSCAN":
        return run_snn_dbscan(X, profile)
    if method == "VDBSCAN-2007":
        return run_vdbscan_2007(X, profile)
    if method == "AMD-DBSCAN":
        return run_amd_dbscan(X, profile)
    if method == "kNN-DBSCAN":
        return run_knn_dbscan(X, profile)
    if method == "kNN+Leiden":
        return run_knn_leiden(X, profile, seed=seed)
    raise ValueError(f"unknown baseline method: {method}")
