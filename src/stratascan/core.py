from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .neighbors import Backend, HNSWConfig, build_knn_graph
from .types import KNNGraph
from .uniform_tail import UniformTailConfig


def stratified_knn_dbscan_from_graph(
    graph: KNNGraph,
    point_eps: ArrayLike,
    *,
    min_samples: int = 5,
    min_cluster_size: int = 5,
    border_multiplier: float = 1.25,
    core_seed_mask: ArrayLike | None = None,
    validate_graph: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Frozen StrictCore connectivity over a prebuilt kNN graph."""

    timings: dict[str, float] = {}
    started = perf_counter()
    if validate_graph:
        graph.validate(deep=False)
    timings["graph_validation"] = perf_counter() - started
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1")
    if not np.isfinite(border_multiplier) or border_multiplier <= 0.0:
        raise ValueError("border_multiplier must be positive and finite")
    rank = min_samples - 1
    if graph.k < rank:
        raise ValueError("graph too narrow for min_samples")

    eps = np.asarray(point_eps, dtype=np.float32)
    if eps.shape != (graph.n_samples,):
        raise ValueError("point_eps must contain one value per graph row")
    if not np.all(np.isfinite(eps)) or np.any(eps < 0.0):
        raise ValueError("point_eps must be finite and nonnegative")
    active = eps > 0.0
    seed_mask = None
    if core_seed_mask is not None:
        seed_mask = np.asarray(core_seed_mask, dtype=bool)
        if seed_mask.shape != (graph.n_samples,):
            raise ValueError("core_seed_mask must align with the graph")

    neighbours = graph.indices[:, :rank]
    distances = graph.distances[:, :rank]
    started = perf_counter()
    core = active & (distances[:, -1] <= eps)
    if seed_mask is not None:
        core &= seed_mask
    core_indices = np.flatnonzero(core).astype(np.int32)
    labels = np.full(graph.n_samples, -1, dtype=np.int64)
    timings["core_detection"] = perf_counter() - started

    started = perf_counter()
    if core_indices.size:
        rows = np.repeat(core_indices, rank)
        columns = neighbours[core_indices].reshape(-1)
        keep = core[columns]
        local = np.full(graph.n_samples, -1, dtype=np.int32)
        local[core_indices] = np.arange(core_indices.size, dtype=np.int32)
        local_rows = local[rows[keep]]
        local_columns = local[columns[keep]]
        adjacency = coo_matrix(
            (
                np.ones(local_rows.size, dtype=np.uint8),
                (local_rows, local_columns),
            ),
            shape=(core_indices.size, core_indices.size),
        ).tocsr()
        _, components = connected_components(adjacency, directed=False, return_labels=True)
        sizes = np.bincount(components)
        valid = sizes >= min_cluster_size
        remap = np.full(sizes.size, -1, dtype=np.int64)
        remap[np.flatnonzero(valid)] = np.arange(np.sum(valid), dtype=np.int64)
        labels[core_indices] = remap[components]
    timings["connected_components"] = perf_counter() - started

    started = perf_counter()
    border = np.flatnonzero(active & ~core).astype(np.int32)
    if border.size:
        border_neighbours = neighbours[border]
        eligible = (
            core[border_neighbours]
            & (labels[border_neighbours] >= 0)
            & (
                distances[border]
                <= eps[border_neighbours] * float(border_multiplier)
            )
        )
        has_match = np.any(eligible, axis=1)
        first = np.argmax(eligible, axis=1)
        chosen = border_neighbours[np.arange(border.size), first]
        labels[border[has_match]] = labels[chosen[has_match]]
    timings["border_assignment"] = perf_counter() - started
    timings.update(
        active_fraction=float(np.mean(active)),
        core_fraction=float(np.mean(core)),
        knn_dbscan_rank=float(rank),
        border_uses_core_epsilon=1.0,
        border_multiplier=float(border_multiplier),
        restricted_core_seeds=float(seed_mask is not None),
    )
    return labels, core, timings


class StrataSCAN:
    """StrataSCAN 0.1.0: the frozen Gamma-StrictCore clustering estimator."""

    def __init__(
        self,
        *,
        k: int = 32,
        backend: Backend = "auto",
        n_jobs: int = 1,
        ambient_dimension: float | None = None,
        hnsw_config: HNSWConfig | None = None,
        min_samples: int = 5,
        min_cluster_size: int = 5,
        dense_core_quantile: float = 0.95,
        tail_probe_quantile: float = 0.02,
        tail_probe_outer_rank: int = 32,
        tail_probe_alpha: float = 0.05,
        tail_core_quantile: float = 0.05,
        border_multiplier: float = 1.25,
        uniform_tail_config: UniformTailConfig | None = None,
    ) -> None:
        self.k = int(k)
        self.backend = backend
        self.n_jobs = int(n_jobs)
        self.ambient_dimension = ambient_dimension
        self.hnsw_config = hnsw_config
        self.min_samples = int(min_samples)
        self.min_cluster_size = int(min_cluster_size)
        self.dense_core_quantile = float(dense_core_quantile)
        self.tail_probe_quantile = float(tail_probe_quantile)
        self.tail_probe_outer_rank = int(tail_probe_outer_rank)
        self.tail_probe_alpha = float(tail_probe_alpha)
        self.tail_core_quantile = float(tail_core_quantile)
        self.border_multiplier = float(border_multiplier)
        self.uniform_tail_config = uniform_tail_config or UniformTailConfig()

    def _config(self):
        from .strict_core import GammaStrictCoreConfig

        return GammaStrictCoreConfig(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            dense_core_quantile=self.dense_core_quantile,
            tail_probe_quantile=self.tail_probe_quantile,
            tail_probe_outer_rank=self.tail_probe_outer_rank,
            tail_probe_alpha=self.tail_probe_alpha,
            tail_core_quantile=self.tail_core_quantile,
            border_multiplier=self.border_multiplier,
            uniform_tail=self.uniform_tail_config,
        )

    def fit_from_graph(
        self,
        graph: KNNGraph,
        *,
        ambient_dimension: float | None = None,
    ) -> "StrataSCAN":
        from .strict_core import gamma_strict_core_from_graph

        dimension = self.ambient_dimension if ambient_dimension is None else ambient_dimension
        result = gamma_strict_core_from_graph(
            graph,
            ambient_dimension=dimension,
            config=self._config(),
        )
        self.labels_ = result.labels
        self.core_sample_indices_ = np.flatnonzero(result.core_mask)
        self.n_clusters_ = int(np.unique(result.labels[result.labels >= 0]).size)
        self.profile_ = dict(result.profile)
        self.graph_ = graph
        return self

    def fit_predict_from_graph(
        self,
        graph: KNNGraph,
        *,
        ambient_dimension: float | None = None,
    ) -> np.ndarray:
        return self.fit_from_graph(graph, ambient_dimension=ambient_dimension).labels_

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "StrataSCAN":
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

    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        return self.fit(X, y).labels_
