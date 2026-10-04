from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .neighbors import Backend, HNSWConfig, build_knn_graph
from .types import KNNGraph


class StrataSCAN:
    """Shared graph construction and fitting methods for the article estimator."""

    def __init__(
        self,
        *,
        k: int = 32,
        backend: Backend = "auto",
        n_jobs: int = 1,
        ambient_dimension: float | None = None,
        hnsw_config: HNSWConfig | None = None,
    ) -> None:
        self.k = int(k)
        self.backend = backend
        self.n_jobs = int(n_jobs)
        self.ambient_dimension = ambient_dimension
        self.hnsw_config = hnsw_config

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
