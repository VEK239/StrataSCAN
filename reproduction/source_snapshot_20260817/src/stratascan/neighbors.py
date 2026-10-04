from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

from .types import KNNGraph

Backend = Literal["auto", "kd_tree", "ball_tree", "brute", "faiss_flat", "faiss_hnsw"]


@dataclass(frozen=True, slots=True)
class HNSWConfig:
    m: int = 12
    ef_construction: int = 32
    ef_search: int = 32
    query_block_size: int = 50_000


def _remove_self(
    indices: np.ndarray,
    distances: np.ndarray,
    k: int,
    *,
    row_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    n = indices.shape[0]
    out_i = np.empty((n, k), dtype=np.int32)
    out_d = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, 50_000):
        stop = min(n, start + 50_000)
        block_i = indices[start:stop]
        block_d = distances[start:stop]
        row_ids = np.arange(row_offset + start, row_offset + stop)[:, None]
        mask = (block_i != row_ids) & (block_i >= 0)
        if np.any(np.sum(mask, axis=1) < k):
            raise RuntimeError("nearest-neighbour backend returned fewer than k valid rows")
        rank = np.cumsum(mask, axis=1) - 1
        valid = mask & (rank < k)
        rr, cc = np.nonzero(valid)
        out_i[start + rr, rank[rr, cc]] = block_i[rr, cc]
        out_d[start + rr, rank[rr, cc]] = block_d[rr, cc]
    return out_i, out_d


def _faiss_search_blocks(index, X: np.ndarray, k: int, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    out_i = np.empty((n, k), dtype=np.int32)
    out_d = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        squared, indices = index.search(X[start:stop], k + 1)
        block_i, block_d = _remove_self(
            indices.astype(np.int32, copy=False),
            np.sqrt(np.maximum(squared, 0.0), dtype=np.float32),
            k,
            row_offset=start,
        )
        out_i[start:stop] = block_i
        out_d[start:stop] = block_d
    return out_i, out_d


def _faiss_flat_knn(X: np.ndarray, k: int, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("backend='faiss_flat' requires faiss-cpu") from exc
    faiss.omp_set_num_threads(max(1, int(n_jobs)))
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    return _faiss_search_blocks(index, X, k, 50_000)


def _faiss_hnsw_knn(
    X: np.ndarray,
    k: int,
    n_jobs: int,
    config: HNSWConfig,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("backend='faiss_hnsw' requires faiss-cpu") from exc
    if config.m < 2 or config.ef_construction < k or config.ef_search < k:
        raise ValueError("HNSW requires m >= 2 and ef_construction/ef_search >= k")
    if config.query_block_size < 1:
        raise ValueError("HNSW query_block_size must be positive")
    faiss.omp_set_num_threads(max(1, int(n_jobs)))
    index = faiss.IndexHNSWFlat(X.shape[1], int(config.m))
    index.hnsw.efConstruction = int(config.ef_construction)
    index.hnsw.efSearch = int(config.ef_search)
    index.add(X)
    return _faiss_search_blocks(index, X, k, int(config.query_block_size))


def build_knn_graph(
    X: ArrayLike,
    *,
    k: int = 32,
    backend: Backend = "auto",
    n_jobs: int = 1,
    hnsw_config: HNSWConfig | None = None,
) -> tuple[KNNGraph, float]:
    """Build one reusable kNN graph.

    ``faiss_flat`` is exact Euclidean search. ``faiss_hnsw`` is approximate and
    uses a bounded-memory block query over an HNSW index. Returned rows are
    distance-sorted and self-free.
    """

    X_arr = np.asarray(X, dtype=np.float32, order="C")
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2-D array")
    if k < 1 or k >= X_arr.shape[0]:
        raise ValueError("k must satisfy 1 <= k < n_samples")

    algorithm: str = backend
    if backend == "auto":
        if X_arr.shape[1] <= 2:
            algorithm = "kd_tree"
        else:
            try:
                import faiss  # noqa: F401
                algorithm = "faiss_hnsw"
            except ImportError:
                algorithm = "brute"

    started = perf_counter()
    if algorithm == "faiss_flat":
        out_i, out_d = _faiss_flat_knn(X_arr, k, n_jobs)
    elif algorithm == "faiss_hnsw":
        out_i, out_d = _faiss_hnsw_knn(
            X_arr,
            k,
            n_jobs,
            hnsw_config or HNSWConfig(),
        )
    else:
        nn = NearestNeighbors(
            n_neighbors=k + 1,
            algorithm=algorithm,
            metric="euclidean",
            n_jobs=n_jobs,
        )
        nn.fit(X_arr)
        distances, indices = nn.kneighbors(X_arr, return_distance=True)
        out_i, out_d = _remove_self(indices, distances, k)
    elapsed = perf_counter() - started
    return KNNGraph(out_i, out_d, sorted_unique=True), elapsed
