from __future__ import annotations

import numpy as np
import pytest

from stratascan import HNSWConfig, build_knn_graph


pytest.importorskip("faiss")


def test_hnsw_preserves_nearest_core_neighbours() -> None:
    X = np.random.default_rng(42).normal(size=(2000, 16)).astype(np.float32)
    exact, _ = build_knn_graph(X, k=16, backend="faiss_flat", n_jobs=1)
    approximate, _ = build_knn_graph(X, k=16, backend="faiss_hnsw", n_jobs=1)
    approximate.validate(deep=True)
    recall4 = np.mean([
        len(set(exact.indices[row, :4]) & set(approximate.indices[row, :4])) / 4
        for row in range(X.shape[0])
    ])
    assert recall4 >= 0.99


def test_hnsw_rejects_search_width_below_k() -> None:
    X = np.random.default_rng(7).normal(size=(100, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="ef_construction/ef_search"):
        build_knn_graph(
            X,
            k=16,
            backend="faiss_hnsw",
            hnsw_config=HNSWConfig(ef_search=8),
        )
