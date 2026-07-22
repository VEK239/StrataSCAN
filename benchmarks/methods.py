from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baselines import dispatch_baseline
from stratascan import StrataSCAN


@dataclass(slots=True)
class MethodResult:
    labels: np.ndarray
    parameters: dict[str, Any]
    profile: dict[str, Any]


def geometry_profile(dimension: int) -> str:
    return "low_dim" if dimension <= 2 else "high_dim"


def run_method(method: str, X: np.ndarray, seed: int) -> MethodResult:
    profile = geometry_profile(int(X.shape[1]))
    backend = "kd_tree" if profile == "low_dim" else "faiss_hnsw"
    if method in {"StrataSCAN", "Gamma-StrictCore-kNN"}:
        model = StrataSCAN(backend=backend, n_jobs=1, ambient_dimension=float(X.shape[1]))
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "0.1.0",
                "geometry_profile": profile,
                "knn_backend": backend,
                "k": 32,
                "min_samples": 5,
                "dense_core_quantile": 0.95,
                "tail_probe_quantile": 0.02,
                "tail_probe_score": "d4/d32",
                "tail_probe_alpha": 0.05,
                "tail_core_quantile": 0.05,
                "border_multiplier": 1.25,
            },
            model.profile_,
        )

    result = dispatch_baseline(method, X, profile, seed=seed)
    return MethodResult(
        result.labels,
        {"geometry_profile": profile, "knn_backend": backend, **result.metadata},
        {},
    )
