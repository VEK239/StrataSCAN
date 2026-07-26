from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baselines import dispatch_baseline
from stratascan import StrataSCAN
from stratascan.experimental import ResidualStrataSCAN


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
    if method == "StrataSCAN":
        model = StrataSCAN(backend=backend, n_jobs=1, ambient_dimension=float(X.shape[1]))
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "0.1.1",
                "geometry_profile": profile,
                "knn_backend": backend,
                "k": 32,
                "min_samples": 5,
                "density_features": ["log_d4", "log_d4/d8", "log_d4/d16", "log_d4/d32"],
                "stratification_model": "four_shell_gamma_mixture",
                "gamma_shell_shapes": [4, 4, 8, 16],
                "max_components": 8,
                "max_fit_samples": 10_000,
                "dense_core_quantile": 0.95,
                "tail_probe_quantile": 0.02,
                "tail_probe_score": "d4/d32",
                "tail_probe_alpha": 0.05,
                "tail_core_quantile": 0.05,
                "border_multiplier": 1.25,
            },
            model.profile_,
        )

    if method == "StrataSCAN-Residual":
        model = ResidualStrataSCAN(backend=backend, n_jobs=1)
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "experimental-v2",
                "geometry_profile": profile,
                "knn_backend": backend,
                "k": 32,
                "intrinsic_dimension": True,
                "retain_intermediate_components": True,
                "background_components": 1,
                "max_passes": 2,
                "accepted_cluster_size": 20,
                "residual_max_cluster_fraction": 0.20,
                "projection_components": None,
            },
            model.profile_,
        )

    result = dispatch_baseline(method, X, profile, seed=seed)
    return MethodResult(
        result.labels,
        {"geometry_profile": profile, "knn_backend": backend, **result.metadata},
        {},
    )
