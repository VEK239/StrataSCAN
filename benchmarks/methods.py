from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baselines import dispatch_baseline
from stratascan import StrataSCAN
from stratascan.predictive import PredictiveMultiscaleStrataSCAN
from stratascan.experimental import (
    AdaptiveMultiscaleTailStrataSCAN,
    CalibratedDBCVTailStrataSCAN,
    ComponentMultiscaleTailStrataSCAN,
    GammaLLRBootstrapTailStrataSCAN,
    PercolationTailStrataSCAN,
    PersistentContrastTailStrataSCAN,
    PersistentDBCVTailStrataSCAN,
    StableDBCVTailStrataSCAN,
    RankedTailStrataSCAN,
    ResidualStrataSCAN,
)


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
                "version": "0.1.2",
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

    if method == "StrataSCAN-PredictiveMultiscale":
        model = PredictiveMultiscaleStrataSCAN(
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "0.1.2",
                "geometry_profile": profile,
                "knn_backend": backend,
                "rate_profile": "log_lambda_sj=mu_s+b_s_log_kj",
                "component_selection": "repeated_holdout_one_standard_error",
                "validation_repeats": 3,
                "assignment_stability_threshold": 0.80,
            },
            model.profile_,
        )

    if method == "StrataSCAN-WideGraph":
        hnsw = None
        if backend == "faiss_hnsw":
            from stratascan.neighbors import HNSWConfig

            hnsw = HNSWConfig(m=12, ef_construction=256, ef_search=256)
        model = StrataSCAN(
            k=256,
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
            hnsw_config=hnsw,
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "wide-graph-control-v1",
                "geometry_profile": profile,
                "knn_backend": backend,
                "k": 256,
                "tail_probe_outer_rank": 32,
            },
            model.profile_,
        )

    if method == "StrataSCAN-RankedTail":
        model = RankedTailStrataSCAN(
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "experimental-ranked-tail-v1",
                "geometry_profile": profile,
                "knn_backend": backend,
                "tail_probe_method": "ranked_shell_mhg",
            },
            model.profile_,
        )

    if method == "StrataSCAN-PercolationTail":
        model = PercolationTailStrataSCAN(
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "experimental-percolation-tail-v1",
                "geometry_profile": profile,
                "knn_backend": backend,
                "tail_probe_method": "stratified_component_percolation",
            },
            model.profile_,
        )

    if method == "StrataSCAN-ComponentMultiscale":
        model = ComponentMultiscaleTailStrataSCAN(
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
            tail_multiscale_random_state=seed,
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "experimental-component-multiscale-v1",
                "geometry_profile": profile,
                "knn_backend": backend,
                "tail_probe_method": "component_multiscale",
                "neighbour_ranks": [4, 8, 16, 32],
                "permutations": 199,
            },
            model.profile_,
        )

    if method == "StrataSCAN-AdaptiveMultiscale":
        hnsw = None
        if backend == "faiss_hnsw":
            from stratascan.neighbors import HNSWConfig

            hnsw = HNSWConfig(m=12, ef_construction=256, ef_search=256)
        model = AdaptiveMultiscaleTailStrataSCAN(
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
            hnsw_config=hnsw,
            tail_multiscale_random_state=seed,
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "experimental-adaptive-multiscale-v1",
                "geometry_profile": profile,
                "knn_backend": backend,
                "tail_probe_method": "component_adaptive_multiscale",
                "neighbour_ranks": [4, 8, 16, 32, 64, 128, 256],
                "breakpoint_rule": "maximum_holdout_connectivity_excess_before_dilution",
                "component_null_size_bins": "floor_log2",
                "component_multiple_testing": "benjamini_hochberg",
                "permutations": 199,
            },
            model.profile_,
        )
    experimental = {
        "StrataSCAN-PersistentContrast": PersistentContrastTailStrataSCAN,
        "StrataSCAN-PersistentDBCV": PersistentDBCVTailStrataSCAN,
        "StrataSCAN-CalibratedDBCV": CalibratedDBCVTailStrataSCAN,
        "StrataSCAN-StableDBCV": StableDBCVTailStrataSCAN,
        "StrataSCAN-GammaLLRBootstrap": GammaLLRBootstrapTailStrataSCAN,
    }
    if method in experimental:
        kwargs = (
            {"random_state": seed}
            if method in {
                "StrataSCAN-GammaLLRBootstrap",
                "StrataSCAN-CalibratedDBCV",
                "StrataSCAN-StableDBCV",
            }
            else {}
        )
        model = experimental[method](
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
            **kwargs,
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "experimental-persistent-tail-v1",
                "geometry_profile": profile,
                "knn_backend": backend,
                "tail_probe_method": method.removeprefix("StrataSCAN-").lower(),
                "bootstrap_replicates": (
                    39
                    if method in {"StrataSCAN-GammaLLRBootstrap", "StrataSCAN-CalibratedDBCV"}
                    else 0
                ),
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
