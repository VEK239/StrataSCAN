from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stratascan.baselines import dispatch_baseline
from stratascan.baselines.dpc_knn import METHOD_ID as DPC_KNN_METHOD_ID, run_dpc_knn_2016
from stratascan import StrataSCAN
from stratascan.optimization import (
    OptimizationStrataSCAN,
    OptimizationStrictCoreConfig,
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
                "version": "0.2.3",
                "geometry_profile": profile,
                "knn_backend": backend,
                "k": 32,
                "shell_ranks": [4, 8, 16, 32],
                "stratification_model": (
                    "adaptive_gamma_density_basis_with_semantic_background"
                ),
                "component_selection": "semantic_icl_with_adaptive_bound",
                "initial_component_bound": 8,
                "hard_component_bound": 24,
                "component_initialization": (
                    "compiled_split_warm_single_start_with_guarded_confirmation"
                ),
                "gamma_tolerance": 1e-4,
                "ambiguous_model_confirmation_tolerance": 1e-5,
                "duplicate_profile_confirmation_minimum": 3,
                "duplicate_profile_confirmation_tolerance": 1e-8,
                "background_model": (
                    "adaptive_gamma_basis_mixture_with_largest_rate_gap"
                ),
                "assignment": "maximum_posterior_basis_then_semantic_role",
                "fit_samples": "all_rows",
                "extraction_objective": "gamma_topology_description_length",
                "core_thresholds": "observed_d4_event_sweep",
                "background_scan": (
                    "adaptive_dyadic_rank_windows_with_poisson_beta_boundary_code"
                ),
                "connectivity": "density_ordered_non_merging_knn_watershed",
                "border_rule": "strict_local_core_radius",
            },
            model.profile_,
        )
    if method == DPC_KNN_METHOD_ID:
        result = run_dpc_knn_2016(X)
        return MethodResult(
            result.labels,
            {
                "geometry_profile": profile,
                "knn_backend": "not_used_exact_blockwise_all_pairs",
                **result.metadata,
            },
            result.profile,
        )

    structural_ablations = {
        "StrataSCAN-NoDensityStratification": (
            OptimizationStrictCoreConfig(signal_strata="single_layer"),
            "density_stratification",
            "single_non_background_density_layer",
        ),
        "StrataSCAN-FixedCoreScale": (
            OptimizationStrictCoreConfig(core_threshold_rule="fixed_quantile"),
            "stratum_specific_event_sweep",
            "predeclared_fixed_q95_core_scale",
        ),
        "StrataSCAN-NoBackgroundRecovery": (
            OptimizationStrictCoreConfig(background_recovery=False),
            "adaptive_initial_background_recovery",
            "initial_background_remains_background",
        ),
    }
    if method in structural_ablations:
        optimization_config, ablated_component, replacement = structural_ablations[method]
        model = OptimizationStrataSCAN(
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
            optimization_config=optimization_config,
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "0.2.4-structural-ablation-v1",
                "geometry_profile": profile,
                "knn_backend": backend,
                "ablation_of": ablated_component,
                "replacement": replacement,
                "optimization_config": {
                    "signal_strata": optimization_config.signal_strata,
                    "core_threshold_rule": optimization_config.core_threshold_rule,
                    "fixed_core_quantile": optimization_config.fixed_core_quantile,
                    "background_recovery": optimization_config.background_recovery,
                    "core_rank": optimization_config.core_rank,
                    "connectivity_rank": optimization_config.connectivity_rank,
                },
            },
            model.profile_,
        )

    result = dispatch_baseline(method, X, profile, seed=seed)
    return MethodResult(
        result.labels,
        {"geometry_profile": profile, "knn_backend": backend, **result.metadata},
        {},
    )
