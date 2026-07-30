from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from baselines import dispatch_baseline
from stratascan import StrataSCAN
from stratascan.optimization import GammaMDLConfig, OptimizationStrataSCAN
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
                "version": "0.2.2",
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
                    "split_warm_plus_independent_restarts"
                ),
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

    # These are deliberately narrow, named component ablations for the locked
    # synthetic evidence protocol.  They use the released extraction stage and
    # alter exactly one part of the Gamma stratification decision, so their
    # outcomes cannot be mistaken for tuned alternatives to the public default.
    ablations = {
        "StrataSCAN-FixedComponentBound": (
            replace(
                GammaMDLConfig(), adaptive_components=False, max_components=8,
                hard_max_components=8
            ),
            "adaptive_component_bound",
            "fixed_candidate_range_1_to_8",
        ),
        "StrataSCAN-ContinuousBackground": (
            replace(GammaMDLConfig(), background_distribution="gamma_rate_mixture"),
            "discrete_gamma_background_basis",
            "continuous_gamma_rate_background",
        ),
        "StrataSCAN-RateChangepoint": (
            replace(GammaMDLConfig(), component_roles="rate_changepoint_mdl"),
            "largest_rate_gap_semantic_role",
            "rate_changepoint_mdl_role",
        ),
    }
    if method in ablations:
        gamma_config, ablated_component, replacement = ablations[method]
        model = OptimizationStrataSCAN(
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
            gamma_config=gamma_config,
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "0.2.2-component-ablation-v1",
                "geometry_profile": profile,
                "knn_backend": backend,
                "ablation_of": ablated_component,
                "replacement": replacement,
                "gamma_config": {
                    "adaptive_components": gamma_config.adaptive_components,
                    "max_components": gamma_config.max_components,
                    "hard_max_components": gamma_config.hard_max_components,
                    "background_distribution": gamma_config.background_distribution,
                    "component_roles": gamma_config.component_roles,
                },
            },
            model.profile_,
        )

    if method in {"StrataSCAN-PredictiveMultiscale", "StrataSCAN-0.1.2"}:
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

    if method == "StrataSCAN-Optimization":
        # Reproduce the algorithm snapshot evaluated in the OEDM manuscript.
        # The public 0.2.2 estimator may explore an adaptive component range;
        # the frozen dev10 experiments searched exactly m=1,...,8.
        model = OptimizationStrataSCAN(
            backend=backend,
            n_jobs=1,
            ambient_dimension=float(X.shape[1]),
            gamma_config=GammaMDLConfig(
                max_components=8,
                hard_max_components=8,
                adaptive_components=False,
                entropy_scope="basis",
            ),
        )
        labels = model.fit_predict(X)
        return MethodResult(
            labels,
            {
                "version": "0.2.0-dev10-oedm2026-evaluated",
                "geometry_profile": profile,
                "knn_backend": backend,
                "k": 32,
                "shell_ranks": [4, 8, 16, 32],
                "stratification_model": "gamma_density_basis_with_semantic_background",
                "component_selection": "basis_icl_fixed_candidate_range",
                "component_range": [1, 8],
                "component_initialization": "split_warm_plus_independent_restarts",
                "background_model": "gamma_basis_mixture_with_largest_rate_gap",
                "assignment": "maximum_posterior_basis_then_semantic_role",
                "fit_samples": "all_rows",
                "extraction_objective": "gamma_topology_description_length",
                "core_thresholds": "observed_d4_event_sweep",
                "background_scan": "adaptive_dyadic_rank_windows_with_poisson_beta_boundary_code",
                "connectivity": "density_ordered_non_merging_knn_watershed",
                "border_rule": "strict_local_core_radius",
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
