from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA

from .core import StrataSCAN
from .neighbors import Backend, HNSWConfig, build_knn_graph
from .uniform_tail import UniformTailConfig


class RankedTailStrataSCAN(StrataSCAN):
    """Experimental cutoff-free tail enrichment variant.

    The released estimator is unchanged.  This variant replaces the fixed 2%
    rank window with a hypergeometric scan over all useful ranked cutoffs and
    validates candidate components on the first neighbour shell not used to
    construct their 4-NN graph.
    """

    def _config(self):
        return replace(super()._config(), tail_probe_method="ranked_shell_mhg")


class PercolationTailStrataSCAN(StrataSCAN):
    """Experimental tail test based on excess 4-NN component connectivity."""

    def __init__(
        self,
        *args,
        tail_percolation_permutations: int = 199,
        tail_percolation_density_bins: int = 4,
        tail_percolation_degree_bins: int = 4,
        tail_percolation_random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tail_percolation_permutations = int(tail_percolation_permutations)
        self.tail_percolation_density_bins = int(tail_percolation_density_bins)
        self.tail_percolation_degree_bins = int(tail_percolation_degree_bins)
        self.tail_percolation_random_state = int(tail_percolation_random_state)

    def _config(self):
        return replace(
            super()._config(),
            tail_probe_method="component_percolation",
            tail_percolation_permutations=self.tail_percolation_permutations,
            tail_percolation_density_bins=self.tail_percolation_density_bins,
            tail_percolation_degree_bins=self.tail_percolation_degree_bins,
            tail_percolation_random_state=self.tail_percolation_random_state,
        )


class ComponentMultiscaleTailStrataSCAN(StrataSCAN):
    """Experimental tail test using component-level multiscale density persistence."""

    def __init__(
        self,
        *args,
        tail_multiscale_permutations: int = 199,
        tail_multiscale_density_bins: int = 4,
        tail_multiscale_degree_bins: int = 4,
        tail_multiscale_random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tail_multiscale_permutations = int(tail_multiscale_permutations)
        self.tail_multiscale_density_bins = int(tail_multiscale_density_bins)
        self.tail_multiscale_degree_bins = int(tail_multiscale_degree_bins)
        self.tail_multiscale_random_state = int(tail_multiscale_random_state)

    def _config(self):
        return replace(
            super()._config(),
            tail_probe_method="component_multiscale",
            tail_percolation_permutations=self.tail_multiscale_permutations,
            tail_percolation_density_bins=self.tail_multiscale_density_bins,
            tail_percolation_degree_bins=self.tail_multiscale_degree_bins,
            tail_percolation_random_state=self.tail_multiscale_random_state,
        )


class AdaptiveMultiscaleTailStrataSCAN(ComponentMultiscaleTailStrataSCAN):
    """Experimental tail test that chooses its outer rank at a profile breakpoint."""

    def __init__(self, *args, k: int = 256, **kwargs) -> None:
        kwargs.setdefault("tail_multiple_components", True)
        super().__init__(*args, k=k, **kwargs)

    def _config(self):
        return replace(
            super()._config(),
            tail_probe_method="component_adaptive_multiscale",
            tail_probe_outer_rank=self.k,
        )


class PersistentContrastTailStrataSCAN(StrataSCAN):
    """Experimental tail selection by density persistence and boundary contrast."""

    def _config(self):
        return replace(super()._config(), tail_probe_method="persistent_contrast")


class PersistentDBCVTailStrataSCAN(StrataSCAN):
    """Experimental tail selection by density persistence and DBCV-like separation."""

    def _config(self):
        return replace(super()._config(), tail_probe_method="persistent_dbcv")


class CalibratedDBCVTailStrataSCAN(StrataSCAN):
    """Persistent DBCV selector calibrated over the complete fixed-graph scan."""

    def __init__(self, *args, bootstrap_replicates: int = 39, random_state: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrap_replicates = int(bootstrap_replicates)
        self.random_state = int(random_state)

    def _config(self):
        return replace(
            super()._config(),
            tail_probe_method="persistent_dbcv_bootstrap",
            tail_llr_bootstrap_replicates=self.bootstrap_replicates,
            tail_llr_random_state=self.random_state,
        )


class StableDBCVTailStrataSCAN(PersistentDBCVTailStrataSCAN):
    """Accept a DBCV tail only when its new cluster recurs after graph rebuilds."""

    def __init__(
        self,
        *args,
        stability_subsamples: int = 2,
        stability_fraction: float = 0.75,
        stability_jaccard: float = 0.50,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stability_subsamples = int(stability_subsamples)
        self.stability_fraction = float(stability_fraction)
        self.stability_jaccard = float(stability_jaccard)
        self.random_state = int(random_state)

    def _model_kwargs(self) -> dict[str, object]:
        return {
            "k": self.k,
            "backend": self.backend,
            "n_jobs": self.n_jobs,
            "ambient_dimension": self.ambient_dimension,
            "hnsw_config": self.hnsw_config,
            "min_samples": self.min_samples,
            "min_cluster_size": self.min_cluster_size,
            "dense_core_quantile": self.dense_core_quantile,
            "tail_probe_quantile": self.tail_probe_quantile,
            "tail_probe_outer_rank": self.tail_probe_outer_rank,
            "tail_probe_alpha": self.tail_probe_alpha,
            "tail_core_quantile": self.tail_core_quantile,
            "tail_multiple_components": self.tail_multiple_components,
            "tail_seed_min_size": self.tail_seed_min_size,
            "border_multiplier": self.border_multiplier,
            "uniform_tail_config": self.uniform_tail_config,
            "multiscale_config": self.multiscale_config,
        }

    @staticmethod
    def _novel_clusters(candidate: np.ndarray, baseline: np.ndarray) -> list[np.ndarray]:
        clusters: list[np.ndarray] = []
        for label in np.unique(candidate[candidate >= 0]):
            members = candidate == label
            if np.mean(baseline[members] < 0) >= 0.50:
                clusters.append(members)
        return clusters

    @staticmethod
    def _best_jaccard(target: np.ndarray, labels: np.ndarray) -> float:
        if not np.any(target):
            return 0.0
        best = 0.0
        for label in np.unique(labels[labels >= 0]):
            predicted = labels == label
            union = int(np.sum(target | predicted))
            if union:
                best = max(best, float(np.sum(target & predicted)) / union)
        return best

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "StableDBCVTailStrataSCAN":
        del y
        values = np.asarray(X, dtype=np.float32, order="C")
        if values.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if self.stability_subsamples < 1:
            raise ValueError("stability_subsamples must be positive")
        if not 0.0 < self.stability_fraction < 1.0:
            raise ValueError("stability_fraction must lie in (0, 1)")
        if not 0.0 <= self.stability_jaccard <= 1.0:
            raise ValueError("stability_jaccard must lie in [0, 1]")

        graph, graph_seconds = build_knn_graph(
            values,
            k=self.k,
            backend=self.backend,
            n_jobs=self.n_jobs,
            hnsw_config=self.hnsw_config,
        )
        dimension = float(values.shape[1]) if self.ambient_dimension is None else float(self.ambient_dimension)
        self.fit_from_graph(graph, ambient_dimension=dimension)
        candidate_labels = self.labels_.copy()
        candidate_core = self.core_sample_indices_.copy()
        candidate_profile = dict(self.profile_)

        baseline = StrataSCAN(**self._model_kwargs()).fit_from_graph(
            graph, ambient_dimension=dimension
        )
        novel = self._novel_clusters(candidate_labels, baseline.labels_)
        scores: list[float] = []
        if novel:
            rng = np.random.default_rng(self.random_state)
            sample_size = max(self.k + 1, int(np.floor(self.stability_fraction * values.shape[0])))
            for draw in range(self.stability_subsamples):
                rows = np.sort(rng.choice(values.shape[0], size=sample_size, replace=False))
                subsample = PersistentDBCVTailStrataSCAN(**self._model_kwargs()).fit(values[rows])
                scores.append(
                    max(
                        self._best_jaccard(cluster[rows], subsample.labels_)
                        for cluster in novel
                    )
                )
        accepted = bool(scores) and bool(np.median(scores) >= self.stability_jaccard)
        if accepted:
            self.labels_ = candidate_labels
            self.core_sample_indices_ = candidate_core
            self.n_clusters_ = int(np.unique(candidate_labels[candidate_labels >= 0]).size)
            self.profile_ = candidate_profile
        else:
            self.labels_ = baseline.labels_.copy()
            self.core_sample_indices_ = baseline.core_sample_indices_.copy()
            self.n_clusters_ = baseline.n_clusters_
            self.profile_ = dict(baseline.profile_)
        self.graph_ = graph
        self.profile_ = {
            "graph_seconds": float(graph_seconds),
            **self.profile_,
            "experimental_stable_dbcv_accepted": accepted,
            "experimental_stable_dbcv_novel_clusters": len(novel),
            "experimental_stable_dbcv_jaccards": scores,
            "experimental_stable_dbcv_subsamples": self.stability_subsamples,
            "experimental_stable_dbcv_fraction": self.stability_fraction,
            "experimental_stable_dbcv_threshold": self.stability_jaccard,
        }
        return self


class GammaLLRBootstrapTailStrataSCAN(StrataSCAN):
    """Experimental local Gamma-rate scan calibrated over the full tail scan."""

    def __init__(self, *args, bootstrap_replicates: int = 39, random_state: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrap_replicates = int(bootstrap_replicates)
        self.random_state = int(random_state)

    def _config(self):
        return replace(
            super()._config(),
            tail_probe_method="gamma_llr_bootstrap",
            tail_llr_bootstrap_replicates=self.bootstrap_replicates,
            tail_llr_random_state=self.random_state,
        )


def _merge_pass_labels(
    output: np.ndarray,
    remaining: np.ndarray,
    local_labels: np.ndarray,
    *,
    next_label: int,
    min_cluster_size: int,
    max_cluster_fraction: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Merge sufficiently large local clusters into a global label vector."""

    accepted = np.zeros(local_labels.size, dtype=bool)
    for local_label in np.unique(local_labels[local_labels >= 0]):
        members = local_labels == local_label
        size = int(np.sum(members))
        if size < min_cluster_size or size > max_cluster_fraction * local_labels.size:
            continue
        output[remaining[members]] = next_label
        accepted[members] = True
        next_label += 1
    return next_label, accepted


class ResidualStrataSCAN:
    """Experimental multi-stratum StrataSCAN with residual re-clustering.

    Frozen :class:`StrataSCAN` defaults are intentionally untouched. This
    estimator retains intermediate gamma-mixture components as operational
    density strata, estimates intrinsic rather than ambient dimension, and
    optionally repeats clustering on observations left as noise.
    """

    def __init__(
        self,
        *,
        k: int = 32,
        backend: Backend = "auto",
        n_jobs: int = 1,
        hnsw_config: HNSWConfig | None = None,
        min_samples: int = 5,
        min_cluster_size: int = 5,
        accepted_cluster_size: int = 20,
        max_passes: int = 2,
        min_residual_size: int = 500,
        residual_max_cluster_fraction: float = 0.20,
        dense_core_quantile: float = 0.95,
        border_multiplier: float = 1.25,
        background_posterior_threshold: float = 0.50,
        background_components: int = 1,
        projection_components: int | None = None,
    ) -> None:
        if accepted_cluster_size < 1:
            raise ValueError("accepted_cluster_size must be positive")
        if max_passes < 1:
            raise ValueError("max_passes must be positive")
        if min_residual_size < 1:
            raise ValueError("min_residual_size must be positive")
        if not 0.0 < residual_max_cluster_fraction <= 1.0:
            raise ValueError("residual_max_cluster_fraction must lie in (0, 1]")
        if projection_components is not None and projection_components < 2:
            raise ValueError("projection_components must be at least 2 or None")
        self.k = int(k)
        self.backend = backend
        self.n_jobs = int(n_jobs)
        self.hnsw_config = hnsw_config
        self.min_samples = int(min_samples)
        self.min_cluster_size = int(min_cluster_size)
        self.accepted_cluster_size = int(accepted_cluster_size)
        self.max_passes = int(max_passes)
        self.min_residual_size = int(min_residual_size)
        self.residual_max_cluster_fraction = float(residual_max_cluster_fraction)
        self.dense_core_quantile = float(dense_core_quantile)
        self.border_multiplier = float(border_multiplier)
        self.background_posterior_threshold = float(background_posterior_threshold)
        self.background_components = int(background_components)
        self.projection_components = (
            None if projection_components is None else int(projection_components)
        )

    def _pass_model(self) -> StrataSCAN:
        uniform_tail = UniformTailConfig(
            background_posterior_threshold=self.background_posterior_threshold,
            retain_intermediate_components=True,
            background_components=self.background_components,
        )
        return StrataSCAN(
            k=self.k,
            backend=self.backend,
            n_jobs=self.n_jobs,
            ambient_dimension=None,
            hnsw_config=self.hnsw_config,
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            dense_core_quantile=self.dense_core_quantile,
            tail_probe_alpha=0.10,
            tail_multiple_components=True,
            tail_seed_min_size=self.min_cluster_size,
            border_multiplier=self.border_multiplier,
            uniform_tail_config=uniform_tail,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "ResidualStrataSCAN":
        del y
        values = np.asarray(X, dtype=np.float32, order="C")
        if values.ndim != 2:
            raise ValueError("X must be a 2-D array")
        geometry = values
        projection_variance: list[float] = []
        if (
            self.projection_components is not None
            and 1 < self.projection_components < values.shape[1]
        ):
            projector = PCA(
                n_components=self.projection_components,
                svd_solver="randomized",
                random_state=42,
            )
            geometry = np.asarray(projector.fit_transform(values), dtype=np.float32)
            projection_variance = projector.explained_variance_ratio_.astype(float).tolist()
        output = np.full(values.shape[0], -1, dtype=np.int64)
        remaining = np.arange(values.shape[0], dtype=np.int64)
        core_parts: list[np.ndarray] = []
        pass_profiles: list[dict[str, object]] = []
        next_label = 0
        started = perf_counter()

        for pass_index in range(self.max_passes):
            if remaining.size < max(self.min_residual_size, self.k + 1):
                break
            model = self._pass_model()
            pass_started = perf_counter()
            graph, graph_seconds = build_knn_graph(
                geometry[remaining],
                k=self.k,
                backend=self.backend,
                n_jobs=self.n_jobs,
                hnsw_config=self.hnsw_config,
            )
            # fit_from_graph preserves ambient_dimension=None, deliberately
            # activating the intrinsic-dimension estimator.
            local_labels = model.fit_predict_from_graph(graph, ambient_dimension=None)
            model.profile_ = {"graph_seconds": float(graph_seconds), **model.profile_}
            next_label, accepted = _merge_pass_labels(
                output,
                remaining,
                local_labels,
                next_label=next_label,
                min_cluster_size=self.accepted_cluster_size,
                max_cluster_fraction=(
                    1.0 if pass_index == 0 else self.residual_max_cluster_fraction
                ),
            )
            local_core = np.zeros(local_labels.size, dtype=bool)
            local_core[model.core_sample_indices_] = True
            core_parts.append(remaining[local_core & accepted])
            pass_profiles.append(
                {
                    "pass": pass_index + 1,
                    "input_size": int(remaining.size),
                    "accepted_fraction": float(np.mean(accepted)),
                    "accepted_clusters": int(np.unique(local_labels[accepted]).size),
                    "runtime_seconds": float(perf_counter() - pass_started),
                    **model.profile_,
                }
            )
            if not np.any(accepted):
                break
            remaining = remaining[~accepted]

        self.labels_ = output
        self.core_sample_indices_ = (
            np.sort(np.concatenate(core_parts)) if core_parts else np.empty(0, dtype=np.int64)
        )
        self.n_clusters_ = int(next_label)
        self.profile_ = {
            "experimental_variant": "intrinsic-multistratum-residual-v2",
            "experimental_max_passes": self.max_passes,
            "experimental_accepted_cluster_size": self.accepted_cluster_size,
            "experimental_residual_max_cluster_fraction": (
                self.residual_max_cluster_fraction
            ),
            "experimental_projection_components": self.projection_components,
            "experimental_projection_explained_variance_ratio": projection_variance,
            "experimental_assigned_fraction": float(np.mean(output >= 0)),
            "experimental_runtime_seconds": float(perf_counter() - started),
            "experimental_passes": pass_profiles,
        }
        return self

    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        return self.fit(X, y).labels_
