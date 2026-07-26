from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA

from .core import StrataSCAN
from .neighbors import Backend, HNSWConfig, build_knn_graph
from .uniform_tail import UniformTailConfig


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
