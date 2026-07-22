from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .core import stratified_knn_dbscan_from_graph
from .types import KNNGraph
from .uniform_tail import UniformTailConfig, estimate_stratification_uniform_tail

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*_args, **_kwargs):
        def decorate(function):
            return function
        return decorate


@njit(cache=True)
def _rank_window_component_sizes(
    indices: np.ndarray,
    ordered_rows: np.ndarray,
    width: int,
    edge_rank: int,
) -> np.ndarray:
    n = indices.shape[0]
    windows = (ordered_rows.size + width - 1) // width
    window_id = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    for position in range(ordered_rows.size):
        row = ordered_rows[position]
        window_id[row] = position // width
        parent[row] = row

    for position in range(ordered_rows.size):
        row = ordered_rows[position]
        row_window = window_id[row]
        for edge in range(edge_rank):
            neighbor = indices[row, edge]
            if window_id[neighbor] != row_window:
                continue
            left = row
            while parent[left] != left:
                parent[left] = parent[parent[left]]
                left = parent[left]
            right = neighbor
            while parent[right] != right:
                parent[right] = parent[parent[right]]
                right = parent[right]
            if left != right:
                parent[right] = left

    counts = np.zeros(n, dtype=np.int32)
    largest = np.zeros(windows, dtype=np.int32)
    for position in range(ordered_rows.size):
        row = ordered_rows[position]
        root = row
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        counts[root] += 1
        window = window_id[row]
        if counts[root] > largest[window]:
            largest[window] = counts[root]
    return largest


@dataclass(frozen=True, slots=True)
class GammaStrictCoreConfig:
    min_samples: int = 5
    min_cluster_size: int = 5
    dense_core_quantile: float = 0.95
    tail_probe_quantile: float = 0.02
    tail_probe_outer_rank: int = 32
    tail_probe_alpha: float = 0.05
    tail_core_quantile: float = 0.05
    border_multiplier: float = 1.25
    uniform_tail: UniformTailConfig = field(default_factory=UniformTailConfig)


@dataclass(frozen=True, slots=True)
class GammaStrictCoreResult:
    labels: np.ndarray
    core_mask: np.ndarray
    profile: dict[str, float | int | list[float] | list[int]]


def _tail_rank_window_pvalue(
    graph: KNNGraph,
    active: np.ndarray,
    score: np.ndarray,
    *,
    quantile: float,
    min_samples: int,
) -> tuple[float, int, int, int]:
    """Empirical null from equally sized score slices in the same tail graph."""

    rows = np.flatnonzero(active)
    order = rows[np.argsort(score[rows], kind="stable")]
    width = max(min_samples, int(np.ceil(quantile * rows.size)))
    sizes = _rank_window_component_sizes(
        np.asarray(graph.indices, dtype=np.int32),
        np.asarray(order, dtype=np.int32),
        width,
        min_samples - 1,
    )
    if sizes.size < 2:
        return 1.0, 0, width, 0
    observed = int(sizes[0])
    null = sizes[1:]
    pvalue = (1 + int(np.sum(null >= observed))) / sizes.size
    return float(pvalue), observed, width, int(null.size)


def _candidate_component_seed_mask(
    graph: KNNGraph,
    candidates: np.ndarray,
    probe: np.ndarray,
    *,
    min_samples: int,
) -> np.ndarray:
    """Return the q-core component with greatest overlap with the probe."""

    rank = min_samples - 1
    rows_idx = np.flatnonzero(candidates).astype(np.int32)
    out = np.zeros(graph.n_samples, dtype=bool)
    if rows_idx.size == 0:
        return out
    local = np.full(graph.n_samples, -1, dtype=np.int32)
    local[rows_idx] = np.arange(rows_idx.size, dtype=np.int32)
    rows = np.repeat(rows_idx, rank)
    cols = graph.indices[rows_idx, :rank].reshape(-1)
    keep = candidates[cols]
    adjacency = coo_matrix(
        (
            np.ones(int(np.sum(keep)), dtype=np.uint8),
            (local[rows[keep]], local[cols[keep]]),
        ),
        shape=(rows_idx.size, rows_idx.size),
    ).tocsr()
    count, labels = connected_components(adjacency, directed=False, return_labels=True)
    overlap = np.bincount(labels, weights=probe[rows_idx].astype(float), minlength=count)
    chosen = int(np.argmax(overlap))
    out[rows_idx[labels == chosen]] = True
    return out


def gamma_strict_core_from_graph(
    graph: KNNGraph,
    *,
    ambient_dimension: float,
    config: GammaStrictCoreConfig | None = None,
) -> GammaStrictCoreResult:
    """Gamma strata with strict core seeds and independently expanded borders."""

    cfg = config or GammaStrictCoreConfig()
    for value, name in (
        (cfg.dense_core_quantile, "dense_core_quantile"),
        (cfg.tail_probe_quantile, "tail_probe_quantile"),
        (cfg.tail_probe_alpha, "tail_probe_alpha"),
        (cfg.tail_core_quantile, "tail_core_quantile"),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0, 1)")
    if cfg.border_multiplier <= 0.0:
        raise ValueError("border_multiplier must be positive")
    if not cfg.min_samples - 1 < cfg.tail_probe_outer_rank <= graph.k:
        raise ValueError("tail_probe_outer_rank must exceed the core rank and fit the graph")

    strat = estimate_stratification_uniform_tail(
        graph,
        eps_rank=cfg.min_samples - 1,
        ambient_dimension=ambient_dimension,
        config=cfg.uniform_tail,
    )
    groups = np.asarray(strat.groups, dtype=np.int32)
    dense_groups = set(map(int, strat.supported_groups.tolist()))
    eps = np.zeros(strat.selected_components, dtype=np.float32)
    point_eps = np.zeros(graph.n_samples, dtype=np.float32)
    core_seed_mask = np.ones(graph.n_samples, dtype=bool)
    tail_probe_pvalues: list[float] = []
    tail_probe_component_sizes: list[int] = []
    tail_probe_null_windows: list[int] = []
    activated_tail_groups: list[int] = []
    kth = graph.distances[:, cfg.min_samples - 2]

    for group in range(strat.selected_components):
        mask = groups == group
        values = kth[mask]
        if values.size == 0:
            continue
        if group in dense_groups:
            eps[group] = np.quantile(values, cfg.dense_core_quantile)
            point_eps[mask] = eps[group]
            continue
        contrast = kth / np.maximum(
            graph.distances[:, cfg.tail_probe_outer_rank - 1],
            np.finfo(np.float32).tiny,
        )
        pvalue, component_size, _, completed = _tail_rank_window_pvalue(
            graph,
            mask,
            contrast,
            quantile=cfg.tail_probe_quantile,
            min_samples=cfg.min_samples,
        )
        tail_probe_pvalues.append(pvalue)
        tail_probe_component_sizes.append(component_size)
        tail_probe_null_windows.append(completed)
        if pvalue < cfg.tail_probe_alpha:
            eps[group] = np.quantile(values, cfg.tail_core_quantile)
            point_eps[mask] = eps[group]
            probe_threshold = float(np.quantile(contrast[mask], cfg.tail_probe_quantile))
            probe = mask & (contrast <= probe_threshold)
            candidates = mask & (kth <= eps[group])
            selected_component = _candidate_component_seed_mask(
                graph,
                candidates,
                probe,
                min_samples=cfg.min_samples,
            )
            core_seed_mask[mask] = selected_component[mask]
            activated_tail_groups.append(group)

    labels, core, clustering_profile = stratified_knn_dbscan_from_graph(
        graph,
        point_eps,
        min_samples=cfg.min_samples,
        min_cluster_size=cfg.min_cluster_size,
        border_multiplier=cfg.border_multiplier,
        core_seed_mask=core_seed_mask,
    )
    return GammaStrictCoreResult(
        labels=labels,
        core_mask=core,
        profile={
            "strict_core_dense_quantile": cfg.dense_core_quantile,
            "strict_core_tail_probe_quantile": cfg.tail_probe_quantile,
            "strict_core_tail_probe_outer_rank": cfg.tail_probe_outer_rank,
            "strict_core_tail_probe_alpha": cfg.tail_probe_alpha,
            "strict_core_tail_quantile": cfg.tail_core_quantile,
            "strict_core_border_multiplier": cfg.border_multiplier,
            "strict_core_dense_groups": len(dense_groups),
            "strict_core_tail_probe_pvalues": tail_probe_pvalues,
            "strict_core_tail_probe_component_sizes": tail_probe_component_sizes,
            "strict_core_tail_probe_null_windows": tail_probe_null_windows,
            "strict_core_activated_tail_groups": activated_tail_groups,
            "strict_core_core_fraction": float(np.mean(core)),
            "strict_core_active_fraction": float(np.mean(labels >= 0)),
            "strict_core_clusters": int(np.unique(labels[labels >= 0]).size),
            "strict_core_gamma_components": int(strat.diagnostics["uniform_tail_components"]),
            "strict_core_gamma_background_fraction": float(
                strat.diagnostics["uniform_tail_background_fraction"]
            ),
            **clustering_profile,
        },
    )
