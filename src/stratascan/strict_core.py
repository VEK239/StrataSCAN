from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.stats import hypergeom
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from .core import stratified_knn_dbscan_from_graph
from .stratification import StratificationResult
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


@njit(cache=True)
def _largest_mask_component(
    indices: np.ndarray,
    candidates: np.ndarray,
    edge_rank: int,
) -> int:
    """Return the largest induced component for one site-percolation mask."""

    n = indices.shape[0]
    parent = np.full(n, -1, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.int32)
    for row in range(n):
        if candidates[row]:
            parent[row] = row
            sizes[row] = 1
    for row in range(n):
        if not candidates[row]:
            continue
        for edge in range(edge_rank):
            neighbor = indices[row, edge]
            if not candidates[neighbor]:
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
                if sizes[left] < sizes[right]:
                    left, right = right, left
                parent[right] = left
                sizes[left] += sizes[right]
    largest = 0
    for row in range(n):
        if parent[row] == row and sizes[row] > largest:
            largest = sizes[row]
    return largest


@dataclass(frozen=True, slots=True)
class GammaStrictCoreConfig:
    min_samples: int = 5
    min_cluster_size: int = 5
    dense_core_quantile: float = 0.95
    min_core_seeds_per_stratum: int = 0
    tail_probe_quantile: float = 0.02
    tail_probe_outer_rank: int = 32
    tail_probe_alpha: float = 0.05
    tail_core_quantile: float = 0.05
    tail_multiple_components: bool = False
    tail_seed_min_size: int = 5
    tail_probe_method: str = "rank_window"
    tail_percolation_permutations: int = 199
    tail_percolation_density_bins: int = 4
    tail_percolation_degree_bins: int = 4
    tail_percolation_random_state: int = 42
    tail_persistence_levels: int = 16
    tail_persistence_max_quantile: float = 0.30
    tail_llr_bootstrap_replicates: int = 39
    tail_llr_random_state: int = 42
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
    select_multiple: bool = False,
    min_component_size: int = 5,
) -> np.ndarray:
    """Return one or more q-core components supported by the tail probe."""

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
    sizes = np.bincount(labels, minlength=count)
    if select_multiple:
        chosen = np.flatnonzero(sizes >= max(min_samples, min_component_size))
        if chosen.size:
            out[rows_idx[np.isin(labels, chosen)]] = True
    else:
        chosen = int(np.argmax(overlap))
        out[rows_idx[labels == chosen]] = True
    return out


def _candidate_component_labels(
    graph: KNNGraph,
    candidates: np.ndarray,
    *,
    min_samples: int,
    min_component_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Label candidate components and return labels that are large enough."""

    rank = min_samples - 1
    rows_idx = np.flatnonzero(candidates).astype(np.int32)
    labels = np.full(graph.n_samples, -1, dtype=np.int32)
    if rows_idx.size == 0:
        return labels, np.empty(0, dtype=np.int32)
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
    count, local_labels = connected_components(adjacency, directed=False, return_labels=True)
    labels[rows_idx] = local_labels
    sizes = np.bincount(local_labels, minlength=count)
    eligible = np.flatnonzero(sizes >= max(min_samples, min_component_size)).astype(np.int32)
    return labels, eligible


def _tail_ranked_shell_enrichment(
    graph: KNNGraph,
    active: np.ndarray,
    candidates: np.ndarray,
    *,
    ambient_dimension: float,
    min_samples: int,
    outer_rank: int,
    min_component_size: int,
    select_multiple: bool,
    alpha: float,
) -> tuple[np.ndarray, float, float, int, int, int, int]:
    """Cutoff-free enrichment scan using a shell not used to form candidates.

    Candidate components use ranks 1..``min_samples - 1``.  The ranked score
    uses the next-neighbour volume increment relative to the remaining outer
    shell, reducing the direct reuse of the candidate-selection statistic.
    A one-sided hypergeometric tail is scanned at every component hit and a
    Bonferroni correction covers every tested cutoff in every component.
    """

    labels, eligible = _candidate_component_labels(
        graph,
        candidates,
        min_samples=min_samples,
        min_component_size=min_component_size,
    )
    selected = np.zeros(graph.n_samples, dtype=bool)
    if eligible.size == 0:
        return selected, 1.0, 1.0, 0, 0, 0, 0

    inner_rank = min_samples - 1
    validation_rank = min_samples
    distances = graph.distances.astype(np.float64, copy=False)
    outer = np.maximum(distances[:, outer_rank - 1], np.finfo(np.float64).tiny)
    inner_ratio = np.clip(distances[:, inner_rank - 1] / outer, 0.0, 1.0)
    validation_ratio = np.clip(distances[:, validation_rank - 1] / outer, 0.0, 1.0)
    inner_volume = np.power(inner_ratio, ambient_dimension)
    validation_volume = np.power(validation_ratio, ambient_dimension)
    score = (validation_volume - inner_volume) / np.maximum(
        1.0 - inner_volume,
        np.finfo(np.float64).tiny,
    )

    active_rows = np.flatnonzero(active)
    order = active_rows[np.argsort(score[active_rows], kind="stable")]
    ranks = np.full(graph.n_samples, -1, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.int64)

    scans: list[tuple[int, float, int, int, int]] = []
    total_tests = 0
    minimum_hits = max(min_samples, min_component_size)
    for component in eligible:
        members = np.flatnonzero(labels == component)
        positions = np.sort(ranks[members])
        hits = np.arange(1, positions.size + 1, dtype=np.int64)
        valid = hits >= minimum_hits
        if not np.any(valid):
            continue
        tested_positions = positions[valid]
        tested_hits = hits[valid]
        pvalues = hypergeom.sf(
            tested_hits - 1,
            active_rows.size,
            members.size,
            tested_positions,
        )
        best = int(np.argmin(pvalues))
        scans.append(
            (
                int(component),
                float(pvalues[best]),
                int(tested_positions[best]),
                int(tested_hits[best]),
                int(members.size),
            )
        )
        total_tests += int(tested_positions.size)

    if not scans:
        return selected, 1.0, 1.0, 0, 0, 0, 0
    best_scan = min(scans, key=lambda item: item[1])
    adjusted = min(1.0, best_scan[1] * total_tests)
    if adjusted < alpha:
        if select_multiple:
            chosen = [scan[0] for scan in scans if min(1.0, scan[1] * total_tests) < alpha]
        else:
            chosen = [best_scan[0]]
        selected = active & np.isin(labels, chosen)
    return (
        selected,
        float(adjusted),
        float(best_scan[1]),
        int(best_scan[4]),
        int(best_scan[3]),
        int(best_scan[2]),
        int(total_tests),
    )


def _equal_count_bins(values: np.ndarray, rows: np.ndarray, bins: int) -> np.ndarray:
    """Assign selected rows to deterministic, approximately equal-count bins."""

    out = np.full(values.size, -1, dtype=np.int32)
    if rows.size == 0:
        return out
    order = rows[np.argsort(values[rows], kind="stable")]
    out[order] = np.minimum(
        bins - 1,
        np.arange(order.size, dtype=np.int64) * bins // order.size,
    ).astype(np.int32)
    return out


def _tail_component_percolation_test(
    graph: KNNGraph,
    active: np.ndarray,
    candidates: np.ndarray,
    *,
    min_samples: int,
    min_component_size: int,
    select_multiple: bool,
    alpha: float,
    permutations: int,
    density_bins: int,
    degree_bins: int,
    random_state: int,
) -> tuple[np.ndarray, float, int, float, int, float]:
    """Test candidate connectivity by stratified site percolation.

    The null keeps the observed tail graph fixed and preserves the number of
    candidate sites within joint outer-density and graph-degree bins.  The
    maximum component size is recomputed for every draw, so the resulting
    Monte Carlo p-value is global over all observed components.
    """

    labels, eligible = _candidate_component_labels(
        graph,
        candidates,
        min_samples=min_samples,
        min_component_size=min_component_size,
    )
    selected = np.zeros(graph.n_samples, dtype=bool)
    if eligible.size == 0:
        return selected, 1.0, 0, 0.0, permutations, 0.0
    component_sizes = np.array(
        [int(np.sum(labels == component)) for component in eligible],
        dtype=np.int64,
    )
    observed = int(np.max(component_sizes))

    rank = min_samples - 1
    active_rows = np.flatnonzero(active)
    outer_distance = graph.distances[:, -1].astype(np.float64, copy=False)
    density_bin = _equal_count_bins(outer_distance, active_rows, density_bins)

    outgoing = np.sum(active[graph.indices[:, :rank]], axis=1).astype(np.int64)
    incoming = np.bincount(
        graph.indices[active_rows, :rank].reshape(-1),
        weights=np.repeat(active[active_rows], rank),
        minlength=graph.n_samples,
    )
    degree = outgoing.astype(np.float64) + incoming
    degree_bin = np.full(graph.n_samples, -1, dtype=np.int32)
    for density in range(density_bins):
        rows = active_rows[density_bin[active_rows] == density]
        local = _equal_count_bins(degree, rows, degree_bins)
        degree_bin[rows] = local[rows]
    joint_bin = density_bin * degree_bins + degree_bin

    bin_members: list[np.ndarray] = []
    bin_candidate_counts: list[int] = []
    for group in range(density_bins * degree_bins):
        members = active_rows[joint_bin[active_rows] == group]
        count = int(np.sum(candidates[members]))
        if members.size and count:
            bin_members.append(members)
            bin_candidate_counts.append(count)

    rng = np.random.default_rng(random_state)
    null_largest = np.empty(permutations, dtype=np.int32)
    null_mask = np.zeros(graph.n_samples, dtype=bool)
    indices = np.asarray(graph.indices, dtype=np.int32)
    for draw in range(permutations):
        null_mask.fill(False)
        for members, count in zip(bin_members, bin_candidate_counts, strict=True):
            chosen = rng.choice(members, size=count, replace=False)
            null_mask[chosen] = True
        null_largest[draw] = _largest_mask_component(indices, null_mask, rank)

    pvalue = (1 + int(np.sum(null_largest >= observed))) / (permutations + 1)
    null_q95 = float(np.quantile(null_largest, 0.95, method="higher"))
    null_mean = float(np.mean(null_largest))
    if pvalue < alpha:
        if select_multiple:
            chosen = eligible[component_sizes > null_q95]
            if chosen.size == 0:
                chosen = eligible[component_sizes == observed]
        else:
            chosen = eligible[[int(np.argmax(component_sizes))]]
        selected = active & np.isin(labels, chosen)
    return selected, float(pvalue), observed, null_q95, permutations, null_mean


def _tail_component_multiscale_test(
    graph: KNNGraph,
    active: np.ndarray,
    candidates: np.ndarray,
    *,
    min_samples: int,
    min_component_size: int,
    select_multiple: bool,
    alpha: float,
    permutations: int,
    density_bins: int,
    degree_bins: int,
    random_state: int,
) -> tuple[np.ndarray, float, int, float, int, float]:
    """Test whether a 4NN component stays dense across larger neighbour ranks.

    The component score is the negative mean robust residual of its median
    ``log(d_k / d_4)`` profile.  Residuals are conditioned on ``d_4`` within
    the active tail.  The Monte Carlo null redraws candidate sites within
    outer-density/degree strata and rebuilds all components, so its maximum
    score accounts for both spatial autocorrelation and component selection.
    """

    labels, eligible = _candidate_component_labels(
        graph,
        candidates,
        min_samples=min_samples,
        min_component_size=min_component_size,
    )
    selected = np.zeros(graph.n_samples, dtype=bool)
    if eligible.size == 0:
        return selected, 1.0, 0, 0.0, permutations, 0.0

    inner_rank = min_samples - 1
    ranks = np.unique(
        np.minimum(
            np.asarray([inner_rank, 2 * inner_rank, 4 * inner_rank, 8 * inner_rank]),
            graph.k,
        )
    )
    if ranks.size < 2:
        return selected, 1.0, 0, 0.0, permutations, 0.0
    tiny = np.finfo(np.float64).tiny
    log_distances = np.log(
        np.maximum(graph.distances[:, ranks - 1].astype(np.float64, copy=False), tiny)
    )
    growth = log_distances[:, 1:] - log_distances[:, [0]]
    active_rows = np.flatnonzero(active)

    inner_bins = max(4, density_bins * degree_bins)
    inner_bin = _equal_count_bins(log_distances[:, 0], active_rows, inner_bins)
    residual = np.zeros_like(growth)
    for bin_id in range(inner_bins):
        rows = active_rows[inner_bin[active_rows] == bin_id]
        if rows.size == 0:
            continue
        center = np.median(growth[rows], axis=0)
        scale = 1.4826 * np.median(np.abs(growth[rows] - center), axis=0)
        scale = np.maximum(scale, 1e-6)
        residual[rows] = (growth[rows] - center) / scale

    def component_score(component_labels: np.ndarray, component: int) -> float:
        members = component_labels == component
        return float(-np.mean(np.median(residual[members], axis=0)))

    observed_scores = np.asarray(
        [component_score(labels, int(component)) for component in eligible], dtype=float
    )
    observed = float(np.max(observed_scores))

    outer_bin = _equal_count_bins(log_distances[:, -1], active_rows, density_bins)
    outgoing = np.sum(active[graph.indices[:, :inner_rank]], axis=1).astype(np.float64)
    incoming = np.bincount(
        graph.indices[active_rows, :inner_rank].reshape(-1), minlength=graph.n_samples
    ).astype(np.float64)
    degree = outgoing + incoming
    degree_bin = np.full(graph.n_samples, -1, dtype=np.int32)
    for density in range(density_bins):
        rows = active_rows[outer_bin[active_rows] == density]
        local = _equal_count_bins(degree, rows, degree_bins)
        degree_bin[rows] = local[rows]
    joint_bin = outer_bin * degree_bins + degree_bin

    strata: list[tuple[np.ndarray, int]] = []
    for group in range(density_bins * degree_bins):
        rows = active_rows[joint_bin[active_rows] == group]
        count = int(np.sum(candidates[rows]))
        if rows.size and count:
            strata.append((rows, count))

    rng = np.random.default_rng(random_state)
    null_scores = np.zeros(permutations, dtype=float)
    null_mask = np.zeros(graph.n_samples, dtype=bool)
    for draw in range(permutations):
        null_mask.fill(False)
        for rows, count in strata:
            null_mask[rng.choice(rows, size=count, replace=False)] = True
        null_labels, null_eligible = _candidate_component_labels(
            graph,
            null_mask,
            min_samples=min_samples,
            min_component_size=min_component_size,
        )
        null_scores[draw] = max(
            (component_score(null_labels, int(component)) for component in null_eligible),
            default=0.0,
        )

    pvalue = (1.0 + float(np.sum(null_scores >= observed))) / (permutations + 1.0)
    if pvalue < alpha:
        if select_multiple:
            cutoff = float(np.quantile(null_scores, 1.0 - alpha, method="higher"))
            chosen = eligible[observed_scores > cutoff]
        else:
            chosen = eligible[[int(np.argmax(observed_scores))]]
        selected = active & np.isin(labels, chosen)
    component_size = int(np.sum(labels == eligible[int(np.argmax(observed_scores))]))
    return (
        selected,
        float(pvalue),
        component_size,
        observed,
        permutations,
        float(np.mean(null_scores)),
    )


def _tail_component_adaptive_multiscale_test(
    graph: KNNGraph,
    active: np.ndarray,
    candidates: np.ndarray,
    *,
    min_samples: int,
    min_component_size: int,
    select_multiple: bool,
    alpha: float,
    permutations: int,
    density_bins: int,
    degree_bins: int,
    random_state: int,
) -> tuple[
    np.ndarray,
    float,
    int,
    float,
    int,
    float,
    int,
    list[float],
    list[float],
    list[int],
]:
    """Scan powers-of-two neighbour ranks until multiscale evidence turns over.

    At every outer rank, the component receives evidence from the number of
    additional within-component neighbours beyond the four ranks used to
    construct it.  The target rank maximises this hold-out connectivity excess
    before larger neighbourhoods dilute it.

    The permutation null repeats component discovery and the rank scan.
    Component scores are compared with null components in the same power-of-two
    size bin, then corrected together with Benjamini-Hochberg.
    """

    labels, eligible = _candidate_component_labels(
        graph,
        candidates,
        min_samples=min_samples,
        min_component_size=min_component_size,
    )
    selected = np.zeros(graph.n_samples, dtype=bool)
    inner_rank = min_samples - 1
    ranks_list = [inner_rank]
    while 2 * ranks_list[-1] <= graph.k:
        ranks_list.append(2 * ranks_list[-1])
    ranks = np.asarray(ranks_list, dtype=np.int32)
    if eligible.size == 0 or ranks.size < 2:
        return selected, 1.0, 0, 0.0, permutations, 0.0, inner_rank, [], [], []

    active_rows = np.flatnonzero(active)
    inner_bins = max(4, density_bins * degree_bins)
    inner_distance = graph.distances[:, inner_rank - 1].astype(np.float64, copy=False)
    inner_bin = _equal_count_bins(inner_distance, active_rows, inner_bins)
    active_size = int(active_rows.size)

    def component_scan(component_labels: np.ndarray, component: int) -> tuple[float, int]:
        members = component_labels == component
        rows = np.flatnonzero(members)
        component_size = int(rows.size)
        probability = max((component_size - 1) / max(active_size - 1, 1), 1e-12)
        scale_evidence = np.zeros(ranks.size - 1, dtype=np.float64)
        for scale_index, outer_rank in enumerate(ranks[1:]):
            trials = float(component_size * (int(outer_rank) - inner_rank))
            observed_edges = float(
                np.sum(members[graph.indices[rows, inner_rank:int(outer_rank)]])
            )
            expected_edges = trials * probability
            variance = max(expected_edges * (1.0 - probability), 1e-9)
            scale_evidence[scale_index] = (
                observed_edges - expected_edges
            ) / np.sqrt(variance)
        best = int(np.argmax(scale_evidence))
        return float(max(0.0, scale_evidence[best])), int(ranks[best + 1])

    observed_scan = [component_scan(labels, int(component)) for component in eligible]
    observed_scores = np.asarray([value[0] for value in observed_scan], dtype=float)
    observed_ranks = np.asarray([value[1] for value in observed_scan], dtype=np.int32)
    best_observed = int(np.argmax(observed_scores))
    observed = float(
        np.sum(observed_scores) if select_multiple else observed_scores[best_observed]
    )
    target_rank = int(observed_ranks[best_observed])

    outgoing = np.sum(active[graph.indices[:, :inner_rank]], axis=1).astype(np.float64)
    incoming = np.bincount(
        graph.indices[active_rows, :inner_rank].reshape(-1), minlength=graph.n_samples
    ).astype(np.float64)
    degree = outgoing + incoming
    degree_bin = np.full(graph.n_samples, -1, dtype=np.int32)
    for density in range(inner_bins):
        rows = active_rows[inner_bin[active_rows] == density]
        local = _equal_count_bins(degree, rows, degree_bins)
        degree_bin[rows] = local[rows]
    joint_bin = inner_bin * degree_bins + degree_bin

    strata: list[tuple[np.ndarray, int]] = []
    for group in range(inner_bins * degree_bins):
        rows = active_rows[joint_bin[active_rows] == group]
        count = int(np.sum(candidates[rows]))
        if rows.size and count:
            strata.append((rows, count))

    rng = np.random.default_rng(random_state)
    null_scores = np.zeros(permutations, dtype=float)
    null_components: list[tuple[int, float]] = []
    null_mask = np.zeros(graph.n_samples, dtype=bool)
    for draw in range(permutations):
        null_mask.fill(False)
        for rows, count in strata:
            null_mask[rng.choice(rows, size=count, replace=False)] = True
        null_labels, null_eligible = _candidate_component_labels(
            graph,
            null_mask,
            min_samples=min_samples,
            min_component_size=min_component_size,
        )
        draw_scores = np.asarray(
            [component_scan(null_labels, int(component))[0] for component in null_eligible],
            dtype=float,
        )
        null_components.extend(
            (int(np.sum(null_labels == component)), float(score))
            for component, score in zip(null_eligible, draw_scores, strict=True)
        )
        null_scores[draw] = float(
            np.sum(draw_scores) if select_multiple else np.max(draw_scores, initial=0.0)
        )

    raw_pvalues = np.ones(eligible.size, dtype=float)
    for index, component in enumerate(eligible):
        component_size = int(np.sum(labels == component))
        size_bin = int(np.floor(np.log2(max(component_size, 1))))
        pool = np.asarray(
            [
                score
                for null_size, score in null_components
                if int(np.floor(np.log2(max(null_size, 1)))) == size_bin
            ],
            dtype=float,
        )
        if pool.size:
            raw_pvalues[index] = (
                1.0 + float(np.sum(pool >= observed_scores[index]))
            ) / (pool.size + 1.0)

    order = np.argsort(raw_pvalues, kind="stable")
    adjusted = np.ones_like(raw_pvalues)
    running = 1.0
    for reverse_position in range(order.size - 1, -1, -1):
        index = int(order[reverse_position])
        running = min(
            running,
            float(raw_pvalues[index]) * order.size / (reverse_position + 1.0),
        )
        adjusted[index] = running
    significant = adjusted <= alpha
    if np.any(significant):
        if select_multiple:
            chosen = eligible[significant]
        else:
            eligible_scores = np.where(significant, observed_scores, -np.inf)
            chosen = eligible[[int(np.argmax(eligible_scores))]]
        selected = active & np.isin(labels, chosen)
    pvalue = float(np.min(adjusted, initial=1.0))
    component_size = int(np.sum(labels == eligible[best_observed]))
    return (
        selected,
        float(pvalue),
        component_size,
        observed,
        permutations,
        float(np.mean(null_scores)),
        target_rank,
        raw_pvalues.astype(float).tolist(),
        adjusted.astype(float).tolist(),
        observed_ranks.astype(int).tolist(),
    )


def _density_component_hierarchy(
    graph: KNNGraph,
    active: np.ndarray,
    density_values: np.ndarray,
    *,
    min_samples: int,
    min_component_size: int,
    levels: int,
    max_quantile: float,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Build persistent branches of induced kNN components over density levels."""

    active_values = density_values[active]
    quantiles = np.linspace(1.0 / levels, max_quantile, levels)
    thresholds = np.unique(np.quantile(active_values, quantiles))
    level_labels: list[np.ndarray] = []
    level_components: list[np.ndarray] = []
    for threshold in thresholds:
        labels, eligible = _candidate_component_labels(
            graph,
            active & (density_values <= threshold),
            min_samples=min_samples,
            min_component_size=min_component_size,
        )
        level_labels.append(labels)
        level_components.append(eligible)

    parent: dict[tuple[int, int], tuple[int, int]] = {}
    child_count: dict[tuple[int, int], int] = {}
    for level in range(len(thresholds) - 1):
        next_eligible = set(map(int, level_components[level + 1]))
        for component in level_components[level]:
            members = np.flatnonzero(level_labels[level] == component)
            if members.size == 0:
                continue
            next_component = int(level_labels[level + 1][members[0]])
            if next_component not in next_eligible:
                continue
            source = (level, int(component))
            target = (level + 1, next_component)
            parent[source] = target
            child_count[target] = child_count.get(target, 0) + 1

    branches: list[dict[str, object]] = []
    visited: set[tuple[int, int]] = set()
    for level, components in enumerate(level_components):
        for component in components:
            node = (level, int(component))
            if node in visited:
                continue
            path = [node]
            visited.add(node)
            while node in parent:
                next_node = parent[node]
                if child_count.get(next_node, 0) != 1:
                    break
                path.append(next_node)
                visited.add(next_node)
                node = next_node
            if len(path) < 2:
                continue
            stability = 0.0
            for left, right in zip(path[:-1], path[1:], strict=True):
                size = int(np.sum(level_labels[left[0]] == left[1]))
                ratio = max(
                    float(thresholds[right[0]]) / max(float(thresholds[left[0]]), np.finfo(float).tiny),
                    1.0,
                )
                stability += size * np.log(ratio)
            last = path[-1]
            members = level_labels[last[0]] == last[1]
            first = path[0]
            seeds = level_labels[first[0]] == first[1]
            branches.append(
                {
                    "members": members,
                    "seeds": seeds,
                    "stability": float(stability),
                    "lifespan": len(path),
                    "birth_threshold": float(thresholds[first[0]]),
                    "merge_threshold": float(thresholds[last[0]]),
                }
            )
    return branches, thresholds


def _component_boundary(graph: KNNGraph, active: np.ndarray, members: np.ndarray) -> np.ndarray:
    boundary = np.zeros(graph.n_samples, dtype=bool)
    rows = np.flatnonzero(members)
    if rows.size:
        neighbours = graph.indices[rows].reshape(-1)
        boundary[neighbours] = active[neighbours] & ~members[neighbours]
    return boundary


def _boundary_log_density_contrast(
    graph: KNNGraph,
    active: np.ndarray,
    members: np.ndarray,
    *,
    min_samples: int,
    density_values: np.ndarray | None = None,
) -> float:
    boundary = _component_boundary(graph, active, members)
    if not np.any(boundary):
        return 0.0
    values = graph.distances[:, min_samples - 2] if density_values is None else density_values
    tiny = np.finfo(np.float64).tiny
    return float(
        np.median(np.log(np.maximum(values[boundary], tiny)))
        - np.median(np.log(np.maximum(values[members], tiny)))
    )


def _dbcv_component_separation(
    graph: KNNGraph,
    active: np.ndarray,
    members: np.ndarray,
    *,
    min_samples: int,
) -> float:
    """DBCV-like separation using mutual-reachability internal MST and cut."""

    rows_idx = np.flatnonzero(members).astype(np.int32)
    if rows_idx.size < min_samples:
        return -1.0
    local = np.full(graph.n_samples, -1, dtype=np.int32)
    local[rows_idx] = np.arange(rows_idx.size, dtype=np.int32)
    core = graph.distances[:, min_samples - 2].astype(np.float64, copy=False)
    edge_rows: list[int] = []
    edge_cols: list[int] = []
    edge_values: list[float] = []
    external: list[float] = []
    for row in rows_idx:
        for rank in range(graph.k):
            neighbour = int(graph.indices[row, rank])
            reach = max(float(core[row]), float(core[neighbour]), float(graph.distances[row, rank]))
            if members[neighbour]:
                edge_rows.append(int(local[row]))
                edge_cols.append(int(local[neighbour]))
                edge_values.append(reach)
            elif active[neighbour]:
                external.append(reach)
    if not edge_values or not external:
        return -1.0
    adjacency = coo_matrix(
        (edge_values, (edge_rows, edge_cols)),
        shape=(rows_idx.size, rows_idx.size),
    ).tocsr()
    tree = minimum_spanning_tree(adjacency.maximum(adjacency.T))
    if tree.nnz < rows_idx.size - 1:
        return -1.0
    sparseness = float(np.max(tree.data))
    separation = float(np.min(external))
    return (separation - sparseness) / max(separation, sparseness, np.finfo(float).tiny)


def _gamma_rate_llr(values: np.ndarray, inside: np.ndarray, outside: np.ndarray, shape: float) -> float:
    inside_values = np.maximum(values[inside], np.finfo(float).tiny)
    outside_values = np.maximum(values[outside], np.finfo(float).tiny)
    if inside_values.size == 0 or outside_values.size == 0:
        return 0.0
    total = np.concatenate((inside_values, outside_values))
    rate_in = shape * inside_values.size / np.sum(inside_values)
    rate_out = shape * outside_values.size / np.sum(outside_values)
    rate_null = shape * total.size / np.sum(total)
    alt = shape * inside_values.size * np.log(rate_in) - rate_in * np.sum(inside_values)
    alt += shape * outside_values.size * np.log(rate_out) - rate_out * np.sum(outside_values)
    null = shape * total.size * np.log(rate_null) - rate_null * np.sum(total)
    return float(max(0.0, 2.0 * (alt - null))) if rate_in > rate_out else 0.0


def _tail_persistent_selector(
    graph: KNNGraph,
    active: np.ndarray,
    *,
    ambient_dimension: float,
    min_samples: int,
    min_component_size: int,
    levels: int,
    max_quantile: float,
    method: str,
    bootstrap_replicates: int = 0,
    random_state: int = 42,
) -> tuple[np.ndarray, float, int, float, int, float]:
    density = graph.distances[:, min_samples - 2].astype(np.float64, copy=False)
    branches, _ = _density_component_hierarchy(
        graph,
        active,
        density,
        min_samples=min_samples,
        min_component_size=min_component_size,
        levels=levels,
        max_quantile=max_quantile,
    )
    selected = np.zeros(graph.n_samples, dtype=bool)
    if not branches:
        return selected, 1.0, 0, 0.0, bootstrap_replicates, 0.0

    def branch_score(branch: dict[str, object], values: np.ndarray = density) -> float:
        members = np.asarray(branch["members"], dtype=bool)
        stability = float(branch["stability"])
        if method == "persistent_contrast":
            contrast = _boundary_log_density_contrast(
                graph, active, members, min_samples=min_samples, density_values=values
            )
            return stability * max(0.0, contrast)
        if method in {"persistent_dbcv", "persistent_dbcv_bootstrap"}:
            return stability * max(
                0.0,
                _dbcv_component_separation(graph, active, members, min_samples=min_samples),
            )
        boundary = _component_boundary(graph, active, members)
        volumes = np.power(np.maximum(values, np.finfo(float).tiny), ambient_dimension)
        return _gamma_rate_llr(volumes, members, boundary, float(min_samples - 1))

    scores = np.array([branch_score(branch) for branch in branches], dtype=float)
    best = int(np.argmax(scores))
    observed = float(scores[best])
    chosen = branches[best]
    component_size = int(np.sum(np.asarray(chosen["seeds"], dtype=bool)))
    pvalue = 0.0 if observed > 0.0 else 1.0
    null_mean = 0.0
    if method in {"gamma_llr_bootstrap", "persistent_dbcv_bootstrap"}:
        rng = np.random.default_rng(random_state)
        active_rows = np.flatnonzero(active)
        null_scores = np.zeros(bootstrap_replicates, dtype=float)
        shuffled = density.copy()
        for draw in range(bootstrap_replicates):
            shuffled[active_rows] = rng.permutation(density[active_rows])
            null_branches, _ = _density_component_hierarchy(
                graph,
                active,
                shuffled,
                min_samples=min_samples,
                min_component_size=min_component_size,
                levels=levels,
                max_quantile=max_quantile,
            )
            null_scores[draw] = max(
                (branch_score(branch, shuffled) for branch in null_branches),
                default=0.0,
            )
        pvalue = (1.0 + float(np.sum(null_scores >= observed))) / (bootstrap_replicates + 1.0)
        null_mean = float(np.mean(null_scores))
    if observed > 0.0:
        selected = np.asarray(chosen["seeds"], dtype=bool)
    return (
        selected,
        float(pvalue),
        component_size,
        observed,
        bootstrap_replicates,
        null_mean,
    )


def gamma_strict_core_from_graph(
    graph: KNNGraph,
    *,
    ambient_dimension: float,
    config: GammaStrictCoreConfig | None = None,
    stratification: StratificationResult | None = None,
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
    if cfg.tail_seed_min_size < 1:
        raise ValueError("tail_seed_min_size must be positive")
    if cfg.tail_probe_method not in {
        "rank_window",
        "ranked_shell_mhg",
        "component_percolation",
        "component_multiscale",
        "component_adaptive_multiscale",
        "persistent_contrast",
        "persistent_dbcv",
        "persistent_dbcv_bootstrap",
        "gamma_llr_bootstrap",
    }:
        raise ValueError(
            "tail_probe_method must be 'rank_window', 'ranked_shell_mhg', "
            "'component_percolation', 'component_multiscale', "
            "'component_adaptive_multiscale', 'persistent_contrast', "
            "'persistent_dbcv', 'persistent_dbcv_bootstrap', or "
            "'gamma_llr_bootstrap'"
        )
    if cfg.tail_percolation_permutations < 19:
        raise ValueError("tail_percolation_permutations must be at least 19")
    if cfg.tail_percolation_density_bins < 1 or cfg.tail_percolation_degree_bins < 1:
        raise ValueError("tail percolation bin counts must be positive")
    if cfg.tail_persistence_levels < 3:
        raise ValueError("tail_persistence_levels must be at least 3")
    if not 0.0 < cfg.tail_persistence_max_quantile < 1.0:
        raise ValueError("tail_persistence_max_quantile must lie in (0, 1)")
    if cfg.tail_llr_bootstrap_replicates < 0:
        raise ValueError("tail_llr_bootstrap_replicates must be nonnegative")
    if cfg.min_core_seeds_per_stratum < 0:
        raise ValueError("min_core_seeds_per_stratum must be nonnegative")
    if not cfg.min_samples - 1 < cfg.tail_probe_outer_rank <= graph.k:
        raise ValueError("tail_probe_outer_rank must exceed the core rank and fit the graph")

    strat = stratification
    if strat is None:
        strat = estimate_stratification_uniform_tail(
            graph,
            eps_rank=cfg.min_samples - 1,
            ambient_dimension=ambient_dimension,
            config=cfg.uniform_tail,
        )
    elif strat.groups.shape != (graph.n_samples,):
        raise ValueError("precomputed stratification must align with the graph")
    groups = np.asarray(strat.groups, dtype=np.int32)
    dense_groups = set(map(int, strat.supported_groups.tolist()))
    eps = np.zeros(strat.selected_components, dtype=np.float32)
    point_eps = np.zeros(graph.n_samples, dtype=np.float32)
    core_seed_mask = np.ones(graph.n_samples, dtype=bool)
    tail_probe_pvalues: list[float] = []
    tail_probe_component_sizes: list[int] = []
    tail_probe_null_windows: list[int] = []
    tail_probe_raw_pvalues: list[float] = []
    tail_probe_best_cutoffs: list[int] = []
    tail_probe_overlaps: list[int] = []
    tail_percolation_null_q95: list[float] = []
    tail_percolation_null_means: list[float] = []
    tail_probe_target_ranks: list[int] = []
    tail_probe_component_pvalues: list[list[float]] = []
    tail_probe_component_qvalues: list[list[float]] = []
    tail_probe_component_target_ranks: list[list[int]] = []
    activated_tail_groups: list[int] = []
    effective_dense_quantiles = np.full(strat.selected_components, np.nan, dtype=float)
    stratum_sizes = np.zeros(strat.selected_components, dtype=np.int64)
    kth = graph.distances[:, cfg.min_samples - 2]

    for group in range(strat.selected_components):
        mask = groups == group
        values = kth[mask]
        stratum_sizes[group] = values.size
        if values.size == 0:
            continue
        if group in dense_groups:
            effective_quantile = cfg.dense_core_quantile
            if cfg.min_core_seeds_per_stratum:
                effective_quantile = max(
                    effective_quantile,
                    min(1.0, cfg.min_core_seeds_per_stratum / values.size),
                )
            eps[group] = np.quantile(values, effective_quantile)
            if cfg.min_core_seeds_per_stratum:
                seed_index = min(cfg.min_core_seeds_per_stratum, values.size) - 1
                eps[group] = max(eps[group], np.partition(values, seed_index)[seed_index])
            effective_dense_quantiles[group] = effective_quantile
            point_eps[mask] = eps[group]
            continue
        contrast = kth / np.maximum(
            graph.distances[:, cfg.tail_probe_outer_rank - 1],
            np.finfo(np.float32).tiny,
        )
        values_threshold = float(np.quantile(values, cfg.tail_core_quantile))
        candidates = mask & (kth <= values_threshold)
        null_q95 = 0.0
        null_mean = 0.0
        if cfg.tail_probe_method == "ranked_shell_mhg":
            (
                selected_component,
                pvalue,
                raw_pvalue,
                component_size,
                overlap,
                best_cutoff,
                completed,
            ) = _tail_ranked_shell_enrichment(
                graph,
                mask,
                candidates,
                ambient_dimension=ambient_dimension,
                min_samples=cfg.min_samples,
                outer_rank=cfg.tail_probe_outer_rank,
                min_component_size=cfg.tail_seed_min_size,
                select_multiple=cfg.tail_multiple_components,
                alpha=cfg.tail_probe_alpha,
            )
        elif cfg.tail_probe_method == "component_percolation":
            (
                selected_component,
                pvalue,
                component_size,
                null_q95,
                completed,
                null_mean,
            ) = _tail_component_percolation_test(
                graph,
                mask,
                candidates,
                min_samples=cfg.min_samples,
                min_component_size=cfg.tail_seed_min_size,
                select_multiple=cfg.tail_multiple_components,
                alpha=cfg.tail_probe_alpha,
                permutations=cfg.tail_percolation_permutations,
                density_bins=cfg.tail_percolation_density_bins,
                degree_bins=cfg.tail_percolation_degree_bins,
                random_state=cfg.tail_percolation_random_state,
            )
            raw_pvalue = pvalue
            overlap = component_size
            best_cutoff = int(null_q95)
        elif cfg.tail_probe_method == "component_multiscale":
            (
                selected_component,
                pvalue,
                component_size,
                observed_score,
                completed,
                null_mean,
            ) = _tail_component_multiscale_test(
                graph,
                mask,
                candidates,
                min_samples=cfg.min_samples,
                min_component_size=cfg.tail_seed_min_size,
                select_multiple=cfg.tail_multiple_components,
                alpha=cfg.tail_probe_alpha,
                permutations=cfg.tail_percolation_permutations,
                density_bins=cfg.tail_percolation_density_bins,
                degree_bins=cfg.tail_percolation_degree_bins,
                random_state=cfg.tail_percolation_random_state + group,
            )
            raw_pvalue = pvalue
            overlap = component_size
            null_q95 = observed_score
            selected_values = kth[selected_component]
            best_cutoff = component_size
            values_threshold = (
                float(np.max(selected_values)) if selected_values.size else values_threshold
            )
            tail_probe_target_ranks.append(int(graph.k))
        elif cfg.tail_probe_method == "component_adaptive_multiscale":
            (
                selected_component,
                pvalue,
                component_size,
                observed_score,
                completed,
                null_mean,
                target_rank,
                component_pvalues,
                component_qvalues,
                component_target_ranks,
            ) = _tail_component_adaptive_multiscale_test(
                graph,
                mask,
                candidates,
                min_samples=cfg.min_samples,
                min_component_size=cfg.tail_seed_min_size,
                select_multiple=cfg.tail_multiple_components,
                alpha=cfg.tail_probe_alpha,
                permutations=cfg.tail_percolation_permutations,
                density_bins=cfg.tail_percolation_density_bins,
                degree_bins=cfg.tail_percolation_degree_bins,
                random_state=cfg.tail_percolation_random_state + group,
            )
            raw_pvalue = pvalue
            overlap = component_size
            null_q95 = observed_score
            selected_values = kth[selected_component]
            best_cutoff = component_size
            values_threshold = (
                float(np.max(selected_values)) if selected_values.size else values_threshold
            )
            tail_probe_target_ranks.append(target_rank)
            tail_probe_component_pvalues.append(component_pvalues)
            tail_probe_component_qvalues.append(component_qvalues)
            tail_probe_component_target_ranks.append(component_target_ranks)
        elif cfg.tail_probe_method in {
            "persistent_contrast",
            "persistent_dbcv",
            "persistent_dbcv_bootstrap",
            "gamma_llr_bootstrap",
        }:
            (
                selected_component,
                pvalue,
                component_size,
                observed_score,
                completed,
                null_mean,
            ) = _tail_persistent_selector(
                graph,
                mask,
                ambient_dimension=ambient_dimension,
                min_samples=cfg.min_samples,
                min_component_size=cfg.tail_seed_min_size,
                levels=cfg.tail_persistence_levels,
                max_quantile=cfg.tail_persistence_max_quantile,
                method=cfg.tail_probe_method,
                bootstrap_replicates=(
                    cfg.tail_llr_bootstrap_replicates
                    if cfg.tail_probe_method in {
                        "gamma_llr_bootstrap",
                        "persistent_dbcv_bootstrap",
                    }
                    else 0
                ),
                random_state=cfg.tail_llr_random_state + group,
            )
            raw_pvalue = pvalue
            overlap = component_size
            null_q95 = observed_score
            selected_values = kth[selected_component]
            best_cutoff = component_size
            values_threshold = (
                float(np.max(selected_values)) if selected_values.size else values_threshold
            )
        else:
            pvalue, component_size, _, completed = _tail_rank_window_pvalue(
                graph,
                mask,
                contrast,
                quantile=cfg.tail_probe_quantile,
                min_samples=cfg.min_samples,
            )
            raw_pvalue = pvalue
            overlap = component_size
            best_cutoff = max(cfg.min_samples, int(np.ceil(cfg.tail_probe_quantile * values.size)))
            selected_component = np.zeros(graph.n_samples, dtype=bool)
        tail_probe_pvalues.append(pvalue)
        tail_probe_raw_pvalues.append(raw_pvalue)
        tail_probe_component_sizes.append(component_size)
        tail_probe_null_windows.append(completed)
        tail_probe_best_cutoffs.append(best_cutoff)
        tail_probe_overlaps.append(overlap)
        tail_percolation_null_q95.append(null_q95)
        tail_percolation_null_means.append(null_mean)
        if pvalue <= cfg.tail_probe_alpha:
            eps[group] = values_threshold
            point_eps[mask] = eps[group]
            if cfg.tail_probe_method == "rank_window":
                probe_threshold = float(np.quantile(contrast[mask], cfg.tail_probe_quantile))
                probe = mask & (contrast <= probe_threshold)
                selected_component = _candidate_component_seed_mask(
                    graph,
                    candidates,
                    probe,
                    min_samples=cfg.min_samples,
                    select_multiple=cfg.tail_multiple_components,
                    min_component_size=cfg.tail_seed_min_size,
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
            "strict_core_min_core_seeds_per_stratum": cfg.min_core_seeds_per_stratum,
            "strict_core_effective_dense_quantiles": effective_dense_quantiles.tolist(),
            "strict_core_stratum_sizes": stratum_sizes.tolist(),
            "strict_core_tail_probe_quantile": cfg.tail_probe_quantile,
            "strict_core_tail_probe_outer_rank": cfg.tail_probe_outer_rank,
            "strict_core_tail_probe_alpha": cfg.tail_probe_alpha,
            "strict_core_tail_probe_method": cfg.tail_probe_method,
            "strict_core_tail_quantile": cfg.tail_core_quantile,
            "strict_core_tail_multiple_components": cfg.tail_multiple_components,
            "strict_core_tail_seed_min_size": cfg.tail_seed_min_size,
            "strict_core_border_multiplier": cfg.border_multiplier,
            "strict_core_dense_groups": len(dense_groups),
            "strict_core_tail_probe_pvalues": tail_probe_pvalues,
            "strict_core_tail_probe_raw_pvalues": tail_probe_raw_pvalues,
            "strict_core_tail_probe_component_sizes": tail_probe_component_sizes,
            "strict_core_tail_probe_null_windows": tail_probe_null_windows,
            "strict_core_tail_probe_best_cutoffs": tail_probe_best_cutoffs,
            "strict_core_tail_probe_overlaps": tail_probe_overlaps,
            "strict_core_tail_probe_target_ranks": tail_probe_target_ranks,
            "strict_core_tail_probe_component_pvalues": tail_probe_component_pvalues,
            "strict_core_tail_probe_component_qvalues": tail_probe_component_qvalues,
            "strict_core_tail_probe_component_target_ranks": (
                tail_probe_component_target_ranks
            ),
            "strict_core_tail_percolation_permutations": cfg.tail_percolation_permutations,
            "strict_core_tail_percolation_density_bins": cfg.tail_percolation_density_bins,
            "strict_core_tail_percolation_degree_bins": cfg.tail_percolation_degree_bins,
            "strict_core_tail_percolation_null_q95": tail_percolation_null_q95,
            "strict_core_tail_percolation_null_means": tail_percolation_null_means,
            "strict_core_tail_persistence_levels": cfg.tail_persistence_levels,
            "strict_core_tail_persistence_max_quantile": cfg.tail_persistence_max_quantile,
            "strict_core_tail_llr_bootstrap_replicates": cfg.tail_llr_bootstrap_replicates,
            "strict_core_activated_tail_groups": activated_tail_groups,
            "strict_core_core_fraction": float(np.mean(core)),
            "strict_core_active_fraction": float(np.mean(labels >= 0)),
            "strict_core_clusters": int(np.unique(labels[labels >= 0]).size),
            "strict_core_stratification_mode": strat.mode,
            "strict_core_stratification_components": int(strat.selected_components),
            "strict_core_stratification_seconds": float(sum(strat.timings.values())),
            "strict_core_gamma_components": int(
                strat.diagnostics.get("uniform_tail_components", 0)
            ),
            "strict_core_gamma_background_fraction": float(
                strat.diagnostics.get("uniform_tail_background_fraction", 0.0)
            ),
            **strat.diagnostics,
            **clustering_profile,
        },
    )
