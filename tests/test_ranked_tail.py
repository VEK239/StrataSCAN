import numpy as np

from stratascan.experimental import StableDBCVTailStrataSCAN
from stratascan.strict_core import (
    _tail_component_adaptive_multiscale_test,
    _tail_component_multiscale_test,
    _tail_component_percolation_test,
    _tail_persistent_selector,
    _tail_ranked_shell_enrichment,
)
from stratascan.types import KNNGraph


def _separated_candidate_graph() -> tuple[KNNGraph, np.ndarray]:
    n, k = 100, 32
    indices = np.empty((n, k), dtype=np.int32)
    distances = np.empty((n, k), dtype=np.float32)
    candidate = np.zeros(n, dtype=bool)
    candidate[:10] = True
    for row in range(n):
        if row < 10:
            near = [value for value in range(10) if value != row]
            far = [value for value in range(10, n)]
            indices[row] = np.asarray((near + far)[:k], dtype=np.int32)
            distances[row, :4] = np.linspace(0.07, 0.10, 4)
            distances[row, 4:] = np.linspace(0.11, 10.0, k - 4)
        else:
            others = [value for value in range(n) if value != row]
            indices[row] = np.asarray(others[:k], dtype=np.int32)
            distances[row, :4] = np.linspace(0.07, 0.10, 4)
            distances[row, 4:] = np.linspace(5.0, 10.0, k - 4)
    return KNNGraph(indices, distances), candidate


def test_ranked_shell_enrichment_finds_component_without_fixed_fraction() -> None:
    graph, candidate = _separated_candidate_graph()
    selected, adjusted, raw, size, overlap, cutoff, tests = _tail_ranked_shell_enrichment(
        graph,
        np.ones(graph.n_samples, dtype=bool),
        candidate,
        ambient_dimension=2.0,
        min_samples=5,
        outer_rank=32,
        min_component_size=5,
        select_multiple=False,
        alpha=0.05,
    )

    assert np.array_equal(np.flatnonzero(selected), np.arange(10))
    assert adjusted < 0.05
    assert raw <= adjusted
    assert (size, overlap, cutoff, tests) == (10, 10, 10, 6)


def test_ranked_shell_enrichment_returns_no_selection_without_candidates() -> None:
    graph, _ = _separated_candidate_graph()
    result = _tail_ranked_shell_enrichment(
        graph,
        np.ones(graph.n_samples, dtype=bool),
        np.zeros(graph.n_samples, dtype=bool),
        ambient_dimension=2.0,
        min_samples=5,
        outer_rank=32,
        min_component_size=5,
        select_multiple=False,
        alpha=0.05,
    )

    selected, adjusted, raw, size, overlap, cutoff, tests = result
    assert not np.any(selected)
    assert (adjusted, raw, size, overlap, cutoff, tests) == (1.0, 1.0, 0, 0, 0, 0)


def test_component_percolation_detects_excess_connectivity_on_fixed_graph() -> None:
    n, k = 200, 32
    offsets: list[int] = []
    for step in range(1, 17):
        offsets.extend((step, -step))
    indices = np.asarray(
        [[(row + offset) % n for offset in offsets] for row in range(n)],
        dtype=np.int32,
    )
    distances = np.tile(np.arange(1, k + 1, dtype=np.float32), (n, 1))
    graph = KNNGraph(indices, distances)
    candidates = np.zeros(n, dtype=bool)
    candidates[:20] = True

    selected, pvalue, observed, null_q95, draws, null_mean = (
        _tail_component_percolation_test(
            graph,
            np.ones(n, dtype=bool),
            candidates,
            min_samples=5,
            min_component_size=5,
            select_multiple=False,
            alpha=0.20,
            permutations=199,
            density_bins=1,
            degree_bins=1,
            random_state=42,
        )
    )

    assert np.array_equal(np.flatnonzero(selected), np.arange(20))
    assert pvalue == 1.0 / 200.0
    assert observed == 20
    assert null_q95 < observed
    assert draws == 199
    assert null_mean < observed


def test_component_multiscale_detects_persistent_outer_density() -> None:
    graph, candidates = _separated_candidate_graph()
    selected, pvalue, size, score, draws, null_mean = _tail_component_multiscale_test(
        graph,
        np.ones(graph.n_samples, dtype=bool),
        candidates,
        min_samples=5,
        min_component_size=5,
        select_multiple=False,
        alpha=0.05,
        permutations=199,
        density_bins=1,
        degree_bins=1,
        random_state=42,
    )

    assert np.array_equal(np.flatnonzero(selected), np.arange(10))
    assert pvalue < 0.05
    assert size == 10
    assert score > null_mean
    assert draws == 199


def test_adaptive_multiscale_selects_and_calibrates_outer_rank() -> None:
    graph, candidates = _separated_candidate_graph()
    (
        selected,
        pvalue,
        size,
        score,
        draws,
        null_mean,
        target_rank,
        component_pvalues,
        component_qvalues,
        component_target_ranks,
    ) = (
        _tail_component_adaptive_multiscale_test(
            graph,
            np.ones(graph.n_samples, dtype=bool),
            candidates,
            min_samples=5,
            min_component_size=5,
            select_multiple=False,
            alpha=0.05,
            permutations=199,
            density_bins=1,
            degree_bins=1,
            random_state=42,
        )
    )

    assert np.array_equal(np.flatnonzero(selected), np.arange(10))
    assert pvalue < 0.05
    assert size == 10
    assert score > null_mean
    assert draws == 199
    assert target_rank in {8, 16, 32}
    assert len(component_pvalues) == len(component_qvalues) == 1
    assert component_qvalues[0] <= 0.05
    assert component_target_ranks == [target_rank]


def test_persistent_local_density_selectors_find_separated_dense_component() -> None:
    n, k = 120, 32
    indices = np.empty((n, k), dtype=np.int32)
    distances = np.empty((n, k), dtype=np.float32)
    for row in range(n):
        if row < 20:
            near = [value for value in range(20) if value != row]
            far = list(range(20, n))
            indices[row] = np.asarray((near + far)[:k], dtype=np.int32)
            distances[row, :19] = np.linspace(0.05, 0.20, 19)
            distances[row, 19:] = np.linspace(1.0, 2.0, k - 19)
        else:
            others = [value for value in range(20, n) if value != row]
            indices[row] = np.asarray(others[:k], dtype=np.int32)
            distances[row] = np.linspace(0.8, 2.0, k)
    graph = KNNGraph(indices, distances)
    active = np.ones(n, dtype=bool)

    for method in ("persistent_contrast", "persistent_dbcv"):
        selected, pvalue, size, score, draws, _ = _tail_persistent_selector(
            graph,
            active,
            ambient_dimension=2.0,
            min_samples=5,
            min_component_size=5,
            levels=12,
            max_quantile=0.30,
            method=method,
        )
        assert pvalue == 0.0
        assert score > 0.0
        assert size >= 5
        assert np.all(np.flatnonzero(selected) < 20)
        assert draws == 0


def test_stable_dbcv_cluster_matching_uses_partition_overlap() -> None:
    baseline = np.array([-1, -1, -1, 0, 0, 0], dtype=np.int32)
    candidate = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)
    novel = StableDBCVTailStrataSCAN._novel_clusters(candidate, baseline)

    assert len(novel) == 1
    assert np.array_equal(np.flatnonzero(novel[0]), np.arange(3))
    assert StableDBCVTailStrataSCAN._best_jaccard(
        novel[0], np.array([4, 4, -1, 2, 2, 2], dtype=np.int32)
    ) == 2.0 / 3.0
