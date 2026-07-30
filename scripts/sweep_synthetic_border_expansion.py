from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from scripts.sweep_synthetic_core_connectivity import (
    QUANTILES,
    prepare_thresholds,
    reconstruct_stratification,
)
from stratascan import build_knn_graph


BORDER_RANKS = (4, 8, 16, 32)
EXPANSION_MODES = (
    "current_border_to_core_1p25",
    "core_to_border_1p25",
    "dbscan_core_to_border",
    "iterative_border_flood",
)
CORE_CONNECTIVITY_RANK = 8


def core_components(graph, point_eps: np.ndarray, seed_mask: np.ndarray):
    d4 = np.asarray(graph.distances[:, 3], dtype=float)
    active = point_eps > 0.0
    core = active & (d4 <= point_eps) & seed_mask
    core_indices = np.flatnonzero(core).astype(np.int32)
    labels = np.full(graph.n_samples, -1, dtype=np.int64)
    if not core_indices.size:
        return labels, core

    neighbours = np.asarray(
        graph.indices[core_indices, :CORE_CONNECTIVITY_RANK], dtype=np.int32
    )
    distances = np.asarray(
        graph.distances[core_indices, :CORE_CONNECTIVITY_RANK], dtype=float
    )
    source = np.repeat(core_indices, CORE_CONNECTIVITY_RANK)
    target = neighbours.reshape(-1)
    edge_distance = distances.reshape(-1)
    keep = core[target] & (edge_distance <= point_eps[source])
    local = np.full(graph.n_samples, -1, dtype=np.int32)
    local[core_indices] = np.arange(core_indices.size, dtype=np.int32)
    adjacency = coo_matrix(
        (
            np.ones(int(np.sum(keep)), dtype=np.uint8),
            (local[source[keep]], local[target[keep]]),
        ),
        shape=(core_indices.size, core_indices.size),
    ).tocsr()
    _, components = connected_components(adjacency, directed=False, return_labels=True)
    sizes = np.bincount(components)
    valid = sizes >= 5
    remap = np.full(sizes.size, -1, dtype=np.int64)
    remap[np.flatnonzero(valid)] = np.arange(np.sum(valid), dtype=np.int64)
    labels[core_indices] = remap[components]
    return labels, core


def nearest_assignments(target, source, distance):
    if not target.size:
        return target, source
    order = np.lexsort((distance, target))
    sorted_target = target[order]
    first = np.r_[True, sorted_target[1:] != sorted_target[:-1]]
    chosen = order[first]
    return target[chosen], source[chosen]


def expand_current(
    graph,
    labels: np.ndarray,
    core: np.ndarray,
    active: np.ndarray,
    point_eps: np.ndarray,
    border_rank: int,
):
    border = np.flatnonzero(active & ~core).astype(np.int32)
    if not border.size:
        return labels, 0
    neighbours = np.asarray(graph.indices[border, :border_rank], dtype=np.int32)
    distances = np.asarray(graph.distances[border, :border_rank], dtype=float)
    eligible = (
        core[neighbours]
        & (labels[neighbours] >= 0)
        & (distances <= point_eps[neighbours] * 1.25)
    )
    has_match = np.any(eligible, axis=1)
    first = np.argmax(eligible, axis=1)
    chosen = neighbours[np.arange(border.size), first]
    labels[border[has_match]] = labels[chosen[has_match]]
    return labels, 1


def expand_core_to_border(
    graph,
    labels: np.ndarray,
    core: np.ndarray,
    active: np.ndarray,
    point_eps: np.ndarray,
    border_rank: int,
    multiplier: float,
):
    sources = np.flatnonzero(core & (labels >= 0)).astype(np.int32)
    if not sources.size:
        return labels, 0
    neighbours = np.asarray(graph.indices[sources, :border_rank], dtype=np.int32)
    distances = np.asarray(graph.distances[sources, :border_rank], dtype=float)
    source = np.repeat(sources, border_rank)
    target = neighbours.reshape(-1)
    edge_distance = distances.reshape(-1)
    keep = (
        active[target]
        & ~core[target]
        & (edge_distance <= point_eps[source] * multiplier)
    )
    target, source = nearest_assignments(
        target[keep], source[keep], edge_distance[keep]
    )
    labels[target] = labels[source]
    return labels, 1


def expand_iteratively(
    graph,
    labels: np.ndarray,
    core: np.ndarray,
    active: np.ndarray,
    point_eps: np.ndarray,
    border_rank: int,
):
    del core
    iterations = 0
    while True:
        sources = np.flatnonzero(active & (labels >= 0)).astype(np.int32)
        if not sources.size:
            break
        neighbours = np.asarray(graph.indices[sources, :border_rank], dtype=np.int32)
        distances = np.asarray(graph.distances[sources, :border_rank], dtype=float)
        source = np.repeat(sources, border_rank)
        target = neighbours.reshape(-1)
        edge_distance = distances.reshape(-1)
        keep = (
            active[target]
            & (labels[target] < 0)
            & (edge_distance <= point_eps[source])
        )
        target, source = nearest_assignments(
            target[keep], source[keep], edge_distance[keep]
        )
        if not target.size:
            break
        labels[target] = labels[source]
        iterations += 1
        if iterations >= graph.n_samples:
            raise RuntimeError("border flood failed to converge")
    return labels, iterations


def expand(
    graph,
    labels,
    core,
    active,
    point_eps,
    border_rank,
    mode,
):
    labels = labels.copy()
    if mode == "current_border_to_core_1p25":
        return expand_current(
            graph, labels, core, active, point_eps, border_rank
        )
    if mode == "core_to_border_1p25":
        return expand_core_to_border(
            graph, labels, core, active, point_eps, border_rank, 1.25
        )
    if mode == "dbscan_core_to_border":
        return expand_core_to_border(
            graph, labels, core, active, point_eps, border_rank, 1.0
        )
    if mode == "iterative_border_flood":
        return expand_iteratively(
            graph, labels, core, active, point_eps, border_rank
        )
    raise ValueError(f"unknown expansion mode: {mode}")


def run_dataset(case, seed: int, n: int, source: pd.Series):
    dataset = load_synthetic({**case, "n": n}, seed)
    dimension = int(dataset.X.shape[1])
    backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
    graph, graph_seconds = build_knn_graph(
        dataset.X, k=32, backend=backend, n_jobs=1
    )
    groups, head = reconstruct_stratification(graph, dimension, source)
    rows = []
    for quantile in QUANTILES:
        point_eps, seed_mask, tail_pvalue, tail_activated = prepare_thresholds(
            graph, groups, head, quantile, "current"
        )
        core_labels, core = core_components(graph, point_eps, seed_mask)
        active = point_eps > 0.0
        for border_rank in BORDER_RANKS:
            for mode in EXPANSION_MODES:
                labels, iterations = expand(
                    graph,
                    core_labels,
                    core,
                    active,
                    point_eps,
                    border_rank,
                    mode,
                )
                metrics = evaluate(dataset, labels, "synthetic")
                rows.append(
                    {
                        "dataset_id": case["id"],
                        "seed": seed,
                        "core_quantile": quantile,
                        "core_connectivity_rank": CORE_CONNECTIVITY_RANK,
                        "border_rank": border_rank,
                        "expansion_mode": mode,
                        "head_strata": head,
                        "tail_pvalue": tail_pvalue,
                        "tail_activated": tail_activated,
                        "expansion_iterations": iterations,
                        "graph_seconds": graph_seconds,
                        "core_fraction": float(np.mean(core)),
                        "assigned_border_fraction": float(
                            np.mean(active & ~core & (labels >= 0))
                        ),
                        **{
                            key: metrics[key]
                            for key in (
                                "macro_target_f1",
                                "pairwise_f1",
                                "binary_macro_f1",
                                "background_rejection",
                                "signal_coverage",
                                "n_clusters",
                                "noise_fraction",
                                "spurious_background_majority_clusters",
                                "merged_predicted_clusters",
                                "extra_signal_fragments",
                            )
                        },
                    }
                )
    return rows


def aggregate(frame: pd.DataFrame, output: Path):
    keys = ["dataset_id", "core_quantile", "border_rank", "expansion_mode"]
    metrics = [
        "macro_target_f1",
        "pairwise_f1",
        "binary_macro_f1",
        "background_rejection",
        "signal_coverage",
        "n_clusters",
        "core_fraction",
        "assigned_border_fraction",
        "expansion_iterations",
        "spurious_background_majority_clusters",
        "merged_predicted_clusters",
        "extra_signal_fragments",
    ]
    summary = frame.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "aggregate-border-grid.csv", index=False)
    global_ranking = (
        summary.groupby(["core_quantile", "border_rank", "expansion_mode"], sort=False)
        .agg(
            mean_dataset_macro_f1=("macro_target_f1_mean", "mean"),
            worst_dataset_macro_f1=("macro_target_f1_mean", "min"),
            mean_background_rejection=("background_rejection_mean", "mean"),
            mean_signal_coverage=("signal_coverage_mean", "mean"),
            mean_border_fraction=("assigned_border_fraction_mean", "mean"),
            mean_expansion_iterations=("expansion_iterations_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_dataset_macro_f1", ascending=False)
    )
    global_ranking.to_csv(output / "global-border-ranking.csv", index=False)
    return summary, global_ranking


def plot_grid(summary: pd.DataFrame, output: Path):
    datasets = summary["dataset_id"].drop_duplicates().tolist()
    short_names = {
        "current_border_to_core_1p25": "current b→c ×1.25",
        "core_to_border_1p25": "core→border ×1.25",
        "dbscan_core_to_border": "DBSCAN core→border",
        "iterative_border_flood": "iterative flood",
    }
    fig, axes = plt.subplots(
        len(datasets), len(EXPANSION_MODES),
        figsize=(14.5, 2.55 * len(datasets)),
        constrained_layout=True,
    )
    for row_index, dataset_id in enumerate(datasets):
        for column_index, mode in enumerate(EXPANSION_MODES):
            axis = axes[row_index, column_index]
            subset = summary.loc[
                (summary["dataset_id"] == dataset_id)
                & (summary["expansion_mode"] == mode)
            ]
            table = subset.pivot(
                index="core_quantile",
                columns="border_rank",
                values="macro_target_f1_mean",
            ).reindex(index=QUANTILES, columns=BORDER_RANKS)
            image = axis.imshow(table.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
            for y in range(len(QUANTILES)):
                for x in range(len(BORDER_RANKS)):
                    value = table.iloc[y, x]
                    axis.text(
                        x, y, f"{value:.2f}", ha="center", va="center",
                        color="white" if value < 0.55 else "black", fontsize=7,
                    )
            axis.set_xticks(range(len(BORDER_RANKS)), BORDER_RANKS)
            axis.set_yticks(range(len(QUANTILES)), QUANTILES)
            axis.set_xlabel("border rank")
            axis.set_ylabel(
                f"{dataset_id}\ncore quantile" if column_index == 0 else ""
            )
            if row_index == 0:
                axis.set_title(short_names[mode])
    fig.colorbar(image, ax=axes, shrink=0.35, label="mean macro target F1")
    fig.savefig(output / "border-expansion-grid.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "benchmarks" / "protocol.full-legacy-grid-7synthetic.json",
    )
    parser.add_argument(
        "--predictive-metrics",
        type=Path,
        default=REPO / "results" / "predictive_strata_10k_20260727" / "per-seed-metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "border_expansion_grid_10k_20260727",
    )
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    predictive = pd.read_csv(args.predictive_metrics)
    checkpoint = args.output / "per-seed-border-grid.csv"
    existing = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    completed = (
        set(zip(existing["dataset_id"], existing["seed"], strict=True))
        if len(existing)
        else set()
    )
    rows = existing.to_dict("records")
    started = perf_counter()
    for case in protocol["synthetic"]["cases"]:
        for seed in (23, 42, 73, 101, 151):
            if (case["id"], seed) in completed:
                continue
            source = predictive.loc[
                (predictive["dataset_id"] == case["id"])
                & (predictive["seed"] == seed)
            ]
            if len(source) != 1:
                raise ValueError(f"missing predictive fit for {case['id']} seed={seed}")
            rows.extend(run_dataset(case, seed, args.n, source.iloc[0]))
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"completed {case['id']} seed={seed}", flush=True)
    frame = pd.DataFrame(rows)
    summary, _ = aggregate(frame, args.output)
    plot_grid(summary, args.output)
    print(f"finished {len(frame)} runs in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
