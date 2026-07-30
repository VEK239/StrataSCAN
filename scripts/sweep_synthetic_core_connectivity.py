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
from stratascan import build_knn_graph
from stratascan.multiscale import _shell_volumes
from stratascan.predictive import _responsibilities
from stratascan.strict_core import (
    _candidate_component_seed_mask,
    _tail_rank_window_pvalue,
)


QUANTILES = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 0.95)
CONNECTIVITY_RANKS = (4, 8, 16, 32)
TAIL_MODES = ("off", "current", "all")


def decode(value):
    return json.loads(value) if isinstance(value, str) else value


def reconstruct_stratification(graph, dimension: int, source: pd.Series):
    shells, _ = _shell_volumes(graph, (4, 8, 16, 32), float(dimension))
    weights = np.asarray(decode(source["weights"]), dtype=float)
    rates = np.asarray(decode(source["rates"]), dtype=float)
    responsibilities, _ = _responsibilities(
        shells,
        shapes=np.array([4.0, 4.0, 8.0, 16.0]),
        weights=weights,
        rates=rates,
    )
    head = int(source["head_strata"])
    background_probability = np.sum(responsibilities[:, head:], axis=1)
    dense_group = np.argmax(responsibilities[:, :head], axis=1).astype(np.int32)
    groups = np.where(background_probability < 0.5, dense_group, head).astype(np.int32)
    return groups, head


def prepare_thresholds(graph, groups: np.ndarray, head: int, quantile: float, mode: str):
    d4 = np.asarray(graph.distances[:, 3], dtype=float)
    point_eps = np.zeros(graph.n_samples, dtype=np.float32)
    seed_mask = np.ones(graph.n_samples, dtype=bool)
    tail_pvalue = np.nan
    tail_activated = False

    for group in range(head):
        mask = groups == group
        if np.any(mask):
            point_eps[mask] = float(np.quantile(d4[mask], quantile))

    tail = groups == head
    if not np.any(tail) or mode == "off":
        return point_eps, seed_mask, tail_pvalue, tail_activated

    if mode == "all":
        point_eps[tail] = float(np.quantile(d4[tail], quantile))
        return point_eps, seed_mask, tail_pvalue, True

    d32 = np.maximum(
        np.asarray(graph.distances[:, 31], dtype=float), np.finfo(float).tiny
    )
    contrast = d4 / d32
    tail_threshold = float(np.quantile(d4[tail], 0.05))
    candidates = tail & (d4 <= tail_threshold)
    tail_pvalue, _, _, _ = _tail_rank_window_pvalue(
        graph,
        tail,
        contrast,
        quantile=0.02,
        min_samples=5,
    )
    if tail_pvalue <= 0.05:
        point_eps[tail] = tail_threshold
        probe_threshold = float(np.quantile(contrast[tail], 0.02))
        probe = tail & (contrast <= probe_threshold)
        selected = _candidate_component_seed_mask(
            graph,
            candidates,
            probe,
            min_samples=5,
            select_multiple=False,
            min_component_size=5,
        )
        seed_mask[tail] = selected[tail]
        tail_activated = True
    return point_eps, seed_mask, tail_pvalue, tail_activated


def cluster_with_connectivity_rank(
    graph,
    point_eps: np.ndarray,
    seed_mask: np.ndarray,
    connectivity_rank: int,
    *,
    border_multiplier: float = 1.25,
):
    active = point_eps > 0.0
    d4 = np.asarray(graph.distances[:, 3], dtype=float)
    core = active & (d4 <= point_eps) & seed_mask
    core_indices = np.flatnonzero(core).astype(np.int32)
    labels = np.full(graph.n_samples, -1, dtype=np.int64)
    neighbours = np.asarray(graph.indices[:, :connectivity_rank], dtype=np.int32)
    distances = np.asarray(graph.distances[:, :connectivity_rank], dtype=float)

    if core_indices.size:
        source = np.repeat(core_indices, connectivity_rank)
        target = neighbours[core_indices].reshape(-1)
        edge_distance = distances[core_indices].reshape(-1)
        # A directed variable-epsilon neighbourhood, symmetrised only when
        # connected components are computed, matching the existing rank-4 rule.
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
        remap = np.full(sizes.size, -1, dtype=np.int64)
        valid = sizes >= 5
        remap[np.flatnonzero(valid)] = np.arange(np.sum(valid), dtype=np.int64)
        labels[core_indices] = remap[components]

    border = np.flatnonzero(active & ~core).astype(np.int32)
    if border.size:
        border_neighbours = neighbours[border]
        eligible = (
            core[border_neighbours]
            & (labels[border_neighbours] >= 0)
            & (
                distances[border]
                <= point_eps[border_neighbours] * float(border_multiplier)
            )
        )
        has_match = np.any(eligible, axis=1)
        first = np.argmax(eligible, axis=1)
        chosen = border_neighbours[np.arange(border.size), first]
        labels[border[has_match]] = labels[chosen[has_match]]
    return labels, core


def run_dataset(case, seed: int, n: int, source: pd.Series):
    dataset = load_synthetic({**case, "n": n}, seed)
    dimension = int(dataset.X.shape[1])
    backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
    graph, graph_seconds = build_knn_graph(
        dataset.X, k=32, backend=backend, n_jobs=1
    )
    groups, head = reconstruct_stratification(graph, dimension, source)
    rows = []
    threshold_cache = {}
    for quantile in QUANTILES:
        for mode in TAIL_MODES:
            threshold_cache[(quantile, mode)] = prepare_thresholds(
                graph, groups, head, quantile, mode
            )
        for connectivity_rank in CONNECTIVITY_RANKS:
            for mode in TAIL_MODES:
                point_eps, seed_mask, tail_pvalue, tail_activated = threshold_cache[
                    (quantile, mode)
                ]
                labels, core = cluster_with_connectivity_rank(
                    graph,
                    point_eps,
                    seed_mask,
                    connectivity_rank,
                )
                metrics = evaluate(dataset, labels, "synthetic")
                rows.append(
                    {
                        "dataset_id": case["id"],
                        "seed": seed,
                        "n": n,
                        "dimension": dimension,
                        "core_quantile": quantile,
                        "connectivity_rank": connectivity_rank,
                        "tail_mode": mode,
                        "head_strata": head,
                        "tail_pvalue": tail_pvalue,
                        "tail_activated": tail_activated,
                        "graph_seconds": graph_seconds,
                        "core_fraction": float(np.mean(core)),
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


def aggregate_results(rows: pd.DataFrame, output: Path):
    keys = ["dataset_id", "core_quantile", "connectivity_rank", "tail_mode"]
    metrics = [
        "macro_target_f1",
        "pairwise_f1",
        "binary_macro_f1",
        "background_rejection",
        "signal_coverage",
        "n_clusters",
        "core_fraction",
        "spurious_background_majority_clusters",
        "merged_predicted_clusters",
        "extra_signal_fragments",
    ]
    aggregate = rows.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(output / "aggregate-grid.csv", index=False)

    global_summary = (
        aggregate.groupby(["core_quantile", "connectivity_rank", "tail_mode"], sort=False)
        .agg(
            mean_dataset_macro_f1=("macro_target_f1_mean", "mean"),
            worst_dataset_macro_f1=("macro_target_f1_mean", "min"),
            mean_background_rejection=("background_rejection_mean", "mean"),
            mean_signal_coverage=("signal_coverage_mean", "mean"),
            mean_extra_fragments=("extra_signal_fragments_mean", "mean"),
            mean_merged_clusters=("merged_predicted_clusters_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_dataset_macro_f1", ascending=False)
    )
    global_summary.to_csv(output / "global-grid-ranking.csv", index=False)
    best_by_dataset = aggregate.loc[
        aggregate.groupby("dataset_id")["macro_target_f1_mean"].idxmax()
    ].sort_values("dataset_id")
    best_by_dataset.to_csv(output / "best-by-dataset.csv", index=False)
    return aggregate, global_summary, best_by_dataset


def plot_heatmaps(aggregate: pd.DataFrame, output: Path):
    datasets = aggregate["dataset_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(datasets), len(TAIL_MODES),
        figsize=(12.0, 2.55 * len(datasets)),
        constrained_layout=True,
    )
    for row_index, dataset_id in enumerate(datasets):
        for column_index, mode in enumerate(TAIL_MODES):
            axis = axes[row_index, column_index]
            subset = aggregate.loc[
                (aggregate["dataset_id"] == dataset_id)
                & (aggregate["tail_mode"] == mode)
            ]
            table = subset.pivot(
                index="core_quantile",
                columns="connectivity_rank",
                values="macro_target_f1_mean",
            ).reindex(index=QUANTILES, columns=CONNECTIVITY_RANKS)
            image = axis.imshow(table.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
            for y in range(len(QUANTILES)):
                for x in range(len(CONNECTIVITY_RANKS)):
                    value = table.iloc[y, x]
                    axis.text(
                        x, y, f"{value:.2f}", ha="center", va="center",
                        color="white" if value < 0.55 else "black", fontsize=7,
                    )
            axis.set_xticks(range(len(CONNECTIVITY_RANKS)), CONNECTIVITY_RANKS)
            axis.set_yticks(range(len(QUANTILES)), QUANTILES)
            axis.set_xlabel("connectivity rank")
            axis.set_ylabel(
                f"{dataset_id}\ncore quantile" if column_index == 0 else ""
            )
            if row_index == 0:
                axis.set_title(f"tail: {mode}")
    fig.colorbar(image, ax=axes, shrink=0.35, label="mean macro target F1")
    fig.savefig(output / "core-connectivity-grid.png", dpi=180)
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
        default=REPO / "results" / "core_connectivity_grid_10k_20260727",
    )
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    predictive = pd.read_csv(args.predictive_metrics)
    checkpoint = args.output / "per-seed-grid.csv"
    existing = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    completed = (
        set(zip(existing["dataset_id"], existing["seed"], strict=True))
        if len(existing)
        else set()
    )
    all_rows = existing.to_dict("records")
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
            rows = run_dataset(case, seed, args.n, source.iloc[0])
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(checkpoint, index=False)
            print(f"completed {case['id']} seed={seed}", flush=True)
    frame = pd.DataFrame(all_rows)
    aggregate, _, _ = aggregate_results(frame, args.output)
    plot_heatmaps(aggregate, args.output)
    print(f"finished {len(frame)} runs in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
