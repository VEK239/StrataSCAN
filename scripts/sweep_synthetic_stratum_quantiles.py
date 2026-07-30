from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from scripts.sweep_synthetic_border_expansion import (
    core_components,
    expand_core_to_border,
)
from scripts.sweep_synthetic_core_connectivity import (
    prepare_thresholds,
    reconstruct_stratification,
)
from stratascan import build_knn_graph


DENSE_QUANTILES = (0.95, 0.97, 0.99, 0.995)
SPARSE_QUANTILES = (0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95)


def quantile_profile(head: int, dense_quantile: float, sparse_quantile: float):
    if head <= 1:
        return np.array([dense_quantile], dtype=float)
    return np.linspace(dense_quantile, sparse_quantile, head)


def scheduled_thresholds(
    graph,
    groups: np.ndarray,
    head: int,
    dense_quantile: float,
    sparse_quantile: float,
):
    # Reuse the current tail test unchanged; only dense-stratum eps changes.
    point_eps, seed_mask, tail_pvalue, tail_activated = prepare_thresholds(
        graph, groups, head, 0.95, "current"
    )
    d4 = np.asarray(graph.distances[:, 3], dtype=float)
    profile = quantile_profile(head, dense_quantile, sparse_quantile)
    eps_profile = np.zeros(head, dtype=float)
    for group, quantile in enumerate(profile):
        mask = groups == group
        if np.any(mask):
            eps_profile[group] = float(np.quantile(d4[mask], quantile))
            point_eps[mask] = eps_profile[group]
    return (
        point_eps,
        seed_mask,
        tail_pvalue,
        tail_activated,
        profile,
        eps_profile,
    )


def run_dataset(case, seed: int, n: int, source: pd.Series):
    dataset = load_synthetic({**case, "n": n}, seed)
    dimension = int(dataset.X.shape[1])
    backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
    graph, graph_seconds = build_knn_graph(
        dataset.X, k=32, backend=backend, n_jobs=1
    )
    groups, head = reconstruct_stratification(graph, dimension, source)
    rows = []
    for dense_quantile in DENSE_QUANTILES:
        for sparse_quantile in SPARSE_QUANTILES:
            if sparse_quantile > dense_quantile:
                continue
            (
                point_eps,
                seed_mask,
                tail_pvalue,
                tail_activated,
                profile,
                eps_profile,
            ) = scheduled_thresholds(
                graph,
                groups,
                head,
                dense_quantile,
                sparse_quantile,
            )
            labels, core = core_components(graph, point_eps, seed_mask)
            labels, _ = expand_core_to_border(
                graph,
                labels,
                core,
                point_eps > 0.0,
                point_eps,
                32,
                1.25,
            )
            metrics = evaluate(dataset, labels, "synthetic")
            rows.append(
                {
                    "dataset_id": case["id"],
                    "seed": seed,
                    "head_strata": head,
                    "dense_quantile": dense_quantile,
                    "sparse_quantile": sparse_quantile,
                    "quantile_profile": profile.tolist(),
                    "eps_profile": eps_profile.tolist(),
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


def aggregate(frame: pd.DataFrame, output: Path):
    keys = ["dataset_id", "dense_quantile", "sparse_quantile"]
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
    summary = frame.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "aggregate-stratum-quantiles.csv", index=False)
    global_ranking = (
        summary.groupby(["dense_quantile", "sparse_quantile"], sort=False)
        .agg(
            mean_dataset_macro_f1=("macro_target_f1_mean", "mean"),
            worst_dataset_macro_f1=("macro_target_f1_mean", "min"),
            mean_background_rejection=("background_rejection_mean", "mean"),
            mean_signal_coverage=("signal_coverage_mean", "mean"),
            mean_core_fraction=("core_fraction_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_dataset_macro_f1", ascending=False)
    )
    global_ranking.to_csv(output / "global-stratum-quantile-ranking.csv", index=False)
    best = summary.loc[
        summary.groupby("dataset_id")["macro_target_f1_mean"].idxmax()
    ].sort_values("dataset_id")
    best.to_csv(output / "best-stratum-quantiles-by-dataset.csv", index=False)
    return summary, global_ranking


def plot_heatmaps(summary: pd.DataFrame, output: Path):
    datasets = summary["dataset_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        2, 4, figsize=(13.5, 6.6), constrained_layout=True
    )
    for axis, dataset_id in zip(axes.flat, datasets, strict=False):
        subset = summary.loc[summary["dataset_id"] == dataset_id]
        table = subset.pivot(
            index="dense_quantile",
            columns="sparse_quantile",
            values="macro_target_f1_mean",
        ).reindex(index=DENSE_QUANTILES, columns=SPARSE_QUANTILES)
        image = axis.imshow(table.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
        for y in range(len(DENSE_QUANTILES)):
            for x in range(len(SPARSE_QUANTILES)):
                value = table.iloc[y, x]
                if np.isfinite(value):
                    axis.text(
                        x, y, f"{value:.2f}", ha="center", va="center",
                        color="white" if value < 0.55 else "black", fontsize=7,
                    )
        axis.set_xticks(range(len(SPARSE_QUANTILES)), SPARSE_QUANTILES, rotation=45)
        axis.set_yticks(range(len(DENSE_QUANTILES)), DENSE_QUANTILES)
        axis.set_xlabel("least-dense stratum quantile")
        axis.set_ylabel("densest stratum quantile")
        axis.set_title(dataset_id)
    for axis in axes.flat[len(datasets):]:
        axis.set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.65, label="mean macro target F1")
    fig.savefig(output / "stratum-quantile-schedules.png", dpi=180)
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
        default=REPO / "results" / "stratum_quantiles_10k_20260727",
    )
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    predictive = pd.read_csv(args.predictive_metrics)
    checkpoint = args.output / "per-seed-stratum-quantiles.csv"
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
    plot_heatmaps(summary, args.output)
    print(f"finished {len(frame)} runs in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
