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
from stratascan.strict_core import _dbcv_component_separation
from stratascan import build_knn_graph


CANDIDATE_QUANTILES = np.array(
    [0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995]
)
RULES = (
    "flat95",
    "jump10",
    "jump20",
    "jump30",
    "giant50",
    "giant70",
    "giant85",
    "dbcv",
    "dbcv_coverage",
)


def stratum_curve(graph, group_mask: np.ndarray):
    d4 = np.asarray(graph.distances[:, 3], dtype=float)
    group_size = int(np.sum(group_mask))
    records = []
    for quantile in CANDIDATE_QUANTILES:
        point_eps = np.zeros(graph.n_samples, dtype=np.float32)
        point_eps[group_mask] = float(np.quantile(d4[group_mask], quantile))
        labels, core = core_components(
            graph, point_eps, np.ones(graph.n_samples, dtype=bool)
        )
        expanded, _ = expand_core_to_border(
            graph,
            labels,
            core,
            group_mask,
            point_eps,
            32,
            1.25,
        )
        components = np.unique(labels[group_mask & (labels >= 0)])
        sizes = np.array(
            [np.sum(labels[group_mask] == component) for component in components],
            dtype=float,
        )
        largest = float(np.max(sizes) / group_size) if sizes.size else 0.0
        coverage = float(np.mean(expanded[group_mask] >= 0))
        dbcv_values = []
        dbcv_weights = []
        for component, size in zip(components, sizes, strict=True):
            members = labels == component
            score = _dbcv_component_separation(
                graph,
                group_mask,
                members,
                min_samples=5,
            )
            if np.isfinite(score) and score >= -1.0:
                dbcv_values.append(score)
                dbcv_weights.append(size)
        if dbcv_values:
            dbcv = float(np.average(dbcv_values, weights=dbcv_weights))
        else:
            dbcv = -1.0
        records.append(
            {
                "quantile": float(quantile),
                "largest_fraction": largest,
                "coverage": coverage,
                "components": int(components.size),
                "dbcv": dbcv,
            }
        )
    return pd.DataFrame(records)


def select_quantile(curve: pd.DataFrame, rule: str) -> float:
    eligible = curve["quantile"] <= 0.95
    baseline = 0.95
    if rule == "flat95":
        return baseline
    if rule.startswith("jump"):
        threshold = float(rule.removeprefix("jump")) / 100.0
        values = curve.loc[eligible, "largest_fraction"].to_numpy()
        jumps = np.diff(values)
        if jumps.size:
            index = int(np.argmax(jumps))
            if jumps[index] >= threshold and values[index + 1] >= 0.50:
                return float(curve.loc[eligible, "quantile"].iloc[index])
        return baseline
    if rule.startswith("giant"):
        cap = float(rule.removeprefix("giant")) / 100.0
        valid = eligible & (curve["largest_fraction"] <= cap) & (curve["coverage"] >= 0.80)
        if np.any(valid):
            return float(curve.loc[valid, "quantile"].max())
        return baseline
    valid = eligible & (curve["coverage"] >= 0.80)
    if not np.any(valid):
        return baseline
    candidates = curve.loc[valid].copy()
    if rule == "dbcv":
        score = candidates["dbcv"]
    elif rule == "dbcv_coverage":
        score = 0.5 * (candidates["dbcv"] + 1.0) * candidates["coverage"]
    else:
        raise ValueError(f"unknown adaptive rule: {rule}")
    best = candidates.loc[score == score.max()]
    return float(best["quantile"].max())


def run_dataset(case, seed: int, n: int, source: pd.Series):
    dataset = load_synthetic({**case, "n": n}, seed)
    dimension = int(dataset.X.shape[1])
    backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
    graph, graph_seconds = build_knn_graph(
        dataset.X, k=32, backend=backend, n_jobs=1
    )
    groups, head = reconstruct_stratification(graph, dimension, source)
    curves = [stratum_curve(graph, groups == group) for group in range(head)]
    rows = []
    d4 = np.asarray(graph.distances[:, 3], dtype=float)
    for rule in RULES:
        point_eps, seed_mask, tail_pvalue, tail_activated = prepare_thresholds(
            graph, groups, head, 0.95, "current"
        )
        selected = []
        largest = []
        coverage = []
        dbcv = []
        for group, curve in enumerate(curves):
            quantile = select_quantile(curve, rule)
            selected.append(quantile)
            record = curve.loc[np.isclose(curve["quantile"], quantile)].iloc[0]
            largest.append(float(record["largest_fraction"]))
            coverage.append(float(record["coverage"]))
            dbcv.append(float(record["dbcv"]))
            mask = groups == group
            point_eps[mask] = float(np.quantile(d4[mask], quantile))
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
                "rule": rule,
                "head_strata": head,
                "selected_quantiles": selected,
                "selected_largest_fractions": largest,
                "selected_coverages": coverage,
                "selected_dbcv": dbcv,
                "mean_selected_quantile": float(np.mean(selected)),
                "min_selected_quantile": float(np.min(selected)),
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
    metrics = [
        "macro_target_f1",
        "pairwise_f1",
        "binary_macro_f1",
        "background_rejection",
        "signal_coverage",
        "n_clusters",
        "core_fraction",
        "mean_selected_quantile",
        "min_selected_quantile",
        "spurious_background_majority_clusters",
        "merged_predicted_clusters",
        "extra_signal_fragments",
    ]
    summary = frame.groupby(["dataset_id", "rule"], sort=False)[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "aggregate-adaptive-q.csv", index=False)
    global_ranking = (
        summary.groupby("rule", sort=False)
        .agg(
            mean_dataset_macro_f1=("macro_target_f1_mean", "mean"),
            worst_dataset_macro_f1=("macro_target_f1_mean", "min"),
            mean_background_rejection=("background_rejection_mean", "mean"),
            mean_signal_coverage=("signal_coverage_mean", "mean"),
            mean_selected_quantile=("mean_selected_quantile_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_dataset_macro_f1", ascending=False)
    )
    global_ranking.to_csv(output / "global-adaptive-q-ranking.csv", index=False)
    return summary, global_ranking


def plot_results(summary: pd.DataFrame, output: Path):
    datasets = summary["dataset_id"].drop_duplicates().tolist()
    table = summary.pivot(
        index="dataset_id", columns="rule", values="macro_target_f1_mean"
    ).reindex(index=datasets, columns=RULES)
    fig, axis = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    image = axis.imshow(table.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    for y in range(len(datasets)):
        for x in range(len(RULES)):
            value = table.iloc[y, x]
            axis.text(
                x, y, f"{value:.2f}", ha="center", va="center",
                color="white" if value < 0.55 else "black", fontsize=8,
            )
    axis.set_xticks(range(len(RULES)), RULES, rotation=35, ha="right")
    axis.set_yticks(range(len(datasets)), datasets)
    axis.set_xlabel("unsupervised per-stratum q rule")
    fig.colorbar(image, ax=axis, shrink=0.75, label="mean macro target F1")
    fig.savefig(output / "adaptive-stratum-q-rules.png", dpi=180)
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
        default=REPO / "results" / "adaptive_stratum_q_10k_20260727",
    )
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    predictive = pd.read_csv(args.predictive_metrics)
    checkpoint = args.output / "per-seed-adaptive-q.csv"
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
    plot_results(summary, args.output)
    print(f"finished {len(frame)} runs in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
