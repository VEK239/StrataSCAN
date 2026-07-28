from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from stratascan import StrataSCAN, build_knn_graph
from stratascan.predictive import (
    PredictiveMultiscaleConfig,
    estimate_predictive_multiscale_stratification,
)
from stratascan.strict_core import gamma_strict_core_from_graph


SEARCH_MODES = ("full", "coarse_refine", "coarse_refine_warm")


def run_mode(dataset, graph, backend: str, mode: str) -> dict:
    if mode == "full":
        config = PredictiveMultiscaleConfig(search_mode="full")
    else:
        config = PredictiveMultiscaleConfig(
            search_mode=mode,
            validation_repeats=2,
            selection_n_init=1,
            final_n_init=1,
            selection_max_iter=100,
            selection_tolerance=1e-5,
        )
    started = perf_counter()
    stratification = estimate_predictive_multiscale_stratification(
        graph,
        ambient_dimension=float(dataset.X.shape[1]),
        config=config,
    )
    stratification_seconds = perf_counter() - started
    model = StrataSCAN(
        backend=backend,
        n_jobs=1,
        ambient_dimension=float(dataset.X.shape[1]),
    )
    started = perf_counter()
    result = gamma_strict_core_from_graph(
        graph,
        ambient_dimension=float(dataset.X.shape[1]),
        config=model._config(),
        stratification=stratification,
    )
    clustering_seconds = perf_counter() - started
    metrics = evaluate(dataset, result.labels, "synthetic")
    diagnostics = stratification.diagnostics
    return {
        "mode": mode,
        "components": diagnostics["predictive_components"],
        "head_strata": diagnostics["predictive_supported_components"],
        "best_components": diagnostics["predictive_best_components"],
        "candidate_components": diagnostics["predictive_candidate_components"],
        "fit_count": diagnostics["predictive_fit_count"],
        "fold_iterations": diagnostics["predictive_fold_iterations"],
        "final_iterations": diagnostics["predictive_final_iterations"],
        "stratification_seconds": stratification_seconds,
        "clustering_seconds": clustering_seconds,
        "total_after_graph_seconds": stratification_seconds + clustering_seconds,
        **{
            key: metrics[key]
            for key in (
                "macro_target_f1",
                "pairwise_f1",
                "background_rejection",
                "signal_coverage",
                "n_clusters",
            )
        },
    }


def aggregate(frame: pd.DataFrame, baseline: pd.DataFrame, output: Path) -> None:
    joined = frame.merge(
        baseline[
            [
                "dataset_id",
                "seed",
                "mixture_components",
                "head_strata",
                "fit_seconds",
                "macro_target_f1",
            ]
        ],
        on=["dataset_id", "seed"],
        suffixes=("", "_full"),
        validate="many_to_one",
    )
    joined["delta_macro_target_f1"] = (
        joined["macro_target_f1"] - joined["macro_target_f1_full"]
    )
    joined["same_components"] = joined["components"] == joined["mixture_components"]
    joined["same_head_strata"] = joined["head_strata"] == joined["head_strata_full"]
    joined.to_csv(output / "comparison-with-full.csv", index=False)
    summary = (
        joined.groupby("mode", sort=False)
        .agg(
            runs=("seed", "size"),
            mean_fit_count=("fit_count", "mean"),
            mean_seconds=("total_after_graph_seconds", "mean"),
            median_seconds=("total_after_graph_seconds", "median"),
            full_recorded_seconds=("fit_seconds", "mean"),
            speedup_vs_recorded_full=(
                "fit_seconds",
                lambda values: float(np.mean(values)),
            ),
            component_agreement=("same_components", "mean"),
            head_agreement=("same_head_strata", "mean"),
            mean_macro_f1=("macro_target_f1", "mean"),
            full_mean_macro_f1=("macro_target_f1_full", "mean"),
            mean_delta_macro_f1=("delta_macro_target_f1", "mean"),
            worst_delta_macro_f1=("delta_macro_target_f1", "min"),
        )
        .reset_index()
    )
    summary["speedup_vs_recorded_full"] = (
        summary["full_recorded_seconds"] / summary["mean_seconds"]
    )
    summary.to_csv(output / "global-search-mode-summary.csv", index=False)
    per_dataset = (
        joined.groupby(["dataset_id", "mode"], sort=False)
        .agg(
            mean_seconds=("total_after_graph_seconds", "mean"),
            component_agreement=("same_components", "mean"),
            macro_target_f1=("macro_target_f1", "mean"),
            full_macro_target_f1=("macro_target_f1_full", "mean"),
            delta_macro_target_f1=("delta_macro_target_f1", "mean"),
        )
        .reset_index()
    )
    per_dataset.to_csv(output / "per-dataset-search-mode-summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "benchmarks" / "protocol.full-legacy-grid-7synthetic.json",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO
        / "results"
        / "predictive_strata_10k_20260727"
        / "per-seed-metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "predictive_search_modes_fast_10k_20260727",
    )
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--seeds", type=str, default="23,42,73,101,151")
    parser.add_argument(
        "--modes", type=str, default="coarse_refine,coarse_refine_warm"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    checkpoint = args.output / "per-seed-search-modes.csv"
    existing = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    completed = (
        set(zip(existing["dataset_id"], existing["seed"], existing["mode"], strict=True))
        if len(existing)
        else set()
    )
    rows = existing.to_dict("records")
    selected_cases = {value for value in args.cases.split(",") if value}
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    selected_modes = tuple(value for value in args.modes.split(",") if value)
    if not selected_modes or any(value not in SEARCH_MODES for value in selected_modes):
        raise ValueError(f"modes must be selected from {SEARCH_MODES}")
    for case in protocol["synthetic"]["cases"]:
        if selected_cases and case["id"] not in selected_cases:
            continue
        for seed in selected_seeds:
            missing = [
                mode
                for mode in selected_modes
                if (case["id"], seed, mode) not in completed
            ]
            if not missing:
                continue
            dataset = load_synthetic({**case, "n": args.n}, seed)
            dimension = int(dataset.X.shape[1])
            backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
            graph, graph_seconds = build_knn_graph(
                dataset.X, k=32, backend=backend, n_jobs=1
            )
            if seed % 2:
                missing.reverse()
            for mode in missing:
                row = run_mode(dataset, graph, backend, mode)
                rows.append(
                    {
                        "dataset_id": case["id"],
                        "seed": seed,
                        "n": args.n,
                        "dimension": dimension,
                        "graph_seconds": graph_seconds,
                        **row,
                    }
                )
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                print(
                    f"completed {case['id']} seed={seed} mode={mode} "
                    f"seconds={row['total_after_graph_seconds']:.2f}",
                    flush=True,
                )
    aggregate(pd.DataFrame(rows), pd.read_csv(args.baseline), args.output)


if __name__ == "__main__":
    main()
