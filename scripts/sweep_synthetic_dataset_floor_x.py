from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from stratascan import build_knn_graph
from stratascan.multiscale import estimate_multiscale_stratification
from stratascan.strict_core import GammaStrictCoreConfig, gamma_strict_core_from_graph


X_GRID = np.asarray(
    [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    dtype=float,
)


def dataset_properties(graph, strat, dimension: int) -> dict[str, float | int]:
    d4 = np.maximum(graph.distances[:, 3].astype(float), np.finfo(float).tiny)
    d32 = np.maximum(graph.distances[:, 31].astype(float), np.finfo(float).tiny)
    ratio = d4 / d32
    sizes = np.bincount(strat.groups, minlength=strat.selected_components)
    weights = sizes / sizes.sum()
    entropy = -np.sum(weights[weights > 0] * np.log(weights[weights > 0]))
    gaps = np.asarray(strat.diagnostics.get("multiscale_density_gap_ratios", []), dtype=float)
    return {
        "dimension": dimension,
        "gamma_components": int(strat.diagnostics["multiscale_components"]),
        "operational_strata": int(strat.selected_components),
        "largest_stratum_fraction": float(np.max(weights)),
        "smallest_stratum_fraction": float(np.min(weights)),
        "stratum_entropy": float(entropy),
        "max_density_gap": float(np.max(gaps)) if gaps.size else np.nan,
        "gamma_background_fraction": float(
            strat.diagnostics.get("multiscale_background_fraction", np.nan)
        ),
        "log_d4_iqr": float(np.quantile(np.log(d4), 0.75) - np.quantile(np.log(d4), 0.25)),
        "d4_d32_p05": float(np.quantile(ratio, 0.05)),
        "d4_d32_p50": float(np.quantile(ratio, 0.50)),
        "d4_d32_iqr": float(np.quantile(ratio, 0.75) - np.quantile(ratio, 0.25)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for case in protocol["synthetic"]["cases"]:
        dataset = load_synthetic({**case, "n": args.n}, args.seed)
        dimension = int(dataset.X.shape[1])
        backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
        graph, _ = build_knn_graph(dataset.X, k=32, backend=backend, n_jobs=1)
        strat = estimate_multiscale_stratification(graph, ambient_dimension=float(dimension))
        all_supported = replace(
            strat,
            supported_groups=np.arange(strat.selected_components, dtype=np.int32),
            mode="multiscale_gamma_all_components_supported",
        )
        properties = dataset_properties(graph, strat, dimension)
        for x in X_GRID:
            result = gamma_strict_core_from_graph(
                graph,
                ambient_dimension=float(dimension),
                config=GammaStrictCoreConfig(
                    dense_core_quantile=float(x),
                    min_core_seeds_per_stratum=32,
                ),
                stratification=all_supported,
            )
            metrics = evaluate(dataset, result.labels, "synthetic")
            rows.append({
                "case_id": case["id"],
                "x": x,
                "pairwise_f1": metrics["pairwise_f1"],
                "pairwise_precision": metrics["pairwise_precision"],
                "pairwise_recall": metrics["pairwise_recall"],
                "macro_target_f1": metrics["macro_target_f1"],
                "background_rejection": metrics["background_rejection"],
                "noise_f1": metrics["noise_f1"],
                "n_clusters": metrics["n_clusters"],
                "noise_fraction": metrics["noise_fraction"],
                "effective_quantiles": json.dumps(
                    result.profile["strict_core_effective_dense_quantiles"]
                ),
                **properties,
            })
        print(f"completed {case['id']}", flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(args.output / "x-sweep.csv", index=False)
    oracle = scores.loc[scores.groupby("case_id")["pairwise_f1"].idxmax()].copy()
    oracle.to_csv(args.output / "oracle-x.csv", index=False)
    summary = scores.groupby("x", as_index=False).agg(
        mean_pairwise_f1=("pairwise_f1", "mean"),
        median_pairwise_f1=("pairwise_f1", "median"),
        mean_background_rejection=("background_rejection", "mean"),
    )
    summary.to_csv(args.output / "x-summary.csv", index=False)
    print(oracle[["case_id", "x", "pairwise_f1", "background_rejection", "n_clusters"]].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
