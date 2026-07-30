from __future__ import annotations

import argparse
from dataclasses import replace
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
from stratascan import build_knn_graph
from stratascan.multiscale import estimate_multiscale_stratification
from stratascan.strict_core import GammaStrictCoreConfig, gamma_strict_core_from_graph


def jobs(protocol: dict, shard_index: int, shard_count: int) -> list[dict]:
    synthetic = protocol["synthetic"]
    expanded = []
    for case in synthetic["cases"]:
        for n in synthetic["quality_sizes"]:
            for seed in synthetic["quality_seeds"]:
                expanded.append(
                    {
                        "dataset_id": f"{case['id']}__quality__n{n}",
                        "case_id": case["id"],
                        "seed": int(seed),
                        "spec": {**case, "n": int(n)},
                    }
                )
    expanded.sort(key=lambda row: (row["dataset_id"], row["seed"]))
    return expanded[shard_index::shard_count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--base-quantile", type=float, default=0.10)
    parser.add_argument("--min-core-seeds-per-stratum", type=int, default=32)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    work = jobs(protocol, args.shard_index, args.shard_count)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "synthetic-multiscale-adaptive.csv"
    if checkpoint.exists():
        prior = pd.read_csv(checkpoint)
        rows = prior.to_dict("records")
        completed = set(zip(prior["dataset_id"].astype(str), prior["seed"].astype(int)))
    else:
        rows = []
        completed = set()
    pending = [row for row in work if (row["dataset_id"], row["seed"]) not in completed]
    for index, job in enumerate(pending, start=1):
        started = perf_counter()
        dataset = load_synthetic(job["spec"], job["seed"])
        dimension = int(dataset.X.shape[1])
        backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
        graph, graph_seconds = build_knn_graph(
            dataset.X, k=32, backend=backend, n_jobs=1
        )
        strat = estimate_multiscale_stratification(
            graph, ambient_dimension=float(dimension)
        )
        all_supported = replace(
            strat,
            supported_groups=np.arange(strat.selected_components, dtype=np.int32),
            mode="multiscale_gamma_all_components_supported",
        )
        result = gamma_strict_core_from_graph(
            graph,
            ambient_dimension=float(dimension),
            config=GammaStrictCoreConfig(
                dense_core_quantile=args.base_quantile,
                min_core_seeds_per_stratum=args.min_core_seeds_per_stratum,
            ),
            stratification=all_supported,
        )
        metrics = evaluate(dataset, result.labels, "synthetic")
        rows.append(
            {
                "dataset_id": job["dataset_id"],
                "case_id": job["case_id"],
                "seed": job["seed"],
                "n": len(dataset.y),
                "dimension": dimension,
                "pairwise_f1": metrics["pairwise_f1"],
                "pairwise_precision": metrics["pairwise_precision"],
                "pairwise_recall": metrics["pairwise_recall"],
                "ari_signal": metrics["ari_signal"],
                "ami_signal": metrics["ami_signal"],
                "macro_target_f1": metrics["macro_target_f1"],
                "background_rejection": metrics["background_rejection"],
                "noise_f1": metrics["noise_f1"],
                "noise_precision": metrics["noise_precision"],
                "noise_recall": metrics["noise_recall"],
                "target_success_rate_f1_080": metrics["target_success_rate_f1_080"],
                "recovered_truth_clusters_f1_080": metrics["recovered_truth_clusters_f1_080"],
                "recovered_truth_clusters_f1_090": metrics["recovered_truth_clusters_f1_090"],
                "truth_clusters": metrics["truth_clusters"],
                "n_clusters": metrics["n_clusters"],
                "noise_fraction": metrics["noise_fraction"],
                "runtime_seconds": perf_counter() - started,
                "graph_seconds": graph_seconds,
                "gamma_components": strat.diagnostics["multiscale_components"],
                "operational_components": strat.selected_components,
            }
        )
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        print(
            f"processed {index}/{len(pending)} remaining: {job['dataset_id']} seed={job['seed']}",
            flush=True,
        )
    print(json.dumps({"jobs": len(rows), "complete": len(rows) == len(work)}, indent=2))


if __name__ == "__main__":
    main()
