from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np
import pandas as pd
import psutil


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import gaia_field_ids, load_cytometry, load_gaia, load_synthetic
from benchmarks.evaluation import evaluate
from stratascan import build_knn_graph
from stratascan.predictive import (
    PredictiveMultiscaleConfig,
    estimate_predictive_multiscale_stratification,
)
from stratascan.strict_core import GammaStrictCoreConfig, gamma_strict_core_from_graph


DEFAULT_Q = 0.95
SPARSE_Q = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.99,
)
SPARSE_DATASETS = {"mosmann", "nilsson"}
METRICS = (
    "pairwise_f1",
    "pairwise_precision",
    "pairwise_recall",
    "noise_f1",
    "binary_macro_f1",
    "background_rejection",
    "signal_coverage",
    "ari_signal",
    "ami_signal",
    "macro_target_f1",
    "target_success_rate_f1_080",
    "recovered_truth_clusters_f1_080",
    "recovered_truth_clusters_f1_090",
    "truth_clusters",
    "spurious_background_majority_clusters",
    "merged_predicted_clusters",
    "extra_signal_fragments",
    "mean_target_fragments",
    "n_clusters",
    "noise_fraction",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_tasks(protocol: dict) -> list[dict]:
    tasks = []
    config = protocol["synthetic"]
    tiers = (
        ("quality", config["quality_sizes"], config["quality_seeds"]),
        ("scalability", config["scaling_sizes"], config["scaling_seeds"]),
    )
    for case in config["cases"]:
        for tier, sizes, seeds in tiers:
            for n in sizes:
                for seed in seeds:
                    tasks.append(
                        {
                            "suite": "synthetic",
                            "dataset_id": case["id"],
                            "tier": tier,
                            "n_requested": int(n),
                            "seed": int(seed),
                            "spec": {**case, "n": int(n)},
                        }
                    )
    return tasks


def cytometry_tasks(protocol: dict, samusik_01: dict, samusik_all: dict) -> list[dict]:
    specs = list(protocol["cytometry"]["datasets"])
    specs.extend(samusik_01["cytometry"]["datasets"])
    specs.extend(samusik_all["cytometry"]["datasets"])
    return [
        {
            "suite": "cytometry",
            "dataset_id": spec["id"],
            "tier": "real",
            "n_requested": 0,
            "seed": 42,
            "spec": spec,
        }
        for spec in specs
    ]


def gaia_tasks(protocol: dict) -> list[dict]:
    config = protocol["gaia"]
    data_root = REPO / config["data_root"]
    return [
        {
            "suite": "gaia",
            "dataset_id": field_id,
            "tier": "real",
            "n_requested": 0,
            "seed": 42,
            "spec": {
                "data_root": config["data_root"],
                "field_id": field_id,
                "ruwe_max": config["ruwe_max"],
                "preprocessing": "unsupervised_field",
            },
        }
        for field_id in gaia_field_ids(data_root)
    ]


def task_key(task: dict) -> str:
    return "|".join(
        map(
            str,
            (
                task["suite"],
                task["dataset_id"],
                task["tier"],
                task["n_requested"],
                task["seed"],
            ),
        )
    )


def load_task(task: dict):
    if task["suite"] == "synthetic":
        return load_synthetic(task["spec"], task["seed"])
    if task["suite"] == "cytometry":
        return load_cytometry(task["spec"], REPO)
    return load_gaia(task["spec"], REPO)


def run_task(
    task: dict,
    *,
    q_values: tuple[float, ...] | None = None,
    q_all: bool = False,
) -> list[dict]:
    started = perf_counter()
    process = psutil.Process()
    rss_start = process.memory_info().rss / 2**20
    try:
        dataset = load_task(task)
        load_seconds = perf_counter() - started
        dimension = int(dataset.X.shape[1])
        backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
        graph_started = perf_counter()
        graph, graph_seconds = build_knn_graph(
            dataset.X, k=32, backend=backend, n_jobs=1
        )
        graph_wall_seconds = perf_counter() - graph_started
        fit_started = perf_counter()
        stratification = estimate_predictive_multiscale_stratification(
            graph,
            ambient_dimension=float(dimension),
            config=PredictiveMultiscaleConfig(search_mode="full"),
        )
        stratification_seconds = perf_counter() - fit_started
        use_override = q_values is not None and (
            q_all or task["dataset_id"] in SPARSE_DATASETS
        )
        quantiles = (
            q_values
            if use_override
            else SPARSE_Q
            if task["dataset_id"] in SPARSE_DATASETS
            else (DEFAULT_Q,)
        )
        rows = []
        for q in quantiles:
            core_started = perf_counter()
            result = gamma_strict_core_from_graph(
                graph,
                ambient_dimension=float(dimension),
                config=GammaStrictCoreConfig(dense_core_quantile=float(q)),
                stratification=stratification,
            )
            core_seconds = perf_counter() - core_started
            metrics = evaluate(dataset, result.labels, task["suite"])
            row = {
                "task_key": task_key(task),
                "status": "ok",
                "suite": task["suite"],
                "dataset_id": task["dataset_id"],
                "tier": task["tier"],
                "seed": task["seed"],
                "n": int(dataset.X.shape[0]),
                "dimension": dimension,
                "q": float(q),
                "q_policy": "targeted_sparse_sweep" if len(quantiles) > 1 else "fixed_0.95",
                "backend": backend,
                "load_seconds": load_seconds,
                "graph_seconds": float(graph_seconds),
                "graph_wall_seconds": graph_wall_seconds,
                "stratification_seconds": stratification_seconds,
                "core_seconds": core_seconds,
                "full_primary_seconds": (
                    load_seconds + graph_wall_seconds + stratification_seconds + core_seconds
                ),
                "rss_start_mb": rss_start,
                "rss_end_mb": process.memory_info().rss / 2**20,
                "predictive_components": int(stratification.selected_components),
                "supported_strata": int(len(stratification.supported_groups)),
                "tail_activated": bool(result.profile.get("strict_core_tail_activated", False)),
                "target_matches_json": json.dumps(metrics["target_matches"], sort_keys=True),
                "profile_json": json.dumps(result.profile, sort_keys=True),
                "error": "",
            }
            row.update({name: metrics.get(name, np.nan) for name in METRICS})
            if task["suite"] == "gaia":
                for name in (
                    "best_cluster_f1",
                    "best_cluster_precision",
                    "best_cluster_recall",
                    "matched_predicted_cluster",
                ):
                    row[name] = metrics[name]
            rows.append(row)
        return rows
    except Exception as exc:
        return [
            {
                "task_key": task_key(task),
                "status": "error",
                "suite": task["suite"],
                "dataset_id": task["dataset_id"],
                "tier": task["tier"],
                "seed": task["seed"],
                "n": task["n_requested"],
                "q": DEFAULT_Q,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "full_primary_seconds": perf_counter() - started,
            }
        ]


def write_checkpoint(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["suite", "dataset_id", "tier", "n", "seed", "q"], kind="stable"
    )
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "benchmarks" / "protocol.full-legacy-grid-7synthetic.json",
    )
    parser.add_argument(
        "--samusik-01",
        type=Path,
        default=REPO / "benchmarks" / "protocol.samusik-01.json",
    )
    parser.add_argument(
        "--samusik-all",
        type=Path,
        default=REPO / "benchmarks" / "protocol.samusik-all.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "synthetic", "cytometry", "gaia"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dataset",
        action="append",
        help="restrict execution to one dataset id; may be supplied more than once",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--min-n", type=int)
    parser.add_argument("--max-n", type=int)
    parser.add_argument(
        "--q-values",
        help="comma-separated q values overriding the sparse-data sweep",
    )
    parser.add_argument(
        "--q-all",
        action="store_true",
        help="apply --q-values to every selected dataset, not only Mosmann/Nilsson",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must lie in [0, shard count)")
    q_values = None
    if args.q_values:
        q_values = tuple(float(value) for value in args.q_values.split(","))
        if not q_values or any(not 0.0 < value <= 1.0 for value in q_values):
            parser.error("all q values must lie in (0, 1]")
    if args.q_all and q_values is None:
        parser.error("--q-all requires --q-values")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(args.protocol)
    task_groups = {}
    if "synthetic" in protocol:
        task_groups["synthetic"] = synthetic_tasks(protocol)
    if "cytometry" in protocol:
        task_groups["cytometry"] = cytometry_tasks(
            protocol, read_json(args.samusik_01), read_json(args.samusik_all)
        )
    if "gaia" in protocol:
        task_groups["gaia"] = gaia_tasks(protocol)
    if args.stage != "all" and args.stage not in task_groups:
        parser.error(f"protocol does not define the {args.stage} suite")
    stages = task_groups if args.stage == "all" else {args.stage: task_groups[args.stage]}
    tasks = [task for group in stages.values() for task in group]
    if args.dataset:
        selected_datasets = set(args.dataset)
        tasks = [task for task in tasks if task["dataset_id"] in selected_datasets]
    if args.min_n is not None:
        tasks = [task for task in tasks if task["n_requested"] >= args.min_n]
    if args.max_n is not None:
        tasks = [task for task in tasks if task["n_requested"] <= args.max_n]
    tasks = tasks[args.shard_index :: args.shard_count]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    checkpoint = args.output / "results.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = set(pd.DataFrame(rows).get("task_key", pd.Series(dtype=str)).astype(str))
    pending = [task for task in tasks if task_key(task) not in completed]
    print(
        f"stage={args.stage} total={len(tasks)} completed={len(tasks)-len(pending)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    if args.workers == 1:
        iterator = (
            (task, run_task(task, q_values=q_values, q_all=args.q_all))
            for task in pending
        )
        for index, (task, task_rows) in enumerate(iterator, start=1):
            rows.extend(task_rows)
            write_checkpoint(rows, checkpoint)
            print(
                f"[{index}/{len(pending)}] {task_key(task)} {task_rows[0]['status']}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_task, task, q_values=q_values, q_all=args.q_all
                ): task
                for task in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                task_rows = future.result()
                rows.extend(task_rows)
                write_checkpoint(rows, checkpoint)
                print(
                    f"[{index}/{len(pending)}] {task_key(task)} {task_rows[0]['status']}",
                    flush=True,
                )
    metadata = {
        "algorithm": "PredictiveMultiscaleStrataSCAN",
        "algorithm_version": "0.1.2",
        "default_q": DEFAULT_Q,
        "sparse_q_candidates": SPARSE_Q,
        "sparse_q_datasets": sorted(SPARSE_DATASETS),
        "task_counts": {name: len(value) for name, value in task_groups.items()},
    }
    (args.output / "benchmark-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
