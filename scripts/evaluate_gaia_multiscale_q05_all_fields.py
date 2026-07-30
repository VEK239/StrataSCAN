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

from benchmarks.datasets import load_gaia
from benchmarks.evaluation import evaluate
from stratascan import build_knn_graph
from stratascan.multiscale import estimate_multiscale_stratification
from stratascan.strict_core import GammaStrictCoreConfig, gamma_strict_core_from_graph


def field_rows(results: Path) -> pd.DataFrame:
    raw = pd.read_csv(results)
    rows = raw.loc[
        raw["suite"].eq("gaia")
        & raw["status"].eq("ok")
        & raw["method"].eq("StrataSCAN")
        & raw["seed"].eq(42)
    ].copy()
    if rows["dataset_id"].nunique() != 359:
        raise ValueError(f"expected 359 fields, found {rows['dataset_id'].nunique()}")
    return rows.sort_values("dataset_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--min-core-seeds-per-stratum", type=int, default=0)
    parser.add_argument("--base-quantile", type=float, default=0.05)
    parser.add_argument("--fields", nargs="*", default=None)
    args = parser.parse_args()
    source = field_rows(args.results)
    if args.fields:
        selected_fields = set(args.fields)
        source = source.loc[
            source["dataset_id"].str.replace("__unsupervised_field", "", regex=False).isin(selected_fields)
        ].copy()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must lie within shard-count")
    source = source.iloc[args.shard_index :: args.shard_count].copy()
    data_root = REPO / "data/raw/mctnc/MCTNC_open_data_release/MCTNC_open_data_release/data"
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "gaia-multiscale-adaptive-fields.csv"
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        rows = existing.to_dict("records")
        completed = set(existing["dataset_id"].astype(str))
        print(f"resuming after {len(completed)} fields", flush=True)
    else:
        rows = []
        completed = set()
    pending = source.loc[~source["dataset_id"].isin(completed)]
    started = perf_counter()
    for index, row in enumerate(pending.itertuples(index=False), start=1):
        metadata = json.loads(row.dataset_metadata_json)
        field_id = metadata["field_id"]
        dataset = load_gaia(
            {
                "data_root": str(data_root.relative_to(REPO)),
                "field_id": field_id,
                "ruwe_max": 1.6,
                "preprocessing": "unsupervised_field",
            },
            REPO,
        )
        graph, graph_seconds = build_knn_graph(
            dataset.X,
            k=32,
            backend="faiss_hnsw",
            n_jobs=1,
        )
        strat = estimate_multiscale_stratification(graph, ambient_dimension=5.0)
        all_supported = replace(
            strat,
            supported_groups=np.arange(strat.selected_components, dtype=np.int32),
            mode="multiscale_gamma_all_components_supported",
        )
        result = gamma_strict_core_from_graph(
            graph,
            ambient_dimension=5.0,
            config=GammaStrictCoreConfig(
                dense_core_quantile=args.base_quantile,
                min_core_seeds_per_stratum=args.min_core_seeds_per_stratum,
            ),
            stratification=all_supported,
        )
        metrics = evaluate(dataset, result.labels, "gaia")
        rows.append(
            {
                "dataset_id": row.dataset_id,
                "field_id": field_id,
                "n": len(dataset.y),
                "reference_members": int(np.sum(dataset.y >= 0)),
                "baseline_f1": float(row.best_cluster_f1),
                "baseline_precision": float(row.best_cluster_precision),
                "baseline_recall": float(row.best_cluster_recall),
                "variant_f1": metrics["best_cluster_f1"],
                "variant_precision": metrics["best_cluster_precision"],
                "variant_recall": metrics["best_cluster_recall"],
                "delta_f1": metrics["best_cluster_f1"] - float(row.best_cluster_f1),
                "n_clusters": metrics["n_clusters"],
                "noise_fraction": metrics["noise_fraction"],
                "gamma_components": strat.diagnostics["multiscale_components"],
                "operational_components": strat.selected_components,
                "graph_seconds": graph_seconds,
                "profile_json": json.dumps(result.profile),
            }
        )
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        if index % 5 == 0 or index == len(pending):
            print(f"processed {index}/{len(pending)} remaining fields", flush=True)
    output = pd.DataFrame(rows)
    summary = {
        "fields": int(len(output)),
        "variant": (
            "multiscale gamma; all components supported; "
            f"base core quantile {args.base_quantile}; "
            f"minimum {args.min_core_seeds_per_stratum} core seeds per stratum"
        ),
        "baseline_mean_f1": float(output["baseline_f1"].mean()),
        "variant_mean_f1": float(output["variant_f1"].mean()),
        "baseline_median_f1": float(output["baseline_f1"].median()),
        "variant_median_f1": float(output["variant_f1"].median()),
        "improved_fields": int((output["delta_f1"] > 1e-12).sum()),
        "unchanged_fields": int((output["delta_f1"].abs() <= 1e-12).sum()),
        "worsened_fields": int((output["delta_f1"] < -1e-12).sum()),
        "baseline_f1_ge_080": float((output["baseline_f1"] >= 0.8).mean()),
        "variant_f1_ge_080": float((output["variant_f1"] >= 0.8).mean()),
        "baseline_f1_ge_090": float((output["baseline_f1"] >= 0.9).mean()),
        "variant_f1_ge_090": float((output["variant_f1"] >= 0.9).mean()),
        "baseline_f1_lt_050": int((output["baseline_f1"] < 0.5).sum()),
        "variant_f1_lt_050": int((output["variant_f1"] < 0.5).sum()),
        "runtime_seconds_current_session": perf_counter() - started,
    }
    (args.output / "gaia-multiscale-adaptive-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
