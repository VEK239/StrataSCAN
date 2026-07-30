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

from benchmarks.datasets import load_gaia
from benchmarks.evaluation import evaluate
from stratascan import build_knn_graph
from stratascan.multiscale import estimate_multiscale_stratification
from stratascan.strict_core import GammaStrictCoreConfig, gamma_strict_core_from_graph


SEED_BUDGETS = (0, 5, 10, 20, 32, 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("q05_results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-quantile", type=float, default=0.10)
    args = parser.parse_args()
    raw = pd.read_csv(args.results)
    baseline = raw.loc[
        raw["suite"].eq("gaia")
        & raw["status"].eq("ok")
        & raw["method"].eq("StrataSCAN")
        & raw["seed"].eq(42)
    ].copy()
    q05 = pd.read_csv(args.q05_results)
    bad_ids = set(baseline.loc[baseline["best_cluster_f1"] < 0.5, "dataset_id"])
    worst_ids = set(q05.nsmallest(15, "delta_f1")["dataset_id"])
    selected_ids = bad_ids | worst_ids
    selected = baseline.loc[baseline["dataset_id"].isin(selected_ids)].sort_values("dataset_id")
    data_root = REPO / "data/raw/mctnc/MCTNC_open_data_release/MCTNC_open_data_release/data"
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "gaia-multiscale-seed-budget-fields.csv"
    if checkpoint.exists():
        rows = pd.read_csv(checkpoint).to_dict("records")
        completed = set(pd.DataFrame(rows)["field_id"].astype(str))
    else:
        rows = []
        completed = set()
    pending = selected.loc[
        ~selected["dataset_id"].str.replace("__unsupervised_field", "", regex=False).isin(completed)
    ]
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
        graph, _ = build_knn_graph(dataset.X, k=32, backend="faiss_hnsw", n_jobs=1)
        strat = estimate_multiscale_stratification(graph, ambient_dimension=5.0)
        all_supported = replace(
            strat,
            supported_groups=np.arange(strat.selected_components, dtype=np.int32),
            mode="multiscale_gamma_all_components_supported",
        )
        case = "original_bad" if row.dataset_id in bad_ids else "q05_worst_regression"
        for budget in SEED_BUDGETS:
            result = gamma_strict_core_from_graph(
                graph,
                ambient_dimension=5.0,
                config=GammaStrictCoreConfig(
                    dense_core_quantile=args.base_quantile,
                    min_core_seeds_per_stratum=budget,
                ),
                stratification=all_supported,
            )
            metrics = evaluate(dataset, result.labels, "gaia")
            rows.append(
                {
                    "dataset_id": row.dataset_id,
                    "field_id": field_id,
                    "case": case,
                    "reference_members": int(np.sum(dataset.y >= 0)),
                    "baseline_f1": float(row.best_cluster_f1),
                    "seed_budget": budget,
                    "f1": metrics["best_cluster_f1"],
                    "precision": metrics["best_cluster_precision"],
                    "recall": metrics["best_cluster_recall"],
                    "n_clusters": metrics["n_clusters"],
                    "noise_fraction": metrics["noise_fraction"],
                }
            )
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        print(f"processed {index}/{len(pending)} remaining fields: {field_id}", flush=True)
    output = pd.DataFrame(rows)
    summary = (
        output.groupby(["case", "seed_budget"], observed=True)
        .agg(
            fields=("field_id", "nunique"),
            mean_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            f1_ge_080=("f1", lambda x: int((x >= 0.8).sum())),
            f1_ge_090=("f1", lambda x: int((x >= 0.9).sum())),
            f1_lt_050=("f1", lambda x: int((x < 0.5).sum())),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "gaia-multiscale-seed-budget-summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
