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


QUANTILES = (0.95, 0.75, 0.50, 0.25, 0.10, 0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.results)
    bad = raw.loc[
        raw["suite"].eq("gaia")
        & raw["status"].eq("ok")
        & raw["method"].eq("StrataSCAN")
        & raw["seed"].eq(42)
        & (raw["best_cluster_f1"] < 0.5)
    ].sort_values("dataset_id")
    if len(bad) != 28:
        raise ValueError(f"expected 28 bad fields, found {len(bad)}")
    data_root = REPO / "data/raw/mctnc/MCTNC_open_data_release/MCTNC_open_data_release/data"
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "gaia-multiscale-quantile-sweep-fields.csv"
    if checkpoint.exists():
        rows = pd.read_csv(checkpoint).to_dict("records")
        completed = set(pd.DataFrame(rows)["field_id"].astype(str))
    else:
        rows = []
        completed = set()
    pending = bad.loc[~bad["dataset_id"].str.replace("__unsupervised_field", "").isin(completed)]
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
        for tail_mode, working_strat in (
            ("probe", strat),
            ("all_supported", all_supported),
        ):
            for quantile in QUANTILES:
                config = GammaStrictCoreConfig(
                    dense_core_quantile=quantile,
                    tail_core_quantile=0.05,
                )
                result = gamma_strict_core_from_graph(
                    graph,
                    ambient_dimension=5.0,
                    config=config,
                    stratification=working_strat,
                )
                metrics = evaluate(dataset, result.labels, "gaia")
                rows.append(
                    {
                        "dataset_id": row.dataset_id,
                        "field_id": field_id,
                        "n": len(dataset.y),
                        "reference_members": int(np.sum(dataset.y >= 0)),
                        "baseline_f1": float(row.best_cluster_f1),
                        "tail_mode": tail_mode,
                        "core_quantile": quantile,
                        "f1": metrics["best_cluster_f1"],
                        "precision": metrics["best_cluster_precision"],
                        "recall": metrics["best_cluster_recall"],
                        "n_clusters": metrics["n_clusters"],
                        "noise_fraction": metrics["noise_fraction"],
                        "gamma_components": strat.diagnostics["multiscale_components"],
                        "supported_components": len(working_strat.supported_groups),
                        "graph_seconds": graph_seconds,
                    }
                )
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        print(f"processed {index}/{len(pending)} remaining fields: {field_id}", flush=True)
    output = pd.DataFrame(rows)
    summary = (
        output.groupby(["tail_mode", "core_quantile"], observed=True)
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
        .sort_values("mean_f1", ascending=False)
    )
    summary.to_csv(args.output / "gaia-multiscale-quantile-sweep-summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
