from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import gaia_field_ids, load_cytometry, load_gaia, load_synthetic
from benchmarks.evaluation import evaluate
from stratascan import build_knn_graph
from stratascan.core import stratified_knn_dbscan_from_graph
from stratascan.multiscale import estimate_multiscale_stratification
from stratascan.strict_core import _rank_window_component_sizes


def point_eps_for_groups(graph, groups, selected_groups, x: float = 0.10) -> tuple[np.ndarray, list[float]]:
    kth = graph.distances[:, 3]
    point_eps = np.zeros(graph.n_samples, dtype=np.float32)
    effective = []
    for group in map(int, selected_groups):
        mask = groups == group
        size = int(np.sum(mask))
        q = max(x, min(1.0, 32.0 / max(size, 1)))
        eps = float(np.quantile(kth[mask], q))
        point_eps[mask] = eps
        effective.append(q)
    return point_eps, effective


def cluster_groups(graph, strat, selected_groups) -> tuple[np.ndarray, np.ndarray, list[float]]:
    point_eps, effective = point_eps_for_groups(graph, strat.groups, selected_groups)
    labels, core, _ = stratified_knn_dbscan_from_graph(
        graph,
        point_eps,
        min_samples=5,
        min_cluster_size=5,
        border_multiplier=1.25,
    )
    return labels, core, effective


def rank_window_features(graph, mask: np.ndarray, score: np.ndarray, prefix: str) -> dict[str, float]:
    rows = np.flatnonzero(mask)
    order = rows[np.argsort(score[rows], kind="stable")]
    result: dict[str, float] = {}
    significant = 0
    ratios = []
    zscores = []
    for q in (0.02, 0.05, 0.10, 0.20):
        width = max(5, int(np.ceil(q * rows.size)))
        sizes = _rank_window_component_sizes(
            np.asarray(graph.indices, dtype=np.int32),
            np.asarray(order, dtype=np.int32),
            width,
            4,
        )
        observed = int(sizes[0]) if sizes.size else 0
        null = sizes[1:].astype(float)
        if null.size:
            pvalue = (1 + int(np.sum(null >= observed))) / sizes.size
            ratio = observed / max(float(np.median(null)), 1.0)
            zscore = (observed - float(np.mean(null))) / max(float(np.std(null)), 1.0)
        else:
            pvalue, ratio, zscore = 1.0, 1.0, 0.0
        significant += int(pvalue <= 0.10 and ratio >= 1.5)
        ratios.append(ratio)
        zscores.append(zscore)
        result[f"{prefix}_q{int(q*100):02d}_observed"] = observed
        result[f"{prefix}_q{int(q*100):02d}_p"] = pvalue
        result[f"{prefix}_q{int(q*100):02d}_ratio"] = ratio
    result[f"{prefix}_significant_scales"] = significant
    result[f"{prefix}_max_ratio"] = float(max(ratios))
    result[f"{prefix}_max_z"] = float(max(zscores))
    return result


def analyze_dataset(dataset_id: str, suite: str, dataset, output_rows: list[dict]) -> None:
    dimension = int(dataset.X.shape[1])
    backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
    graph, graph_seconds = build_knn_graph(dataset.X, k=32, backend=backend, n_jobs=1)
    strat = estimate_multiscale_stratification(graph, ambient_dimension=float(dimension))
    head = np.asarray(strat.supported_groups, dtype=np.int32)
    tail = np.asarray(
        sorted(set(range(strat.selected_components)) - set(map(int, head))), dtype=np.int32
    )
    if tail.size != 1:
        raise ValueError(f"expected one combined tail group, got {tail.tolist()}")
    modes = {
        "head_only": head,
        "tail_only": tail,
        "all": np.arange(strat.selected_components, dtype=np.int32),
    }
    mode_metrics = {}
    mode_effective = {}
    for mode, selected in modes.items():
        labels, core, effective = cluster_groups(graph, strat, selected)
        metrics = evaluate(dataset, labels, suite)
        matches = metrics["target_matches"]
        mode_metrics[mode] = {
            "macro_f1": float(metrics["macro_target_f1"]),
            "max_target_f1": float(max((item["f1"] for item in matches), default=0.0)),
            "pairwise_f1": float(metrics["pairwise_f1"]),
            "background_rejection": float(metrics["background_rejection"]),
            "n_clusters": int(metrics["n_clusters"]),
            "core_fraction": float(np.mean(core)),
        }
        if suite == "gaia":
            mode_metrics[mode]["target_f1"] = float(metrics["best_cluster_f1"])
        mode_effective[mode] = effective

    tail_group = int(tail[0])
    tail_mask = strat.groups == tail_group
    d4 = np.maximum(graph.distances[:, 3].astype(float), np.finfo(float).tiny)
    d32 = np.maximum(graph.distances[:, 31].astype(float), np.finfo(float).tiny)
    contrast = d4 / d32
    target = dataset.y >= 0
    target_in_tail = int(np.sum(target & tail_mask))
    properties = {
        "dataset_id": dataset_id,
        "suite": suite,
        "n": graph.n_samples,
        "dimension": dimension,
        "targets": len(dataset.target_names),
        "target_points": int(np.sum(target)),
        "gamma_components": int(strat.diagnostics["multiscale_components"]),
        "head_strata": int(head.size),
        "tail_fraction": float(np.mean(tail_mask)),
        "tail_size": int(np.sum(tail_mask)),
        "target_fraction_in_tail": float(target_in_tail / max(np.sum(target), 1)),
        "tail_target_purity": float(target_in_tail / max(np.sum(tail_mask), 1)),
        "tail_effective_q": float(mode_effective["tail_only"][0]),
        "tail_log_d4_iqr": float(
            np.quantile(np.log(d4[tail_mask]), 0.75) - np.quantile(np.log(d4[tail_mask]), 0.25)
        ),
        "tail_contrast_p05": float(np.quantile(contrast[tail_mask], 0.05)),
        "tail_contrast_p50": float(np.quantile(contrast[tail_mask], 0.50)),
        "graph_seconds": graph_seconds,
    }
    properties.update(rank_window_features(graph, tail_mask, d4, "d4"))
    properties.update(rank_window_features(graph, tail_mask, contrast, "contrast"))
    for mode, metrics in mode_metrics.items():
        for name, value in metrics.items():
            properties[f"{mode}_{name}"] = value
    properties["tail_useful_f1_050"] = int(mode_metrics["tail_only"]["max_target_f1"] >= 0.50)
    properties["tail_improves_macro"] = float(
        mode_metrics["all"]["macro_f1"] - mode_metrics["head_only"]["macro_f1"]
    )
    if suite == "gaia":
        properties["tail_improves_target"] = float(
            mode_metrics["all"]["target_f1"] - mode_metrics["head_only"]["target_f1"]
        )
    output_rows.append(properties)


def selected_gaia_fields(adaptive_path: Path) -> list[str]:
    table = pd.read_csv(adaptive_path)
    table["variant_f1"] = pd.to_numeric(table["variant_f1"])
    table["delta_f1"] = pd.to_numeric(table["delta_f1"])
    chosen: list[str] = []
    chosen.extend(table.nlargest(10, "delta_f1")["field_id"].astype(str))
    chosen.extend(table.nsmallest(10, "delta_f1")["field_id"].astype(str))
    stable = table.loc[(table["delta_f1"].abs() < 0.01) & (table["variant_f1"] >= 0.90)]
    if len(stable):
        chosen.extend(stable.sample(min(10, len(stable)), random_state=42)["field_id"].astype(str))
    difficult = table.loc[table["variant_f1"] < 0.80].nsmallest(10, "variant_f1")
    chosen.extend(difficult["field_id"].astype(str))
    return list(dict.fromkeys(chosen))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--adaptive-gaia", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-gaia", action="store_true")
    parser.add_argument("--gaia-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "stratum-activation-diagnostics.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {str(row["dataset_id"]) for row in rows}

    if not args.gaia_only:
        for case in protocol["synthetic"]["cases"]:
            dataset_id = case["id"]
            if dataset_id in completed:
                continue
            dataset = load_synthetic({**case, "n": 20_000}, seed=42)
            analyze_dataset(dataset_id, "synthetic", dataset, rows)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"completed synthetic {dataset_id}", flush=True)

        levine_spec = next(item for item in protocol["cytometry"]["datasets"] if item["id"] == "levine")
        if "levine" not in completed:
            analyze_dataset("levine", "cytometry", load_cytometry(levine_spec, REPO), rows)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print("completed cytometry levine", flush=True)

    data_root = REPO / protocol["gaia"]["data_root"]
    fields = gaia_field_ids(data_root) if args.all_gaia else selected_gaia_fields(args.adaptive_gaia)
    fields = fields[args.shard_index :: args.shard_count]
    for field_id in fields:
        dataset_id = f"gaia_{field_id}"
        if dataset_id in completed:
            continue
        dataset = load_gaia(
            {
                "data_root": str(data_root.relative_to(REPO)),
                "field_id": field_id,
                "ruwe_max": protocol["gaia"]["ruwe_max"],
                "preprocessing": "unsupervised_field",
            },
            REPO,
        )
        analyze_dataset(dataset_id, "gaia", dataset, rows)
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        print(f"completed gaia {field_id}", flush=True)

    print(pd.DataFrame(rows).groupby("suite")["dataset_id"].count().to_string())


if __name__ == "__main__":
    main()
