from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_gaia
from benchmarks.evaluation import evaluate
from stratascan import StrataSCAN


def low_contrast_components(
    graph,
    *,
    inner_rank: int = 4,
    outer_rank: int = 32,
    quantile: float = 0.02,
    min_size: int = 5,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    d_inner = graph.distances[:, inner_rank - 1]
    d_outer = np.maximum(graph.distances[:, outer_rank - 1], np.finfo(np.float32).tiny)
    contrast = d_inner / d_outer
    threshold = float(np.quantile(contrast, quantile))
    probe = contrast <= threshold
    rows_idx = np.flatnonzero(probe).astype(np.int32)
    if rows_idx.size == 0:
        return [], contrast, threshold
    local = np.full(graph.n_samples, -1, dtype=np.int32)
    local[rows_idx] = np.arange(rows_idx.size, dtype=np.int32)
    rows = np.repeat(rows_idx, inner_rank)
    cols = graph.indices[rows_idx, :inner_rank].reshape(-1)
    keep = probe[cols]
    adjacency = coo_matrix(
        (
            np.ones(int(np.sum(keep)), dtype=np.uint8),
            (local[rows[keep]], local[cols[keep]]),
        ),
        shape=(rows_idx.size, rows_idx.size),
    ).tocsr()
    count, labels = connected_components(adjacency, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=count)
    components = [rows_idx[labels == label] for label in np.flatnonzero(sizes >= min_size)]
    components.sort(key=len, reverse=True)
    return components, contrast, threshold


def split_global_contrast_components(
    labels: np.ndarray,
    graph,
    *,
    quantile: float = 0.02,
    min_size: int = 5,
    parent_size_multiplier: float = 3.0,
    large_parent_fraction: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    components, contrast, threshold = low_contrast_components(
        graph,
        quantile=quantile,
        min_size=min_size,
    )
    out = labels.copy()
    next_label = int(out[out >= 0].max() + 1) if np.any(out >= 0) else 0
    accepted: list[np.ndarray] = []
    skipped_compact_parent = 0
    for component in components:
        if large_parent_fraction is not None:
            eligible = labels[component] < 0
            for parent in np.unique(labels[component][labels[component] >= 0]):
                parent_size = int(np.sum(labels == parent))
                if parent_size / labels.size >= large_parent_fraction:
                    eligible |= labels[component] == parent
            component = component[eligible]
            if component.size < min_size:
                skipped_compact_parent += 1
                continue
        parent_labels, parent_counts = np.unique(labels[component], return_counts=True)
        dominant_parent = int(parent_labels[np.argmax(parent_counts)])
        if dominant_parent >= 0 and large_parent_fraction is None:
            parent_size = int(np.sum(labels == dominant_parent))
            if parent_size <= parent_size_multiplier * component.size:
                skipped_compact_parent += 1
                continue
        out[component] = next_label
        accepted.append(component)
        next_label += 1
    return out, {
        "contrast_threshold": threshold,
        "contrast_median": float(np.median(contrast)),
        "candidate_components": len(components),
        "accepted_components": len(accepted),
        "accepted_points": int(sum(component.size for component in accepted)),
        "accepted_component_sizes": [int(component.size) for component in accepted],
        "skipped_compact_parent": skipped_compact_parent,
    }


def field_specs(results: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(results, usecols=["suite", "dataset_id", "dataset_metadata_json"])
    frame = frame.loc[frame["suite"].eq("gaia")].drop_duplicates("dataset_id")
    specs = []
    for row in frame.itertuples(index=False):
        metadata = json.loads(row.dataset_metadata_json)
        specs.append(
            {
                "dataset_id": row.dataset_id,
                "field_id": metadata["field_id"],
                "ruwe_max": 1.6,
                "preprocessing": "unsupervised_field",
            }
        )
    return sorted(specs, key=lambda value: value["field_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--quantile", type=float, default=0.02)
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--parent-size-multiplier", type=float, default=3.0)
    parser.add_argument("--backend", default="faiss_hnsw")
    parser.add_argument("--large-parent-fraction", type=float, default=None)
    args = parser.parse_args()
    data_root = REPO / "data/raw/mctnc/MCTNC_open_data_release/MCTNC_open_data_release/data"
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "gaia-global-contrast-split-fields.csv"
    specs = field_specs(args.results)
    if args.fields:
        selected = set(args.fields)
        specs = [spec for spec in specs if spec["field_id"] in selected]
    if checkpoint.exists():
        previous = pd.read_csv(checkpoint)
        rows: list[dict[str, object]] = previous.to_dict("records")
        completed = set(previous["field_id"].astype(str))
        specs = [spec for spec in specs if spec["field_id"] not in completed]
        print(f"resuming after {len(completed)} completed fields", flush=True)
    else:
        rows = []
    started = perf_counter()
    for index, spec in enumerate(specs, start=1):
        dataset = load_gaia(
            {
                "data_root": str(data_root.relative_to(REPO)),
                "field_id": spec["field_id"],
                "ruwe_max": spec["ruwe_max"],
                "preprocessing": spec["preprocessing"],
            },
            REPO,
        )
        model = StrataSCAN(backend=args.backend, n_jobs=1, ambient_dimension=5.0).fit(dataset.X)
        baseline = evaluate(dataset, model.labels_, "gaia")
        variant_labels, diagnostics = split_global_contrast_components(
            model.labels_,
            model.graph_,
            quantile=args.quantile,
            min_size=args.min_size,
            parent_size_multiplier=args.parent_size_multiplier,
            large_parent_fraction=args.large_parent_fraction,
        )
        variant = evaluate(dataset, variant_labels, "gaia")
        rows.append(
            {
                "dataset_id": spec["dataset_id"],
                "field_id": spec["field_id"],
                "n": len(dataset.y),
                "reference_members": int(np.sum(dataset.y >= 0)),
                "baseline_f1": baseline["best_cluster_f1"],
                "baseline_precision": baseline["best_cluster_precision"],
                "baseline_recall": baseline["best_cluster_recall"],
                "variant_f1": variant["best_cluster_f1"],
                "variant_precision": variant["best_cluster_precision"],
                "variant_recall": variant["best_cluster_recall"],
                "delta_f1": variant["best_cluster_f1"] - baseline["best_cluster_f1"],
                **diagnostics,
            }
        )
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        if index % 5 == 0 or index == len(specs):
            print(f"processed {index}/{len(specs)} remaining fields", flush=True)
    output = pd.DataFrame(rows)
    output.to_csv(checkpoint, index=False)
    summary = {
        "fields": int(len(output)),
        "quantile": args.quantile,
        "min_size": args.min_size,
        "parent_size_multiplier": args.parent_size_multiplier,
        "large_parent_fraction": args.large_parent_fraction,
        "baseline_mean_f1": float(output["baseline_f1"].mean()),
        "variant_mean_f1": float(output["variant_f1"].mean()),
        "mean_delta_f1": float(output["delta_f1"].mean()),
        "baseline_median_f1": float(output["baseline_f1"].median()),
        "variant_median_f1": float(output["variant_f1"].median()),
        "improved_fields": int((output["delta_f1"] > 1e-12).sum()),
        "unchanged_fields": int((output["delta_f1"].abs() <= 1e-12).sum()),
        "worsened_fields": int((output["delta_f1"] < -1e-12).sum()),
        "baseline_f1_lt_050": int((output["baseline_f1"] < 0.5).sum()),
        "variant_f1_lt_050": int((output["variant_f1"] < 0.5).sum()),
        "baseline_f1_ge_080": float((output["baseline_f1"] >= 0.8).mean()),
        "variant_f1_ge_080": float((output["variant_f1"] >= 0.8).mean()),
        "runtime_seconds": perf_counter() - started,
    }
    (args.output / "gaia-global-contrast-split-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
