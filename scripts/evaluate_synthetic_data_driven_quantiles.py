from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from stratascan import build_knn_graph
from stratascan.core import stratified_knn_dbscan_from_graph
from stratascan.multiscale import estimate_multiscale_stratification


Q_GRID = np.asarray([0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95])


def stratum_diagnostics(graph, mask: np.ndarray, eps: float) -> dict[str, float]:
    rank = 4
    core = mask & (graph.distances[:, rank - 1] <= eps)
    rows_idx = np.flatnonzero(core).astype(np.int32)
    accepted = np.zeros(graph.n_samples, dtype=bool)
    components = 0
    if rows_idx.size:
        local = np.full(graph.n_samples, -1, dtype=np.int32)
        local[rows_idx] = np.arange(rows_idx.size, dtype=np.int32)
        rows = np.repeat(rows_idx, rank)
        cols = graph.indices[rows_idx, :rank].reshape(-1)
        keep = core[cols]
        adjacency = coo_matrix(
            (
                np.ones(int(np.sum(keep)), dtype=np.uint8),
                (local[rows[keep]], local[cols[keep]]),
            ),
            shape=(rows_idx.size, rows_idx.size),
        ).tocsr()
        count, labels = connected_components(adjacency, directed=False, return_labels=True)
        sizes = np.bincount(labels, minlength=count)
        valid = sizes >= 5
        components = int(np.sum(valid))
        accepted[rows_idx] = valid[labels]

    covered = accepted.copy()
    border = np.flatnonzero(mask & ~accepted).astype(np.int32)
    if border.size:
        neighbours = graph.indices[border, :rank]
        eligible = accepted[neighbours] & (
            graph.distances[border, :rank] <= eps * 1.25
        )
        covered[border] = np.any(eligible, axis=1)
    size = int(np.sum(mask))
    return {
        "core_fraction": float(np.sum(accepted & mask) / max(size, 1)),
        "coverage": float(np.sum(covered & mask) / max(size, 1)),
        "components": components,
    }


def choose_quantiles(
    graph, groups: np.ndarray, n_groups: int, supported_groups: np.ndarray, target: float
):
    kth = graph.distances[:, 3]
    quantiles = np.zeros(n_groups, dtype=float)
    eps = np.zeros(n_groups, dtype=np.float32)
    trace: list[dict] = []
    for group in map(int, supported_groups):
        mask = groups == group
        values = kth[mask]
        chosen = len(Q_GRID) - 1
        group_rows = []
        for index, q in enumerate(Q_GRID):
            candidate_eps = float(np.quantile(values, q))
            diagnostic = stratum_diagnostics(graph, mask, candidate_eps)
            group_rows.append({"group": group, "q": q, **diagnostic})
            if diagnostic["coverage"] >= target:
                chosen = index
                break
        quantiles[group] = Q_GRID[chosen]
        eps[group] = np.quantile(values, quantiles[group])
        trace.extend(group_rows)
    return quantiles, eps, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = []

    for case in protocol["synthetic"]["cases"]:
        dataset = load_synthetic({**case, "n": args.n}, args.seed)
        dimension = int(dataset.X.shape[1])
        backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
        graph, _ = build_knn_graph(dataset.X, k=32, backend=backend, n_jobs=1)
        strat = estimate_multiscale_stratification(
            graph, ambient_dimension=float(dimension)
        )
        support_modes = {
            "gamma_supported": strat.supported_groups,
            "all_supported": np.arange(strat.selected_components, dtype=np.int32),
        }
        for support_mode, supported_groups in support_modes.items():
          for target in (0.50, 0.70, 0.85, 0.95):
            quantiles, eps, trace = choose_quantiles(
                graph, strat.groups, strat.selected_components, supported_groups, target
            )
            point_eps = eps[strat.groups]
            labels, core, _ = stratified_knn_dbscan_from_graph(
                graph,
                point_eps,
                min_samples=5,
                min_cluster_size=5,
                border_multiplier=1.25,
            )
            metrics = evaluate(dataset, labels, "synthetic")
            selected_q = quantiles[supported_groups]
            rows.append({
                "case_id": case["id"],
                "support_mode": support_mode,
                "method": f"coverage_{target:.2f}",
                "pairwise_f1": metrics["pairwise_f1"],
                "pairwise_precision": metrics["pairwise_precision"],
                "pairwise_recall": metrics["pairwise_recall"],
                "macro_target_f1": metrics["macro_target_f1"],
                "background_rejection": metrics["background_rejection"],
                "noise_f1": metrics["noise_f1"],
                "n_clusters": metrics["n_clusters"],
                "noise_fraction": metrics["noise_fraction"],
                "mean_q": float(np.mean(selected_q)),
                "min_q": float(np.min(selected_q)),
                "max_q": float(np.max(selected_q)),
                "quantiles": json.dumps(quantiles.tolist()),
                "core_fraction": float(np.mean(core)),
            })
            traces.extend(
                {"case_id": case["id"], "support_mode": support_mode, "target": target, **item}
                for item in trace
            )
        print(f"completed {case['id']}", flush=True)

    pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(traces).to_csv(args.output / "selection-traces.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
