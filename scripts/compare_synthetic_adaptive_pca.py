from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from benchmarks.methods import run_method
from stratascan import build_knn_graph
from stratascan.multiscale import estimate_multiscale_stratification
from stratascan.strict_core import GammaStrictCoreConfig, gamma_strict_core_from_graph


DISPLAY_NAMES = {
    "multidensity_2d": "Multidensity (2D)",
    "ultrasparse_16d": "Ultrasparse (16D)",
    "overlapping_density_16d": "Overlapping density (16D)",
    "moons_2d": "Moons (2D)",
    "rings_2d": "Rings (2D)",
    "overlap_8d": "Gaussian overlap (8D)",
    "imbalanced_16d": "Gaussian imbalanced (16D)",
}


def adaptive_labels(X: np.ndarray, q: float, min_seeds: int) -> tuple[np.ndarray, dict, float]:
    dimension = int(X.shape[1])
    backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
    started = perf_counter()
    graph, _ = build_knn_graph(X, k=32, backend=backend, n_jobs=1)
    strat = estimate_multiscale_stratification(graph, ambient_dimension=float(dimension))
    strat = replace(
        strat,
        supported_groups=np.arange(strat.selected_components, dtype=np.int32),
        mode="multiscale_gamma_all_components_supported",
    )
    result = gamma_strict_core_from_graph(
        graph,
        ambient_dimension=float(dimension),
        config=GammaStrictCoreConfig(
            dense_core_quantile=q,
            min_core_seeds_per_stratum=min_seeds,
        ),
        stratification=strat,
    )
    return result.labels, result.profile, perf_counter() - started


def metric_row(case_id: str, method: str, labels: np.ndarray, dataset, seconds: float) -> dict:
    metrics = evaluate(dataset, labels, "synthetic")
    return {
        "case_id": case_id,
        "method": method,
        "pairwise_f1": metrics["pairwise_f1"],
        "pairwise_precision": metrics["pairwise_precision"],
        "pairwise_recall": metrics["pairwise_recall"],
        "ari_signal": metrics["ari_signal"],
        "ami_signal": metrics["ami_signal"],
        "macro_target_f1": metrics["macro_target_f1"],
        "background_rejection": metrics["background_rejection"],
        "noise_f1": metrics["noise_f1"],
        "n_clusters": metrics["n_clusters"],
        "noise_fraction": metrics["noise_fraction"],
        "runtime_seconds": seconds,
    }


def draw_panel(ax, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    noise = labels < 0
    if np.any(noise):
        ax.scatter(
            coords[noise, 0], coords[noise, 1], s=1.1, c="#b8b8b8",
            alpha=0.20, linewidths=0, rasterized=True,
        )
    for index, label in enumerate(sorted(np.unique(labels[labels >= 0]).tolist())):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=2.2,
            color=plt.get_cmap("tab20")(index % 20), alpha=0.72,
            linewidths=0, rasterized=True,
        )
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
        spine.set_linewidth(0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-quantile", type=float, default=0.10)
    parser.add_argument("--min-core-seeds-per-stratum", type=int, default=32)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    cases = protocol["synthetic"]["cases"]
    rows: list[dict] = []
    rendered: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    for case in cases:
        spec = {**case, "n": args.n}
        dataset = load_synthetic(spec, args.seed)

        started = perf_counter()
        baseline = run_method("StrataSCAN", dataset.X, args.seed)
        baseline_seconds = perf_counter() - started
        adaptive, _, adaptive_seconds = adaptive_labels(
            dataset.X, args.base_quantile, args.min_core_seeds_per_stratum
        )

        rows.append(metric_row(case["id"], "StrataSCAN", baseline.labels, dataset, baseline_seconds))
        rows.append(metric_row(case["id"], "Adaptive multiscale", adaptive, dataset, adaptive_seconds))

        coords = PCA(n_components=2, random_state=args.seed).fit_transform(dataset.X)
        rendered.append((case["id"], coords, dataset.y, baseline.labels, adaptive))
        np.savez_compressed(
            args.output / f"{case['id']}-pca-labels.npz",
            pca=coords,
            truth=dataset.y,
            original=baseline.labels,
            adaptive=adaptive,
        )
        print(f"completed {case['id']}", flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(args.output / "metrics.csv", index=False)

    fig, axes = plt.subplots(len(rendered), 3, figsize=(10.2, 20.0), constrained_layout=True)
    for row_index, (case_id, coords, truth, baseline, adaptive) in enumerate(rendered):
        base = scores[(scores.case_id == case_id) & (scores.method == "StrataSCAN")].iloc[0]
        test = scores[(scores.case_id == case_id) & (scores.method == "Adaptive multiscale")].iloc[0]
        draw_panel(axes[row_index, 0], coords, truth, f"{DISPLAY_NAMES[case_id]} — truth")
        draw_panel(
            axes[row_index, 1], coords, baseline,
            f"Original  F1={base.pairwise_f1:.3f}  BG={base.background_rejection:.3f}",
        )
        draw_panel(
            axes[row_index, 2], coords, adaptive,
            f"Adaptive  F1={test.pairwise_f1:.3f}  BG={test.background_rejection:.3f}",
        )
        if row_index == len(rendered) - 1:
            for column in range(3):
                axes[row_index, column].set_xlabel("PC1 / PC2", fontsize=7)

    fig.savefig(args.output / "pca-comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "pca-comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    wide = scores.pivot(index="case_id", columns="method", values=["pairwise_f1", "background_rejection"])
    wide.to_csv(args.output / "metrics-wide.csv")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
