from __future__ import annotations

import argparse
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
from stratascan import StrataSCAN, build_knn_graph
from stratascan.optimization import (
    GammaMDLConfig,
    OptimizationStrictCoreConfig,
    estimate_mdl_multiscale_stratification,
    optimize_strict_core_from_graph,
)


DISPLAY_NAMES = {
    "multidensity_2d": "Multidensity (2D)",
    "ultrasparse_16d": "Ultrasparse (16D)",
    "overlapping_density_16d": "Overlapping density (16D)",
    "moons_2d": "Moons (2D)",
    "rings_2d": "Rings (2D)",
    "overlap_8d": "Gaussian overlap (8D)",
    "imbalanced_16d": "Gaussian imbalanced (16D)",
}


def metric_row(
    case_id: str,
    seed: int,
    method: str,
    labels: np.ndarray,
    dataset,
    seconds: float,
    graph_seconds: float,
    profile: dict[str, object],
) -> dict[str, object]:
    metrics = evaluate(dataset, labels, "synthetic")
    return {
        "case_id": case_id,
        "seed": seed,
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
        "fit_seconds_without_graph": seconds,
        "graph_seconds": graph_seconds,
        "selected_gamma_components": profile.get("mdl_gamma_selected_components"),
        "selected_output_groups": profile.get("mdl_gamma_output_groups"),
        "background_components": profile.get("optimization_selected_local_components", [None])[-1],
    }


def draw_panel(ax, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    noise = labels < 0
    if np.any(noise):
        ax.scatter(
            coords[noise, 0],
            coords[noise, 1],
            s=1.5,
            color="#b8b8b8",
            alpha=0.22,
            linewidths=0,
            rasterized=True,
        )
    for index, label in enumerate(sorted(np.unique(labels[labels >= 0]).tolist())):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2.8,
            color=plt.get_cmap("tab20")(index % 20),
            alpha=0.75,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
        spine.set_linewidth(0.5)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "benchmarks" / "protocol.full-legacy-grid-7synthetic.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5_000)
    parser.add_argument("--seeds", default="42,73")
    parser.add_argument("--criteria", default="bic,aicc,icl")
    parser.add_argument("--cases", default="")
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--pca-seed", type=int, default=None)
    parser.add_argument("--pca-method", default="icl")
    args = parser.parse_args()

    seeds = [int(value) for value in parse_csv(args.seeds)]
    criteria = parse_csv(args.criteria)
    requested_cases = set(parse_csv(args.cases))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    rendered: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for seed in seeds:
        for case in protocol["synthetic"]["cases"]:
            if requested_cases and case["id"] not in requested_cases:
                continue
            dataset = load_synthetic({**case, "n": args.n}, seed)
            dimension = float(dataset.X.shape[1])
            backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
            graph, graph_seconds = build_knn_graph(
                dataset.X, k=32, backend=backend, n_jobs=1
            )

            baseline_started = perf_counter()
            baseline = StrataSCAN(ambient_dimension=dimension).fit_from_graph(
                graph, ambient_dimension=dimension
            )
            baseline_seconds = perf_counter() - baseline_started
            rows.append(
                metric_row(
                    case["id"],
                    seed,
                    "stratascan_0.1.2",
                    baseline.labels_,
                    dataset,
                    baseline_seconds,
                    graph_seconds,
                    baseline.profile_,
                )
            )

            candidate_labels: dict[str, np.ndarray] = {}
            for criterion in criteria:
                gamma_started = perf_counter()
                stratification = estimate_mdl_multiscale_stratification(
                    graph,
                    ambient_dimension=dimension,
                    config=GammaMDLConfig(
                        criterion=criterion,
                        n_init=args.n_init,
                        random_state=42,
                    ),
                )
                gamma_seconds = perf_counter() - gamma_started
                core_started = perf_counter()
                result = optimize_strict_core_from_graph(
                    graph,
                    stratification,
                    ambient_dimension=dimension,
                    config=OptimizationStrictCoreConfig(),
                )
                fit_seconds = gamma_seconds + perf_counter() - core_started
                method = criterion
                profile = {**stratification.diagnostics, **result.profile}
                rows.append(
                    metric_row(
                        case["id"],
                        seed,
                        method,
                        result.labels,
                        dataset,
                        fit_seconds,
                        graph_seconds,
                        profile,
                    )
                )
                candidate_labels[method] = result.labels.copy()

            if args.pca_seed == seed:
                chosen = candidate_labels[args.pca_method]
                coords = PCA(n_components=2, random_state=seed).fit_transform(dataset.X)
                rendered.append(
                    (case["id"], coords, dataset.y, baseline.labels_.copy(), chosen)
                )
                np.savez_compressed(
                    args.output / f"{case['id']}-seed{seed}-pca-labels.npz",
                    pca=coords,
                    truth=dataset.y,
                    baseline=baseline.labels_,
                    candidate=chosen,
                )
            pd.DataFrame(rows).to_csv(args.output / "metrics.checkpoint.csv", index=False)
            print(f"completed seed={seed} case={case['id']}", flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(args.output / "metrics.csv", index=False)
    baseline = scores.loc[scores.method.eq("stratascan_0.1.2")].set_index(
        ["case_id", "seed"]
    )
    comparisons = []
    for method in sorted(set(scores.method) - {"stratascan_0.1.2"}):
        candidate = scores.loc[scores.method.eq(method)].set_index(["case_id", "seed"])
        delta = candidate[["macro_target_f1", "pairwise_f1", "noise_f1"]] - baseline[
            ["macro_target_f1", "pairwise_f1", "noise_f1"]
        ]
        comparisons.append(
            {
                "method": method,
                "mean_macro_delta": delta.macro_target_f1.mean(),
                "worst_macro_delta": delta.macro_target_f1.min(),
                "mean_pairwise_delta": delta.pairwise_f1.mean(),
                "worst_pairwise_delta": delta.pairwise_f1.min(),
                "mean_noise_f1_delta": delta.noise_f1.mean(),
                "worst_noise_f1_delta": delta.noise_f1.min(),
                "mean_macro_target_f1": candidate.macro_target_f1.mean(),
                "mean_pairwise_f1": candidate.pairwise_f1.mean(),
                "mean_noise_f1": candidate.noise_f1.mean(),
            }
        )
    summary = pd.DataFrame(comparisons).sort_values(
        ["worst_macro_delta", "mean_macro_delta"], ascending=False
    )
    summary.to_csv(args.output / "candidate-summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    if rendered:
        lookup = scores.set_index(["case_id", "seed", "method"])
        fig, axes = plt.subplots(
            len(rendered), 3, figsize=(10.2, 20.0), constrained_layout=True
        )
        axes = np.atleast_2d(axes)
        for row_index, (case_id, coords, truth, old, candidate) in enumerate(rendered):
            base = lookup.loc[(case_id, args.pca_seed, "stratascan_0.1.2")]
            test = lookup.loc[(case_id, args.pca_seed, args.pca_method)]
            draw_panel(axes[row_index, 0], coords, truth, f"{DISPLAY_NAMES[case_id]} — truth")
            draw_panel(
                axes[row_index, 1],
                coords,
                old,
                f"0.1.2  macro={base.macro_target_f1:.3f}  pair={base.pairwise_f1:.3f}",
            )
            draw_panel(
                axes[row_index, 2],
                coords,
                candidate,
                f"{args.pca_method}  macro={test.macro_target_f1:.3f}  pair={test.pairwise_f1:.3f}",
            )
        fig.savefig(args.output / "pca-comparison.png", dpi=220, bbox_inches="tight")
        fig.savefig(args.output / "pca-comparison.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
