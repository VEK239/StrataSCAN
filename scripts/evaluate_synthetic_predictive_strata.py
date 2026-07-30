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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from stratascan import StrataSCAN, build_knn_graph
from stratascan.multiscale import estimate_multiscale_stratification
from stratascan.predictive import estimate_predictive_multiscale_stratification
from stratascan.strict_core import gamma_strict_core_from_graph


SEEDS = (23, 42, 73, 101, 151)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def stratification_metrics(y: np.ndarray, groups: np.ndarray, head: int) -> dict[str, float]:
    signal = y >= 0
    active = groups < head
    tp = int(np.sum(signal & active))
    fp = int(np.sum(~signal & active))
    fn = int(np.sum(signal & ~active))
    tn = int(np.sum(~signal & ~active))
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    background_recall = safe_ratio(tn, tn + fp)
    return {
        "strata_signal_precision": precision,
        "strata_signal_recall": recall,
        "strata_signal_f1": safe_ratio(2 * precision * recall, precision + recall),
        "strata_background_recall": background_recall,
        "strata_balanced_accuracy": 0.5 * (recall + background_recall),
    }


def save_display_payload(
    output: Path,
    dataset_id: str,
    X: np.ndarray,
    y: np.ndarray,
    baseline_groups: np.ndarray,
    baseline_head: int,
    predictive_groups: np.ndarray,
    predictive_head: int,
) -> None:
    if X.shape[1] == 2:
        coordinates = X
        projection = "original 2D"
    else:
        coordinates = PCA(n_components=2, random_state=42).fit_transform(X)
        projection = "PCA"
    sample = np.sort(
        np.random.default_rng(42).choice(X.shape[0], min(3000, X.shape[0]), replace=False)
    )
    np.savez_compressed(
        output / f"display-{dataset_id}.npz",
        coordinates=coordinates[sample],
        truth=y[sample],
        baseline_groups=baseline_groups[sample],
        predictive_groups=predictive_groups[sample],
        baseline_head=np.array([baseline_head]),
        predictive_head=np.array([predictive_head]),
        projection=np.array([projection]),
    )


def analyze(case: dict, seed: int, n: int, output: Path) -> tuple[dict, list[dict]]:
    dataset = load_synthetic({**case, "n": n}, seed)
    dimension = int(dataset.X.shape[1])
    backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
    graph, graph_seconds = build_knn_graph(
        dataset.X, k=32, backend=backend, n_jobs=1
    )
    started = perf_counter()
    strat = estimate_predictive_multiscale_stratification(
        graph, ambient_dimension=float(dimension)
    )
    model = StrataSCAN(
        backend=backend,
        n_jobs=1,
        ambient_dimension=float(dimension),
    )
    result = gamma_strict_core_from_graph(
        graph,
        ambient_dimension=float(dimension),
        config=model._config(),
        stratification=strat,
    )
    metrics = evaluate(dataset, result.labels, "synthetic")
    diagnostics = strat.diagnostics
    mixture_components = int(diagnostics["predictive_components"])
    head = int(diagnostics["predictive_supported_components"])
    groups = np.asarray(strat.groups)

    if seed == 42:
        baseline = estimate_multiscale_stratification(
            graph, ambient_dimension=float(dimension)
        )
        save_display_payload(
            output,
            case["id"],
            dataset.X,
            dataset.y,
            baseline.groups,
            int(baseline.diagnostics["multiscale_supported_components"]),
            groups,
            head,
        )

    row = {
        "dataset_id": case["id"],
        "seed": seed,
        "n": n,
        "dimension": dimension,
        "mixture_components": mixture_components,
        "head_strata": head,
        "operational_strata": int(strat.selected_components),
        "best_predictive_components": int(diagnostics["predictive_best_components"]),
        "background_fraction": float(diagnostics["predictive_background_fraction"]),
        "posterior_entropy": float(diagnostics["predictive_mean_posterior_entropy"]),
        "selected_assignment_stability": float(
            diagnostics["predictive_assignment_stability"][mixture_components - 2]
        ),
        "selected_min_component_mass": float(
            diagnostics["predictive_minimum_component_masses"][mixture_components - 2]
        ),
        "graph_seconds": float(graph_seconds),
        "fit_seconds": float(perf_counter() - started),
        **stratification_metrics(dataset.y, groups, head),
        **{
            key: metrics[key]
            for key in (
                "macro_target_f1",
                "pairwise_f1",
                "signal_f1",
                "noise_f1",
                "binary_macro_f1",
                "background_rejection",
                "signal_coverage",
                "n_clusters",
                "noise_fraction",
                "recovered_truth_clusters_f1_080",
                "spurious_background_majority_clusters",
                "merged_predicted_clusters",
                "extra_signal_fragments",
            )
        },
        "candidate_components": diagnostics["predictive_candidate_components"],
        "validation_log_likelihood": diagnostics[
            "predictive_validation_log_likelihood"
        ],
        "validation_standard_errors": diagnostics[
            "predictive_validation_standard_errors"
        ],
        "assignment_stability": diagnostics["predictive_assignment_stability"],
        "minimum_component_masses": diagnostics[
            "predictive_minimum_component_masses"
        ],
        "within_one_se": diagnostics["predictive_within_one_se"],
        "weights": diagnostics["predictive_weights"],
        "mu": diagnostics["predictive_mu"],
        "slopes": diagnostics["predictive_slopes"],
        "rates": diagnostics["predictive_rates"],
        "density_gap_ratios": diagnostics["predictive_density_gap_ratios"],
    }
    d4 = np.asarray(graph.distances[:, 3], dtype=float)
    strata = []
    signal = dataset.y >= 0
    for group in range(strat.selected_components):
        mask = groups == group
        strata.append(
            {
                "dataset_id": case["id"],
                "seed": seed,
                "stratum": group,
                "role": "dense" if group < head else "combined_background",
                "size": int(np.sum(mask)),
                "fraction": float(np.mean(mask)),
                "signal_purity": safe_ratio(int(np.sum(mask & signal)), int(np.sum(mask))),
                "signal_allocation": safe_ratio(
                    int(np.sum(mask & signal)), int(np.sum(signal))
                ),
                "median_d4": float(np.median(d4[mask])) if np.any(mask) else np.nan,
            }
        )
    return row, strata


def plot_partitions(protocol: dict, output: Path, metrics: pd.DataFrame) -> None:
    cases = protocol["synthetic"]["cases"]
    fig, axes = plt.subplots(
        len(cases), 3, figsize=(13.2, 3.0 * len(cases)), constrained_layout=True
    )
    for row_index, case in enumerate(cases):
        payload = np.load(output / f"display-{case['id']}.npz")
        coordinates = payload["coordinates"]
        truth = payload["truth"]
        baseline = payload["baseline_groups"]
        predictive = payload["predictive_groups"]
        baseline_head = int(payload["baseline_head"][0])
        predictive_head = int(payload["predictive_head"][0])
        projection = str(payload["projection"][0])
        values = metrics.loc[
            (metrics["dataset_id"] == case["id"]) & (metrics["seed"] == 42)
        ].iloc[0]
        panels = [
            (truth, None, f"{case['id']} — truth ({projection})"),
            (baseline, baseline_head, f"current: {baseline_head} dense + background"),
            (
                predictive,
                predictive_head,
                f"predictive: K={int(values['mixture_components'])}, "
                f"{predictive_head} dense + background",
            ),
        ]
        for axis, (labels, head, title) in zip(axes[row_index], panels, strict=True):
            for label in np.unique(labels):
                mask = labels == label
                background = label < 0 if head is None else label >= head
                axis.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=2 if background else 3,
                    c="#b7b7b7" if background else None,
                    alpha=0.30 if background else 0.70,
                    rasterized=True,
                )
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[:].set_visible(False)
    fig.savefig(output / "partition-comparison-seed42.png", dpi=180)
    plt.close(fig)


def plot_selection(metrics: pd.DataFrame, output: Path) -> None:
    def decode(value):
        return json.loads(value) if isinstance(value, str) else value

    datasets = metrics["dataset_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12), constrained_layout=True)
    for axis, dataset_id in zip(axes.flat, datasets, strict=False):
        subset = metrics.loc[metrics["dataset_id"] == dataset_id]
        components = np.asarray(decode(subset.iloc[0]["candidate_components"]))
        scores = np.vstack(subset["validation_log_likelihood"].map(decode))
        errors = np.vstack(subset["validation_standard_errors"].map(decode))
        axis.errorbar(
            components,
            np.mean(scores, axis=0),
            yerr=np.sqrt(np.mean(errors**2, axis=0)),
            marker="o",
            linewidth=2,
            capsize=3,
        )
        selected = subset["mixture_components"].astype(int).to_numpy()
        for component in np.unique(selected):
            axis.axvline(component, color="#7a7a7a", alpha=0.15 + 0.12 * np.sum(selected == component))
        axis.set_title(dataset_id)
        axis.set_xlabel("components K; vertical lines = selected seeds")
        axis.set_ylabel("held-out log likelihood / point")
        axis.grid(alpha=0.20)
    for axis in axes.flat[len(datasets) :]:
        axis.set_visible(False)
    fig.savefig(output / "predictive-selection-5seeds.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "benchmarks" / "protocol.full-legacy-grid-7synthetic.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "predictive_strata_10k_20260727",
    )
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "per-seed-metrics.csv"
    strata_checkpoint = args.output / "per-stratum-metrics.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    strata_rows = (
        pd.read_csv(strata_checkpoint).to_dict("records")
        if strata_checkpoint.exists()
        else []
    )
    completed = {(str(row["dataset_id"]), int(row["seed"])) for row in rows}
    for case in protocol["synthetic"]["cases"]:
        for seed in SEEDS:
            if (case["id"], seed) in completed:
                continue
            row, strata = analyze(case, seed, args.n, args.output)
            rows.append(row)
            strata_rows.extend(strata)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            pd.DataFrame(strata_rows).to_csv(strata_checkpoint, index=False)
            print(
                f"completed {case['id']} seed={seed}: "
                f"K={row['mixture_components']} head={row['head_strata']} "
                f"macro_f1={row['macro_target_f1']:.3f}",
                flush=True,
            )

    metrics = pd.DataFrame(rows)
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "n", "dimension"}
    ]
    aggregate = metrics.groupby("dataset_id", sort=False)[numeric].agg(
        ["mean", "std", "min", "max"]
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(args.output / "aggregate-metrics.csv", index=False)

    baseline_path = REPO / "results" / "stratification_10k_20260727" / "per-seed-metrics.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        compare_columns = [
            "dataset_id",
            "seed",
            "mixture_components",
            "head_strata",
            "strata_signal_precision",
            "strata_signal_recall",
            "strata_background_recall",
            "macro_target_f1",
            "pairwise_f1",
            "background_rejection",
            "signal_coverage",
            "n_clusters",
        ]
        comparison = baseline[compare_columns].merge(
            metrics[compare_columns],
            on=["dataset_id", "seed"],
            suffixes=("_current", "_predictive"),
        )
        for metric in compare_columns[2:]:
            comparison[f"delta_{metric}"] = (
                comparison[f"{metric}_predictive"]
                - comparison[f"{metric}_current"]
            )
        comparison.to_csv(args.output / "comparison-with-current.csv", index=False)

    plot_partitions(protocol, args.output, metrics)
    plot_selection(metrics, args.output)


if __name__ == "__main__":
    main()
