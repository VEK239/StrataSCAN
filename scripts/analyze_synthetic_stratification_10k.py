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
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from stratascan.core import StrataSCAN
from stratascan.multiscale import estimate_multiscale_stratification
from stratascan.neighbors import build_knn_graph
from stratascan.strict_core import gamma_strict_core_from_graph


SEEDS = (23, 42, 73, 101, 151)
DISPLAY_SEED = 42


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(y: np.ndarray, active: np.ndarray) -> dict[str, float]:
    signal = y >= 0
    tp = int(np.sum(signal & active))
    fp = int(np.sum(~signal & active))
    fn = int(np.sum(signal & ~active))
    tn = int(np.sum(~signal & ~active))
    signal_precision = safe_divide(tp, tp + fp)
    signal_recall = safe_divide(tp, tp + fn)
    background_recall = safe_divide(tn, tn + fp)
    signal_f1 = safe_divide(
        2 * signal_precision * signal_recall,
        signal_precision + signal_recall,
    )
    return {
        "strata_signal_precision": signal_precision,
        "strata_signal_recall": signal_recall,
        "strata_signal_f1": signal_f1,
        "strata_background_recall": background_recall,
        "strata_balanced_accuracy": 0.5 * (signal_recall + background_recall),
    }


def analyze_case(
    case: dict,
    seed: int,
    n: int,
) -> tuple[dict, list[dict], dict | None]:
    dataset = load_synthetic({**case, "n": n}, seed)
    dimension = int(dataset.X.shape[1])
    backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
    started = perf_counter()
    graph, graph_seconds = build_knn_graph(
        dataset.X, k=32, backend=backend, n_jobs=1
    )
    strat = estimate_multiscale_stratification(
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
    clustering = evaluate(dataset, result.labels, "synthetic")

    diagnostics = strat.diagnostics
    mixture_components = int(diagnostics["multiscale_components"])
    head_strata = int(diagnostics["multiscale_supported_components"])
    bics = np.asarray(diagnostics["multiscale_candidate_bics"], dtype=float)
    candidate_components = np.arange(2, 2 + bics.size)
    bic_improvements = bics[:-1] - bics[1:]
    groups = np.asarray(strat.groups)
    active = groups < head_strata
    truth_labels = np.where(dataset.y >= 0, dataset.y, int(np.max(dataset.y)) + 2)
    group_rows: list[dict] = []
    d4 = np.asarray(graph.distances[:, 3], dtype=float)
    for group in range(strat.selected_components):
        mask = groups == group
        signal = dataset.y >= 0
        group_rows.append(
            {
                "dataset_id": case["id"],
                "seed": seed,
                "stratum": group,
                "role": "dense" if group < head_strata else "combined_background",
                "size": int(np.sum(mask)),
                "fraction": float(np.mean(mask)),
                "signal_points": int(np.sum(mask & signal)),
                "signal_purity": safe_divide(
                    int(np.sum(mask & signal)), int(np.sum(mask))
                ),
                "signal_allocation": safe_divide(
                    int(np.sum(mask & signal)), int(np.sum(signal))
                ),
                "median_d4": float(np.median(d4[mask])) if np.any(mask) else np.nan,
            }
        )

    row = {
        "dataset_id": case["id"],
        "family": case["family"],
        "seed": seed,
        "n": n,
        "dimension": dimension,
        "truth_clusters": int(np.unique(dataset.y[dataset.y >= 0]).size),
        "truth_noise_fraction": float(np.mean(dataset.y < 0)),
        "mixture_components": mixture_components,
        "head_strata": head_strata,
        "operational_strata": int(strat.selected_components),
        "selected_bic": float(strat.bic),
        "bic_argmin_at_boundary": int(mixture_components == int(candidate_components[-1])),
        "last_bic_improvement": float(bic_improvements[-1]),
        "max_bic_improvement_step": int(candidate_components[np.argmax(bic_improvements)]),
        "max_bic_improvement": float(np.max(bic_improvements)),
        "background_fraction": float(diagnostics["multiscale_background_fraction"]),
        "strata_ari_truth": float(adjusted_rand_score(truth_labels, groups)),
        "strata_ami_truth": float(adjusted_mutual_info_score(truth_labels, groups)),
        "graph_seconds": float(graph_seconds),
        "analysis_seconds": float(perf_counter() - started),
        **binary_metrics(dataset.y, active),
        **{
            key: clustering[key]
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
        "candidate_components": candidate_components.tolist(),
        "candidate_bics": bics.tolist(),
        "density_gap_ratios": diagnostics["multiscale_density_gap_ratios"],
        "tail_masses": diagnostics["multiscale_tail_masses"],
        "mixture_weights": diagnostics["multiscale_weights"],
    }

    display = None
    if seed == DISPLAY_SEED:
        if dimension == 2:
            coordinates = dataset.X
            projection = "original 2D"
        else:
            coordinates = PCA(n_components=2, random_state=DISPLAY_SEED).fit_transform(
                dataset.X
            )
            projection = "PCA"
        rng = np.random.default_rng(DISPLAY_SEED)
        sample = np.sort(
            rng.choice(n, size=min(3000, n), replace=False)
        )
        display = {
            "dataset_id": case["id"],
            "projection": projection,
            "coordinates": coordinates[sample],
            "truth": dataset.y[sample],
            "groups": groups[sample],
            "head_strata": head_strata,
            "metrics": row,
        }
    return row, group_rows, display


def plot_partitions(displays: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(
        len(displays), 2, figsize=(10.5, 3.05 * len(displays)), constrained_layout=True
    )
    for index, item in enumerate(displays):
        coordinates = item["coordinates"]
        truth = item["truth"]
        groups = item["groups"]
        ax_truth, ax_strata = axes[index]
        for label in np.unique(truth):
            mask = truth == label
            if label < 0:
                ax_truth.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=2,
                    c="#b7b7b7",
                    alpha=0.30,
                    rasterized=True,
                )
            else:
                ax_truth.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=3,
                    alpha=0.70,
                    rasterized=True,
                )
        for group in np.unique(groups):
            mask = groups == group
            if group >= item["head_strata"]:
                ax_strata.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=2,
                    c="#b7b7b7",
                    alpha=0.30,
                    rasterized=True,
                )
            else:
                ax_strata.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=3,
                    alpha=0.70,
                    rasterized=True,
                )
        metric = item["metrics"]
        ax_truth.set_title(f"{item['dataset_id']} — truth ({item['projection']})")
        ax_strata.set_title(
            f"strata: {metric['head_strata']} dense + background; "
            f"mix K={metric['mixture_components']}"
        )
        for axis in (ax_truth, ax_strata):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[:].set_visible(False)
    fig.savefig(output / "partitions-seed42.png", dpi=180)
    plt.close(fig)


def plot_bic(summary: pd.DataFrame, output: Path) -> None:
    def as_list(value):
        return json.loads(value) if isinstance(value, str) else value

    datasets = summary["dataset_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12), constrained_layout=True)
    for axis, dataset_id in zip(axes.flat, datasets, strict=False):
        subset = summary.loc[summary["dataset_id"] == dataset_id]
        curves = np.vstack(subset["candidate_bics"].map(as_list))
        components = np.asarray(as_list(subset.iloc[0]["candidate_components"]))
        relative = curves - np.min(curves, axis=1, keepdims=True)
        for curve in relative:
            axis.plot(components, curve, color="#7a7a7a", alpha=0.28, linewidth=1)
        axis.plot(
            components,
            np.median(relative, axis=0),
            color="#1f77b4",
            marker="o",
            linewidth=2,
        )
        axis.set_yscale("symlog", linthresh=100)
        axis.set_title(dataset_id)
        axis.set_xlabel("mixture components K")
        axis.set_ylabel("BIC − min(BIC)")
        axis.grid(alpha=0.20)
    for axis in axes.flat[len(datasets) :]:
        axis.set_visible(False)
    fig.savefig(output / "bic-curves-5seeds.png", dpi=180)
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
        default=REPO / "results" / "stratification_10k_20260727",
    )
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    group_rows: list[dict] = []
    displays: list[dict] = []
    checkpoint = args.output / "per-seed-metrics.csv"
    completed: set[tuple[str, int]] = set()
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        rows = existing.to_dict("records")
        completed = set(zip(existing["dataset_id"], existing["seed"], strict=True))

    for case in protocol["synthetic"]["cases"]:
        for seed in SEEDS:
            if (case["id"], seed) in completed:
                continue
            row, groups, display = analyze_case(case, seed, args.n)
            rows.append(row)
            group_rows.extend(groups)
            if display is not None:
                displays.append(display)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(
                f"completed {case['id']} seed={seed}: "
                f"Kmix={row['mixture_components']} head={row['head_strata']}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(checkpoint, index=False)
    if group_rows:
        pd.DataFrame(group_rows).to_csv(args.output / "per-stratum-metrics.csv", index=False)
    aggregate_columns = [
        column
        for column in summary.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "n", "dimension"}
    ]
    aggregate = (
        summary.groupby("dataset_id", sort=False)[aggregate_columns]
        .agg(["mean", "std", "min", "max"])
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(args.output / "aggregate-metrics.csv", index=False)

    # Recreate the display payload if this was a resumed run.
    if len(displays) != len(protocol["synthetic"]["cases"]):
        displays = []
        for case in protocol["synthetic"]["cases"]:
            _, _, display = analyze_case(case, DISPLAY_SEED, args.n)
            assert display is not None
            displays.append(display)
    plot_partitions(displays, args.output)
    plot_bic(summary, args.output)


if __name__ == "__main__":
    main()
