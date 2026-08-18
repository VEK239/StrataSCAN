from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[variable] = "1"

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from benchmarks.methods import run_method


METHODS = [
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "AMD-DBSCAN",
    "kNN-DBSCAN",
    "kNN+Leiden",
    "StrataSCAN",
]

CASE_LABELS = {
    "multidensity_2d": "Multidensity 2D",
    "ultrasparse_16d": "Ultrasparse 16D\n(PCA view)",
    "overlapping_density_16d": "Overlapping density 16D\n(PCA view)",
    "moons_2d": "Moons 2D",
    "rings_2d": "Rings 2D",
    "overlap_8d": "Gaussian overlap 8D\n(PCA view)",
    "imbalanced_16d": "Imbalanced 16D\n(PCA view)",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def project(X: np.ndarray, display_indices: np.ndarray, seed: int) -> np.ndarray:
    if X.shape[1] == 2:
        return np.asarray(X[display_indices], dtype=float)
    model = PCA(n_components=2, svd_solver="randomized", random_state=seed).fit(X)
    return model.transform(X[display_indices])


def ordered_cluster_colors(labels: np.ndarray, projection: np.ndarray) -> np.ndarray:
    colors = np.full((labels.size, 4), mpl.colors.to_rgba("#D3D3D3"), dtype=float)
    clusters = np.unique(labels[labels >= 0])
    if clusters.size == 0:
        return colors
    centroids = np.array([projection[labels == cluster].mean(axis=0) for cluster in clusters])
    center = np.nanmedian(centroids, axis=0)
    angles = np.arctan2(centroids[:, 1] - center[1], centroids[:, 0] - center[0])
    order = clusters[np.argsort(angles)]
    palette = mpl.colormaps["tab20"](np.linspace(0, 1, 20, endpoint=False))
    for index, cluster in enumerate(order):
        color = palette[index % len(palette)].copy()
        if index >= len(palette):
            color[:3] = 0.65 * color[:3] + 0.35
        colors[labels == cluster] = color
    return colors


def axis_limits(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.002, 0.998])
    span = max(float(high - low), 1e-6)
    return float(low - 0.03 * span), float(high + 0.03 * span)


def draw_panel(
    ax: plt.Axes,
    projection: np.ndarray,
    labels: np.ndarray,
    annotation: str,
    *,
    failed: bool = False,
    highlight: bool = False,
) -> None:
    colors = ordered_cluster_colors(labels, projection)
    noise = labels < 0
    if np.any(noise):
        ax.scatter(
            projection[noise, 0],
            projection[noise, 1],
            s=1.0,
            c=colors[noise],
            alpha=0.48,
            linewidths=0,
            rasterized=True,
        )
    if np.any(~noise):
        ax.scatter(
            projection[~noise, 0],
            projection[~noise, 1],
            s=1.45,
            c=colors[~noise],
            alpha=0.90,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xlim(axis_limits(projection[:, 0]))
    ax.set_ylim(axis_limits(projection[:, 1]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.02,
        0.02,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        fontweight="bold" if highlight else "normal",
        color="#A50026" if failed else "#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
    )


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), dpi=600)
    fig.savefig(output.with_suffix(".svg"), dpi=600)
    fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an sklearn-style clustering comparison grid")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--benchmark-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-display-points", type=int, default=8000)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="include only this protocol case; repeat to build a multi-page gallery",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="include only this method; repeat to split a legible multi-page gallery",
    )
    args = parser.parse_args()

    configure_style()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cases = protocol["synthetic"]["cases"]
    if args.case_id:
        requested = set(args.case_id)
        known = {case["id"] for case in cases}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown case IDs: {unknown}")
        cases = [case for case in cases if case["id"] in requested]
    methods = METHODS
    if args.method:
        unknown_methods = sorted(set(args.method) - set(METHODS))
        if unknown_methods:
            raise ValueError(f"unknown methods: {unknown_methods}")
        methods = [method for method in METHODS if method in set(args.method)]
    if not cases:
        raise ValueError("the behavior grid requires at least one case")
    if not methods:
        raise ValueError("the behavior grid requires at least one method")
    benchmark = pd.read_csv(args.benchmark_results)
    in_quality = args.n in protocol["synthetic"]["quality_sizes"]
    in_scaling = args.n in protocol["synthetic"]["scaling_sizes"]
    if in_quality == in_scaling:
        raise ValueError(
            f"n={args.n} must occur in exactly one of quality_sizes or scaling_sizes"
        )
    tier = "quality" if in_quality else "scaling"
    benchmark = benchmark.loc[
        (benchmark["seed"] == args.seed)
        & benchmark["dataset_id"].str.contains(fr"__{tier}__n{args.n}$")
    ].set_index(["dataset_id", "method"])

    columns = ["Ground truth", *methods]
    fig, axes = plt.subplots(
        len(cases),
        len(columns),
        figsize=(7.16, 0.62 + 1.22 * len(cases)),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.17, right=0.995, bottom=0.055, top=0.975, hspace=0.10, wspace=0.08)
    records: list[dict[str, object]] = []
    with threadpool_limits(limits=1):
        for row, case in enumerate(cases):
            spec = {**case, "n": args.n, "tier": tier}
            dataset = load_synthetic(spec, args.seed)
            display_count = min(args.max_display_points, dataset.X.shape[0])
            display_rng = np.random.default_rng(args.seed + row * 1009)
            display_indices = np.sort(
                display_rng.choice(dataset.X.shape[0], size=display_count, replace=False)
            )
            projection = project(dataset.X, display_indices, args.seed)
            truth_clusters = int(np.unique(dataset.y[dataset.y >= 0]).size)
            draw_panel(
                axes[row, 0],
                projection,
                dataset.y[display_indices],
                f"truth k={truth_clusters}\nnoise={np.mean(dataset.y < 0):.0%}",
            )
            axes[row, 0].text(
                -0.12,
                0.5,
                CASE_LABELS[case["id"]],
                transform=axes[row, 0].transAxes,
                ha="right",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
            dataset_id = f"{case['id']}__{tier}__n{args.n}"
            for column, method in enumerate(methods, start=1):
                frozen_status = "missing"
                frozen_pairwise = np.nan
                frozen_runtime = np.nan
                if (dataset_id, method) in benchmark.index:
                    frozen = benchmark.loc[(dataset_id, method)]
                    if isinstance(frozen, pd.DataFrame):
                        frozen = frozen.iloc[0]
                    frozen_status = str(frozen["status"])
                    frozen_pairwise = frozen.get("pairwise_f1", np.nan)
                    frozen_runtime = frozen.get("runtime_seconds", np.nan)
                started = perf_counter()
                failed = False
                error = ""
                if frozen_status != "ok":
                    failed = True
                    error = (
                        str(frozen.get("error", frozen_status))
                        if frozen_status != "missing"
                        else "not measured"
                    )
                    labels = np.full(dataset.X.shape[0], -1, dtype=np.int64)
                    runtime = float(frozen_runtime) if pd.notna(frozen_runtime) else np.nan
                    metrics = {"pairwise_f1": np.nan}
                    clusters = 0
                    annotation = (
                        "NOT MEASURED"
                        if frozen_status == "missing"
                        else frozen_status.replace("_", " ").upper()
                    )
                else:
                    try:
                        result = run_method(method, dataset.X, args.seed)
                        runtime = perf_counter() - started
                        labels = result.labels
                        metrics = evaluate(dataset, labels, "synthetic")
                        clusters = int(np.unique(labels[labels >= 0]).size)
                        annotation = f"F1={float(metrics['pairwise_f1']):.2f}  k={clusters}"
                    except Exception as exception:  # controlled visual evidence
                        failed = True
                        error = f"{type(exception).__name__}: {exception}"
                        labels = np.full(dataset.X.shape[0], -1, dtype=np.int64)
                        runtime = perf_counter() - started
                        metrics = {"pairwise_f1": np.nan}
                        clusters = 0
                        annotation = "FAILED"
                draw_panel(
                    axes[row, column],
                    projection,
                    labels[display_indices],
                    annotation,
                    failed=failed,
                    highlight=method == "StrataSCAN",
                )
                records.append(
                    {
                        "dataset_id": dataset_id,
                        "method": method,
                        "seed": args.seed,
                        "n": args.n,
                        "visual_run_status": "failed" if failed else "ok",
                        "visual_pairwise_f1": metrics["pairwise_f1"],
                        "visual_runtime_seconds": runtime,
                        "visual_clusters": clusters,
                        "frozen_benchmark_status": frozen_status,
                        "frozen_benchmark_pairwise_f1": frozen_pairwise,
                        "frozen_benchmark_runtime_seconds": frozen_runtime,
                        "absolute_pairwise_f1_delta": (
                            abs(float(metrics["pairwise_f1"]) - float(frozen_pairwise))
                            if pd.notna(metrics["pairwise_f1"]) and pd.notna(frozen_pairwise)
                            else np.nan
                        ),
                        "error": error,
                    }
                )
    for column, title in enumerate(columns):
        axes[0, column].set_title(
            title,
            fontsize=7.5,
            fontweight="bold",
            color="#8A5A00" if title == "StrataSCAN" else "black",
            pad=4,
        )
    fig.text(
        0.5,
        0.012,
        "Predicted clusters are colored categorically and colors are not label-matched across methods; predicted noise is light gray.\n"
        "High-dimensional datasets are shown by PCA without changing the clustering input.",
        ha="center",
        fontsize=6.5,
    )
    save(fig, args.output)
    pd.DataFrame(records).to_csv(args.output.with_name(args.output.name + "-data.csv"), index=False)
    successful_deltas = [
        float(record["absolute_pairwise_f1_delta"])
        for record in records
        if pd.notna(record["absolute_pairwise_f1_delta"])
    ]
    max_delta = max(successful_deltas, default=0.0)
    provenance = {
        "protocol": str(args.protocol.resolve()),
        "case_ids": [case["id"] for case in cases],
        "methods": methods,
        "benchmark_results": str(args.benchmark_results.resolve()),
        "n": args.n,
        "seed": args.seed,
        "tier": tier,
        "display_points_per_dataset": min(args.max_display_points, args.n),
        "display_sampling": "deterministic uniform sample used only for plotting",
        "projection": "original coordinates for 2D; deterministic PCA for dimensions > 2",
        "clustering_input": "full native-dimensional dataset",
        "annotations": "pairwise F1 and cluster count from this full recomputation; runtime retained only in the accompanying provenance CSV",
        "frozen_benchmark_comparison": "retained in the accompanying CSV; this figure does not mix frozen metrics with newly recomputed labels",
        "max_absolute_pairwise_f1_delta": max_delta,
        "formats": ["PDF", "SVG", "PNG 600 dpi"],
    }
    args.output.with_name(args.output.name + "-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
