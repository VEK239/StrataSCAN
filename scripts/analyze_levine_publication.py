from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

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

from benchmarks.datasets import load_cytometry
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
DISPLAY_METHODS = [
    "Ground truth",
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
METRIC_COLUMNS = [
    "macro_target_f1",
    "pairwise_f1",
    "noise_f1",
    "background_rejection",
    "signal_coverage",
    "n_clusters",
    "truth_clusters",
    "mean_target_fragments",
    "merged_predicted_clusters",
    "spurious_background_majority_clusters",
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.svg")
    fig.savefig(directory / f"{stem}.png", dpi=600)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(results_path: Path, protocol_path: Path, repo: Path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    specs = [item for item in protocol["cytometry"]["datasets"] if item["id"] == "levine"]
    if len(specs) != 1:
        raise ValueError("expected one Levine cytometry specification")
    dataset = load_cytometry(specs[0], repo)
    results = pd.read_csv(results_path)
    results = results.loc[
        results["suite"].eq("cytometry") & results["dataset_id"].eq("levine")
    ].copy()
    if len(results) != 13 or set(results["method"]) != set(METHODS):
        raise ValueError("expected 13 frozen Levine jobs spanning all nine methods")
    selected = []
    for method in METHODS:
        part = results.loc[results["method"].eq(method)]
        if 42 in set(part["seed"]):
            selected.append(part.loc[part["seed"].eq(42)].iloc[0])
        else:
            selected.append(part.iloc[0])
    selected_results = pd.DataFrame(selected).set_index("method").reindex(METHODS)
    return dataset, results, selected_results, specs[0]


def rerun_successful_methods(
    dataset,
    selected_results: pd.DataFrame,
    cache: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    cache.mkdir(parents=True, exist_ok=True)
    labels_by_method: dict[str, np.ndarray] = {}
    verification: dict[str, dict[str, object]] = {}
    for method in METHODS:
        frozen = selected_results.loc[method]
        if frozen["status"] != "ok":
            verification[method] = {
                "status": str(frozen["status"]),
                "rerun": False,
                "reason": str(frozen.get("error", "controlled benchmark failure")),
            }
            continue
        path = cache / f"{method.replace('+', '-plus-').replace('/', '-')}-seed42.npz"
        started = perf_counter()
        if path.exists():
            labels = np.load(path)["labels"].astype(np.int64, copy=False)
            source = "cache"
        else:
            print(f"Running {method} on Levine (seed 42)...", flush=True)
            if method == "SNN-DBSCAN":
                from baselines import warm_numba_kernels

                warm_numba_kernels()
            with threadpool_limits(limits=1):
                result = run_method(method, dataset.X, seed=42)
            labels = np.asarray(result.labels, dtype=np.int64)
            np.savez_compressed(path, labels=labels)
            source = "fresh_rerun"
        if labels.shape != dataset.y.shape:
            raise ValueError(f"cached labels for {method} have the wrong shape")
        metrics = evaluate(dataset, labels, "cytometry")
        comparisons = {}
        for metric in ("macro_target_f1", "pairwise_f1", "noise_f1"):
            frozen_value = float(frozen[metric])
            rerun_value = float(metrics[metric])
            comparisons[metric] = {
                "frozen": frozen_value,
                "rerun": rerun_value,
                "absolute_delta": abs(rerun_value - frozen_value),
            }
        max_delta = max(item["absolute_delta"] for item in comparisons.values())
        if max_delta > 1e-9:
            raise ValueError(f"{method} rerun does not reproduce frozen metrics; max delta={max_delta}")
        labels_by_method[method] = labels
        verification[method] = {
            "status": "ok",
            "rerun": True,
            "source": source,
            "elapsed_seconds": perf_counter() - started,
            "metrics": comparisons,
        }
        print(f"Verified {method}; max metric delta={max_delta:.3g}", flush=True)
    return labels_by_method, verification


def stratified_indices(y: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = []
    for label in np.unique(y):
        available = np.flatnonzero(y == label)
        limit = 20000 if label < 0 else 2500
        if len(available) > limit:
            available = rng.choice(available, size=limit, replace=False)
        groups.append(available)
    indices = np.concatenate(groups)
    rng.shuffle(indices)
    return indices


def get_projection(dataset, cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = cache / "pca-display-seed42.npz"
    if path.exists():
        stored = np.load(path)
        return stored["indices"], stored["projection"], stored["explained_variance_ratio"]
    print("Fitting common two-dimensional PCA projection...", flush=True)
    indices = stratified_indices(dataset.y)
    with threadpool_limits(limits=1):
        model = PCA(n_components=2, svd_solver="randomized", random_state=42).fit(dataset.X)
        projection = model.transform(dataset.X[indices])
    variance = model.explained_variance_ratio_
    np.savez_compressed(
        path,
        indices=indices,
        projection=projection.astype(np.float32),
        explained_variance_ratio=variance,
    )
    return indices, projection, variance


def target_palette(names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    colors = mpl.colormaps["tab20"](np.linspace(0, 1, 20, endpoint=False))
    order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7]
    return {name: tuple(colors[order[index]]) for index, name in enumerate(names)}


def truth_colors(y: np.ndarray, names: list[str], palette: dict[str, tuple]) -> np.ndarray:
    colors = np.tile(mpl.colors.to_rgba("#D9D9D9"), (len(y), 1))
    for label, name in enumerate(names):
        colors[y == label] = palette[name]
    return colors


def recovered_colors(
    labels: np.ndarray,
    target_matches: list[dict[str, object]],
    palette: dict[str, tuple],
) -> np.ndarray:
    colors = np.tile(mpl.colors.to_rgba("#B8B8B8"), (len(labels), 1))
    colors[labels < 0] = mpl.colors.to_rgba("#E5E5E5")
    candidates: dict[int, dict[str, object]] = {}
    for item in target_matches:
        cluster = item["matched_predicted_cluster"]
        if cluster is None:
            continue
        cluster = int(cluster)
        if cluster not in candidates or float(item["f1"]) > float(candidates[cluster]["f1"]):
            candidates[cluster] = item
    for cluster, item in candidates.items():
        colors[labels == cluster] = palette[str(item["target"])]
    return colors


def limits(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.002, 0.998])
    span = max(float(high - low), 1e-6)
    return float(low - 0.03 * span), float(high + 0.03 * span)


def scatter_panel(ax: plt.Axes, projection: np.ndarray, colors: np.ndarray) -> None:
    ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=colors,
        s=1.1,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlim(limits(projection[:, 0]))
    ax.set_ylim(limits(projection[:, 1]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_pca(
    dataset,
    selected_results: pd.DataFrame,
    labels_by_method: dict[str, np.ndarray],
    indices: np.ndarray,
    projection: np.ndarray,
    variance: np.ndarray,
    figures: Path,
) -> None:
    palette = target_palette(dataset.target_names)
    fig, axes = plt.subplots(2, 5, figsize=(10.2, 5.7), constrained_layout=True)
    for panel_index, (method, ax) in enumerate(zip(DISPLAY_METHODS, axes.flat)):
        letter = chr(ord("a") + panel_index)
        ax.text(-0.04, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
        if method == "Ground truth":
            colors = truth_colors(dataset.y[indices], dataset.target_names, palette)
            scatter_panel(ax, projection, colors)
            ax.set_title("Expert annotation")
            annotation = f"{len(dataset.target_names)} populations + background"
        else:
            frozen = selected_results.loc[method]
            ax.set_title(method, fontweight="bold" if method == "StrataSCAN" else "normal")
            if method not in labels_by_method:
                ax.set_facecolor("#F5F5F5")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                status = str(frozen["status"]).replace("_", " ")
                reason = str(frozen.get("error", ""))
                if status == "timeout":
                    detail = "600 s wall-time limit"
                elif status == "memory limit":
                    detail = "8,192 MB RSS limit"
                elif method == "AMD-DBSCAN":
                    detail = "quadratic distance matrix"
                else:
                    detail = reason[:48]
                ax.text(0.5, 0.53, status.upper(), ha="center", va="center", color="#A50026", fontweight="bold")
                ax.text(0.5, 0.43, detail, ha="center", va="center", fontsize=6.5, color="#555555")
                annotation = "No cell-level assignments"
            else:
                labels = labels_by_method[method]
                metrics = evaluate(dataset, labels, "cytometry")
                colors = recovered_colors(labels[indices], metrics["target_matches"], palette)
                scatter_panel(ax, projection, colors)
                annotation = f"macro F1={metrics['macro_target_f1']:.3f}; K={metrics['n_clusters']}"
        ax.text(
            0.02,
            0.02,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
        )
    handles = [
        mpl.lines.Line2D([], [], linestyle="", marker="o", markersize=4, color=palette[name], label=name)
        for name in dataset.target_names
    ]
    handles.append(
        mpl.lines.Line2D([], [], linestyle="", marker="o", markersize=4, color="#D9D9D9", label="background / unmatched")
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Levine cytometry: common PCA view and population-matched cluster assignments",
        y=1.02,
    )
    save(fig, figures, "fig1-levine-pca-cluster-recovery")


def selected_target_rows(selected_results: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        frozen = selected_results.loc[method]
        parsed = json.loads(frozen["target_matches_json"]) if frozen["status"] == "ok" else []
        lookup = {str(item["target"]): item for item in parsed}
        for target in names:
            item = lookup.get(target, {})
            rows.append(
                {
                    "method": method,
                    "target": target,
                    "target_size": item.get("target_size", np.nan),
                    "f1": float(item.get("f1", 0.0)),
                    "precision": float(item.get("precision", 0.0)),
                    "recall": float(item.get("recall", 0.0)),
                    "fragments": float(item.get("fragments", 0.0)),
                    "status": str(frozen["status"]),
                }
            )
    return pd.DataFrame(rows)


def _heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    cmap: str | mpl.colors.Colormap,
    vmax: float,
    statuses: pd.Series,
    value_format: str = ".2f",
) -> mpl.image.AxesImage:
    values = table.to_numpy(float)
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=55, ha="right")
    method_labels = [method + ("†" if statuses[method] != "ok" else "") for method in METHODS]
    ax.set_yticks(range(len(METHODS)), method_labels)
    ax.tick_params(length=0)
    ax.set_title(title)
    for i, method in enumerate(METHODS):
        for j in range(len(table.columns)):
            value = values[i, j]
            color = "white" if value >= 0.62 * vmax else "black"
            ax.text(j, i, format(value, value_format), ha="center", va="center", fontsize=5.3, color=color)
    ax.axhline(METHODS.index("StrataSCAN") - 0.5, color="black", linewidth=1.0)
    return image


def fig_recovery(targets: pd.DataFrame, selected_results: pd.DataFrame, figures: Path) -> None:
    statuses = selected_results["status"]
    fig, axes = plt.subplots(3, 1, figsize=(8.7, 8.2), constrained_layout=True)
    specs = [
        ("f1", "a", "Best-matched cluster F1", "Blues", 1.0, ".2f"),
        ("recall", "b", "Recovered fraction of each annotated population", "Greens", 1.0, ".2f"),
        ("fragments", "c", "Substantial predicted fragments (≥5% of target)", "magma_r", max(1.0, targets["fragments"].quantile(0.98)), ".0f"),
    ]
    for ax, (metric, label, title, cmap, vmax, value_format) in zip(axes, specs):
        table = targets.pivot(index="method", columns="target", values=metric).reindex(index=METHODS)
        image = _heatmap(ax, table, title, cmap, float(vmax), statuses, value_format)
        cbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012)
        cbar.set_label(metric.capitalize())
        ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.suptitle("Levine population-level cluster recovery", y=1.01)
    fig.text(0.5, -0.01, "† controlled failure; recovery set to zero", ha="center", fontsize=7)
    save(fig, figures, "fig2-levine-population-recovery")


def write_table_variants(table: pd.DataFrame, stem: Path, index: bool = True) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.with_suffix(".csv"), index=index)
    display = table.reset_index() if index else table.copy()
    rows = [
        "| " + " | ".join(map(str, display.columns)) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = ["—" if pd.isna(value) else str(value).replace("|", "\\|") for value in row]
        rows.append("| " + " | ".join(values) + " |")
    stem.with_suffix(".md").write_text("\n".join(rows), encoding="utf-8")
    def escape(value: object) -> str:
        text = "--" if pd.isna(value) else str(value)
        mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
        return "".join(mapping.get(character, character) for character in text)
    latex = [r"\begin{tabular}{l" + "r" * (len(display.columns) - 1) + "}", r"\toprule"]
    latex.append(" & ".join(escape(value) for value in display.columns) + r" \\")
    latex.append(r"\midrule")
    for row in display.itertuples(index=False, name=None):
        latex.append(" & ".join(escape(value) for value in row) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    stem.with_suffix(".tex").write_text("\n".join(latex), encoding="utf-8")


def make_tables(
    selected_results: pd.DataFrame,
    targets: pd.DataFrame,
    tables: Path,
) -> pd.DataFrame:
    overview_rows = []
    for method in METHODS:
        row = selected_results.loc[method]
        overview = {"method": method, "status": row["status"]}
        for metric in METRIC_COLUMNS:
            overview[metric] = 0.0 if row["status"] != "ok" and metric in METRIC_COLUMNS[:5] else row.get(metric, np.nan)
        overview["runtime_or_termination_seconds"] = row["runtime_seconds"] if row["status"] == "ok" else row["process_runtime_seconds"]
        overview["process_peak_rss_mb"] = row["process_peak_rss_mb"]
        overview_rows.append(overview)
    overview = pd.DataFrame(overview_rows).set_index("method")
    numeric = overview.select_dtypes(include=[np.number]).columns
    overview[numeric] = overview[numeric].round(5)
    write_table_variants(overview, tables / "table1-levine-method-overview")
    write_table_variants(targets.round(5), tables / "table2-levine-target-recovery", index=False)
    matrix = targets.pivot(index="method", columns="target", values="f1").reindex(index=METHODS)
    write_table_variants(matrix.round(4), tables / "tableS1-levine-target-f1-matrix")
    return overview


def write_text(
    dataset,
    overview: pd.DataFrame,
    targets: pd.DataFrame,
    selected_results: pd.DataFrame,
    variance: np.ndarray,
    output: Path,
) -> None:
    best_method = overview["macro_target_f1"].astype(float).idxmax()
    best_score = float(overview.loc[best_method, "macro_target_f1"])
    strata = overview.loc["StrataSCAN"]
    strata_targets = targets.loc[targets["method"].eq("StrataSCAN")].sort_values("f1", ascending=False)
    top = ", ".join(f"{row.target} ({row.f1:.3f})" for row in strata_targets.head(5).itertuples())
    zero = strata_targets.loc[strata_targets["f1"].eq(0), "target"].tolist()
    completed = overview.index[overview["status"].eq("ok")].tolist()
    failed = overview.loc[~overview["status"].eq("ok"), "status"].to_dict()
    text = f"""# Levine cytometry: publication-ready analysis

## Scope

The final cytometry analysis is restricted to the Levine dataset ({dataset.X.shape[0]:,} cells, {dataset.X.shape[1]} type markers, {len(dataset.target_names)} annotated signal populations, and an unassigned background class). All nine benchmark algorithms remain in the comparison. Cell-level cluster assignments were reproduced at the frozen seed (42) only for methods that completed the original benchmark, and the reproduced quality metrics were required to match the frozen records within 1e-9.

## Results — manuscript text

On Levine, {best_method} achieved the highest failure-penalized macro target F1 ({best_score:.3f}). The methods completing the benchmark were {', '.join(completed)}; the remaining controlled outcomes were {json.dumps(failed, sort_keys=True)}. No method recovered an annotated population at F1 ≥ 0.80.

StrataSCAN completed all three prespecified seeds and, at the representative seed 42, produced {int(float(strata['n_clusters']))} clusters for 14 annotated populations. Its macro target F1 was {float(strata['macro_target_f1']):.3f}, pairwise F1 was {float(strata['pairwise_f1']):.3f}, background-classification F1 was {float(strata['noise_f1']):.3f}, background rejection was {float(strata['background_rejection']):.3f}, and signal coverage was {float(strata['signal_coverage']):.3f}. The best recovered populations were {top}. Populations with zero best-cluster F1 were {', '.join(zero) if zero else 'none'}.

The PCA display uses the same two-dimensional projection for expert annotations and every algorithm (PC1 {100 * variance[0]:.1f}% and PC2 {100 * variance[1]:.1f}% of total variance). PCA was used only for visualization; clustering and all recovery metrics were calculated in the full 32-dimensional marker space. Predicted clusters are colored by the annotated population for which they provide the highest matched-cluster F1; unmatched clusters and predicted noise remain gray. Panels for controlled failures contain no inferred assignments.

## Interpretation

The Levine result supports a narrow conclusion: among the frozen profiles, StrataSCAN gave the strongest equal-weight population recovery while completing reliably, but absolute recovery remained low and the partition was strongly over-segmented. The population-level heatmaps should therefore accompany the aggregate macro F1. The analysis does not support a claim of automated replacement for expert gating or a universal advantage across cytometry datasets.

## Methods — manuscript text

Levine type-marker intensities were transformed using arcsinh(x/5) and robustly scaled marker-wise. Nine clustering algorithms were evaluated using fixed, predeclared profiles under one CPU thread, a 600 s wall-time limit, and an 8,192 MB resident-memory limit. Macro target F1 was computed by matching each annotated population independently to the predicted cluster with the highest F1 and averaging equally across populations. Pairwise F1, background-classification F1, background rejection, signal coverage, fragmentation, merging, and background-majority cluster counts were secondary diagnostics. Controlled failures contributed zero to quality summaries. For the PCA figure, all cells were used to fit PCA and a reproducible stratified subset was displayed to retain rare populations; PCA coordinates did not enter clustering or scoring.
"""
    (output / "LEVINE_ANALYSIS.md").write_text(text, encoding="utf-8")
    captions = """# Figure captions

## Figure 1 — Levine PCA and population-matched cluster assignments

Common two-dimensional PCA representation of expert annotations (a) and the nine frozen clustering algorithms (b–j). PCA was fitted to arcsinh-transformed, robust-scaled type markers and used only for visualization; clustering and scoring used the full 32-dimensional marker space. A reproducible stratified subset is shown to preserve rare populations. For completed methods, each predicted cluster is colored according to the annotated population for which it yields the highest matched-cluster F1; unmatched clusters and predicted noise are gray. Panels without points denote controlled benchmark failures and do not impute cell assignments.

## Figure 2 — Levine population-level recovery

Best-matched cluster F1 (a), recall of each annotated population (b), and number of substantial predicted fragments containing at least 5% of the target population (c). A dagger marks methods that did not complete the frozen benchmark; their recovery values remain zero. The horizontal rule highlights StrataSCAN.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Levine-only publication analysis")
    parser.add_argument("results", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    repo = args.repo.resolve()
    configure_style()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    tables = args.output / "tables"
    cache = args.output / "cache"
    dataset, all_results, selected_results, spec = load_inputs(args.results, args.protocol, repo)
    labels_by_method, verification = rerun_successful_methods(dataset, selected_results, cache)
    indices, projection, variance = get_projection(dataset, cache)
    fig_pca(dataset, selected_results, labels_by_method, indices, projection, variance, figures)
    targets = selected_target_rows(selected_results, dataset.target_names)
    fig_recovery(targets, selected_results, figures)
    overview = make_tables(selected_results, targets, tables)
    write_text(dataset, overview, targets, selected_results, variance, args.output)
    (args.output / "RERUN_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8"
    )
    provenance = {
        "input_results": str(args.results.resolve()),
        "input_results_sha256": sha256(args.results),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "dataset": spec,
        "frozen_levine_jobs": int(len(all_results)),
        "representative_seed": 42,
        "displayed_cells": int(len(indices)),
        "pca_explained_variance_ratio": variance.tolist(),
        "successful_methods_rerun": sorted(labels_by_method),
        "failed_methods_not_imputed": sorted(set(METHODS) - set(labels_by_method)),
        "figure_formats": ["PDF", "SVG", "PNG 600 dpi"],
    }
    (args.output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
