from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_gaia
from benchmarks.evaluation import evaluate
from stratascan import StrataSCAN
from stratascan.strict_core import (
    _candidate_component_seed_mask,
    _tail_rank_window_pvalue,
)
from stratascan.uniform_tail import estimate_stratification_uniform_tail


FIELDS = ["UBC1306", "UBC1016"]
FIELD_LABELS = {
    "UBC1306": "target merged with field component",
    "UBC1016": "target assigned to noise",
}


def probe_diagnostics(field_id: str, dataset, model: StrataSCAN, strat) -> list[dict[str, object]]:
    reference = dataset.y >= 0
    groups = np.asarray(strat.groups, dtype=np.int32)
    supported = set(map(int, strat.supported_groups))
    graph = model.graph_
    d4 = graph.distances[:, model.min_samples - 2]
    d32 = graph.distances[:, model.tail_probe_outer_rank - 1]
    contrast = d4 / np.maximum(d32, np.finfo(np.float32).tiny)
    rows: list[dict[str, object]] = []
    for group in range(strat.selected_components):
        mask = groups == group
        role = "dense" if group in supported else "tail/background"
        row: dict[str, object] = {
            "field_id": field_id,
            "stratum": group,
            "role": role,
            "stars": int(mask.sum()),
            "reference_members": int(np.sum(mask & reference)),
            "reference_median_d4_d32": float(np.median(contrast[mask & reference])) if np.any(mask & reference) else np.nan,
            "stratum_median_d4_d32": float(np.median(contrast[mask])),
            "reference_contrast_percentile": (
                float(np.mean(contrast[mask] <= np.median(contrast[mask & reference])))
                if np.any(mask & reference)
                else np.nan
            ),
            "probe_runs": group not in supported,
            "probe_pvalue": np.nan,
            "probe_threshold": np.nan,
            "tail_epsilon": np.nan,
            "probe_points": 0,
            "probe_reference": 0,
            "candidate_points": 0,
            "candidate_reference": 0,
            "selected_seed_points": 0,
            "selected_seed_reference": 0,
        }
        if group not in supported:
            pvalue, component_size, _, completed = _tail_rank_window_pvalue(
                graph,
                mask,
                contrast,
                quantile=model.tail_probe_quantile,
                min_samples=model.min_samples,
            )
            epsilon = float(np.quantile(d4[mask], model.tail_core_quantile))
            threshold = float(np.quantile(contrast[mask], model.tail_probe_quantile))
            probe = mask & (contrast <= threshold)
            candidates = mask & (d4 <= epsilon)
            selected = _candidate_component_seed_mask(
                graph,
                candidates,
                probe,
                min_samples=model.min_samples,
                select_multiple=model.tail_multiple_components,
                min_component_size=model.tail_seed_min_size,
            )
            row.update(
                {
                    "probe_pvalue": pvalue,
                    "probe_component_size": component_size,
                    "probe_null_windows": completed,
                    "probe_threshold": threshold,
                    "tail_epsilon": epsilon,
                    "probe_points": int(probe.sum()),
                    "probe_reference": int(np.sum(probe & reference)),
                    "candidate_points": int(candidates.sum()),
                    "candidate_reference": int(np.sum(candidates & reference)),
                    "selected_seed_points": int(selected.sum()),
                    "selected_seed_reference": int(np.sum(selected & reference)),
                    "selected_seed_final_core": int(np.sum(selected & np.isin(np.arange(selected.size), model.core_sample_indices_))),
                    "selected_seed_final_clustered": int(np.sum(selected & (model.labels_ >= 0))),
                }
            )
        rows.append(row)
    return rows


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def best_match(labels: np.ndarray, reference: np.ndarray) -> int | None:
    best_label: int | None = None
    best_f1 = -1.0
    for label in np.unique(labels[labels >= 0]):
        selected = labels == label
        tp = int(np.sum(selected & reference))
        if tp == 0:
            continue
        precision = tp / int(selected.sum())
        recall = tp / int(reference.sum())
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_label = int(label)
    return best_label


def plot_field(
    axes: tuple[plt.Axes, plt.Axes],
    field_id: str,
    dataset,
    model: StrataSCAN,
    groups: np.ndarray,
    eps: np.ndarray,
    supported: set[int],
) -> list[dict[str, object]]:
    ax_hist, ax_scatter = axes
    reference = dataset.y >= 0
    labels = model.labels_
    matched = best_match(labels, reference)
    matched_mask = labels == matched if matched is not None else np.zeros(labels.size, dtype=bool)
    d4 = np.maximum(model.graph_.distances[:, 3], np.finfo(np.float32).tiny)
    d32 = np.maximum(model.graph_.distances[:, 31], np.finfo(np.float32).tiny)
    x = np.log10(d4)
    y = np.log10(d32)
    unique_groups = np.unique(groups)
    colors = mpl.colormaps["viridis"](np.linspace(0.12, 0.82, 4))
    color_by_group = {int(group): colors[index] for index, group in enumerate(unique_groups)}
    background_group = int(unique_groups.max())
    color_by_group[background_group] = np.array([0.62, 0.62, 0.62, 1.0])

    bins = np.linspace(float(np.quantile(x, 0.001)), float(np.quantile(x, 0.999)), 70)
    hist_groups = [x[groups == group] for group in unique_groups]
    ax_hist.hist(
        hist_groups,
        bins=bins,
        stacked=True,
        density=False,
        color=[color_by_group[int(group)] for group in unique_groups],
        alpha=0.72,
        linewidth=0,
    )
    if matched is not None:
        matched_hist, edges = np.histogram(x[matched_mask], bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax_hist.step(centers, matched_hist, where="mid", color="#D55E00", linewidth=1.25)
    ymax = ax_hist.get_ylim()[1]
    ax_hist.scatter(
        x[reference],
        np.full(reference.sum(), -0.025 * ymax),
        marker="|",
        s=26,
        color="black",
        linewidths=0.75,
        clip_on=False,
        zorder=7,
    )
    for group in unique_groups:
        value = float(eps[int(group)]) if int(group) < eps.size else 0.0
        if value > 0:
            ax_hist.scatter(
                np.log10(value),
                0.97 * ymax,
                marker="v",
                s=24,
                color=color_by_group[int(group)],
                edgecolor="black",
                linewidth=0.35,
                zorder=8,
            )
    ax_hist.set_ylim(0, ymax)
    ax_hist.set_xlabel(r"log$_{10}$ distance to 4th neighbour")
    ax_hist.set_ylabel("Number of stars")
    ax_hist.set_title(f"{field_id}: kNN-distance strata")

    rng = np.random.default_rng(20260723)
    rows = np.arange(labels.size)
    if rows.size > 9000:
        keep = rng.choice(rows[~reference], 9000 - int(reference.sum()), replace=False)
        rows = np.concatenate([keep, np.flatnonzero(reference)])
    for group in unique_groups:
        selected = rows[groups[rows] == group]
        ax_scatter.scatter(
            x[selected],
            y[selected],
            s=3,
            color=color_by_group[int(group)],
            alpha=0.28,
            linewidths=0,
            rasterized=True,
        )
    if matched is not None:
        selected = rows[matched_mask[rows]]
        ax_scatter.scatter(
            x[selected],
            y[selected],
            s=7,
            facecolors="none",
            edgecolors="#D55E00",
            alpha=0.18,
            linewidths=0.35,
            rasterized=True,
        )
    ax_scatter.scatter(
        x[reference],
        y[reference],
        s=23,
        facecolors="none",
        edgecolors="black",
        linewidths=0.65,
        zorder=8,
    )
    ax_scatter.set_xlabel(r"log$_{10}$ distance to 4th neighbour")
    ax_scatter.set_ylabel(r"log$_{10}$ distance to 32nd neighbour")
    ax_scatter.set_title(f"{field_id}: local density and tail contrast")

    metrics = evaluate(dataset, labels, "gaia")
    ax_hist.text(
        0.02,
        0.94,
        f"F1={metrics['best_cluster_f1']:.3f}; P={metrics['best_cluster_precision']:.3f}; "
        f"R={metrics['best_cluster_recall']:.3f}",
        transform=ax_hist.transAxes,
        va="top",
        fontsize=7,
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="#E1E1E1", linewidth=0.4, zorder=0)

    rows_out: list[dict[str, object]] = []
    for group in unique_groups:
        mask = groups == group
        rows_out.append(
            {
                "field_id": field_id,
                "stratum": int(group),
                "role": "dense" if int(group) in supported else "tail/background",
                "stars": int(mask.sum()),
                "reference_members": int(np.sum(mask & reference)),
                "matched_cluster_members": int(np.sum(mask & matched_mask)),
                "median_d4": float(np.median(d4[mask])),
                "epsilon": float(eps[int(group)]) if int(group) < eps.size else 0.0,
            }
        )
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    data_root = repo / "data/raw/mctnc/MCTNC_open_data_release/MCTNC_open_data_release/data"
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    summaries: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    legend_handles: list[Line2D] = []
    for row, field_id in enumerate(FIELDS):
        spec = {
            "data_root": str(data_root.relative_to(repo)),
            "field_id": field_id,
            "ruwe_max": 1.6,
            "preprocessing": "unsupervised_field",
        }
        dataset = load_gaia(spec, repo)
        model = StrataSCAN(backend="faiss_flat", n_jobs=1, ambient_dimension=5.0).fit(dataset.X)
        strat = estimate_stratification_uniform_tail(
            model.graph_,
            eps_rank=model.min_samples - 1,
            ambient_dimension=5.0,
            config=model.uniform_tail_config,
        )
        summaries.extend(
            plot_field(
                (axes[row, 0], axes[row, 1]),
                field_id,
                dataset,
                model,
                np.asarray(strat.groups),
                np.asarray(strat.eps_by_group),
                set(map(int, strat.supported_groups)),
            )
        )
        probe_rows.extend(probe_diagnostics(field_id, dataset, model, strat))
        if row == 0:
            palette = mpl.colormaps["viridis"](np.linspace(0.12, 0.82, 4))
            for index in range(3):
                legend_handles.append(
                    Line2D([0], [0], marker="s", linestyle="none", color=palette[index], label=f"Dense stratum {index}")
                )
            legend_handles.append(
                Line2D([0], [0], marker="s", linestyle="none", color="#9E9E9E", label="Tail/background stratum")
            )
    legend_handles.extend(
        [
            Line2D([0], [0], color="#D55E00", linewidth=1.3, label="Best-matched predicted cluster"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="black", label="Catalogue target"),
            Line2D([0], [0], marker="v", linestyle="none", markerfacecolor="#777777", markeredgecolor="black", label="Per-stratum core epsilon"),
        ]
    )
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=3, frameon=False)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "gaia-neighbor-distance-failures"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(args.output / "gaia-neighbor-distance-strata.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(args.output / "gaia-tail-probe-diagnostics.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(probe_rows).to_string(index=False))


if __name__ == "__main__":
    main()
