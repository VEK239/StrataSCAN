from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topology_noise_benchmark.generators import generate


METHODS = ["StrataSCAN", "HDBSCAN", "OPTICS", "DBSCAN", "kNN-DBSCAN", "SNN-DBSCAN", "VDBSCAN-2007", "kNN+Leiden"]
ALL_METHODS = METHODS + ["AMD-DBSCAN"]
COLORS = dict(zip(METHODS, plt.cm.tab10(np.arange(len(METHODS))), strict=True))
METRICS = [("hungarian_macro_target_f1", "Hungarian target F1", False), ("pairwise_f1", "Pairwise F1", False), ("noise_f1", "Background / noise F1", False), ("predicted_true_cluster_ratio", "Predicted / true cluster ratio", True)]


def summary(data: pd.DataFrame) -> pd.DataFrame:
    ok = data[data.status.eq("ok")].copy()
    return ok.groupby(["cell_id", "topology", "density_ratio", "noise_fraction", "n", "method"], as_index=False).agg(
        seeds_completed=("sampling_seed", "nunique"),
        hungarian_mean=("hungarian_macro_target_f1", "mean"), hungarian_sd=("hungarian_macro_target_f1", "std"),
        pairwise_mean=("pairwise_f1", "mean"), pairwise_sd=("pairwise_f1", "std"),
        noise_mean=("noise_f1", "mean"), noise_sd=("noise_f1", "std"),
        ratio_mean=("predicted_true_cluster_ratio", "mean"), ratio_sd=("predicted_true_cluster_ratio", "std"),
        runtime_median_s=("runtime_seconds", "median"), runtime_max_s=("runtime_seconds", "max"), peak_rss_median_mb=("peak_rss_mb", "median"),
        fragments_mean=("extra_signal_fragments", "mean"), merges_mean=("merged_predicted_clusters", "mean"),
    )


def _metric_columns(metric: str) -> tuple[str, str]:
    return {"hungarian_macro_target_f1": ("hungarian_mean", "hungarian_sd"), "pairwise_f1": ("pairwise_mean", "pairwise_sd"), "noise_f1": ("noise_mean", "noise_sd"), "predicted_true_cluster_ratio": ("ratio_mean", "ratio_sd")}[metric]


def gaussian_curve(table: pd.DataFrame, output: Path) -> None:
    subset = table[table.cell_id.eq("gaussian_noise_curve_r4")]
    levels = sorted(subset.noise_fraction.unique()); positions = {value: index for index, value in enumerate(levels)}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True)
    for method in METHODS:
        rows = subset[subset.method.eq(method)].sort_values("noise_fraction")
        x = [positions[value] for value in rows.noise_fraction]
        for axis, (metric, title, logarithmic) in zip(axes.flat, METRICS, strict=True):
            mean, sd = _metric_columns(metric)
            axis.plot(x, rows[mean], marker="o", lw=1.7, ms=4.5, color=COLORS[method], label=method)
            axis.fill_between(x, np.maximum(0, rows[mean] - rows[sd].fillna(0)), rows[mean] + rows[sd].fillna(0), color=COLORS[method], alpha=.12)
            axis.set_title(title); axis.grid(alpha=.2)
            if logarithmic: axis.set_yscale("log"); axis.axhline(1, color="black", lw=1, ls="--")
            else: axis.set_ylim(-.02, 1.04)
    for axis in axes[1]: axis.set_xlabel("Diffuse-background fraction (%)")
    for axis in axes.flat: axis.set_xticks(range(len(levels)), [f"{100*x:g}" for x in levels])
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("PILOT / four-seed compact-target noise response (mean +/- SD)", fontsize=16)
    fig.subplots_adjust(left=.07, right=.98, bottom=.14, top=.91, hspace=.28, wspace=.18)
    fig.savefig(output / "gaussian_noise_response.png", dpi=180, bbox_inches="tight"); fig.savefig(output / "gaussian_noise_response.pdf", bbox_inches="tight"); plt.close(fig)


def panel(table: pd.DataFrame, output: Path, *, cell_ids: list[str], filename: str, title: str, noise_levels: list[float] | None = None) -> None:
    subset = table[table.cell_id.isin(cell_ids)].copy()
    if noise_levels is not None:
        subset = subset[subset.noise_fraction.isin(noise_levels)]
    levels = sorted(subset.noise_fraction.unique())
    labels = {"gaussian_noise_curve_r4": "Gaussian 4x", "anisotropic_ellipses_screen": "Ellipses", "moon_arcs_screen": "Moon arcs", "rings_screen": "Rings", "gaussian_density_r16": "Gaussian 16x", "gaussian_density_r64": "Gaussian 64x"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True)
    for method in METHODS:
        rows = subset[subset.method.eq(method)]
        for axis, (metric, metric_title, logarithmic) in zip(axes.flat, METRICS, strict=True):
            mean, sd = _metric_columns(metric)
            for level_index, level in enumerate(levels):
                part = rows[np.isclose(rows.noise_fraction, level)].set_index("cell_id").reindex(cell_ids)
                x = np.arange(len(cell_ids)) + (level_index - (len(levels) - 1) / 2) * .065
                marker = ["o", "s", "^", "D", "v", "P"][level_index % 6]
                axis.plot(x, part[mean], marker=marker, ms=3.7, lw=.8, alpha=.75, color=COLORS[method])
            axis.set_title(metric_title); axis.grid(alpha=.2, axis="y")
            if logarithmic: axis.set_yscale("log"); axis.axhline(1, color="black", lw=1, ls="--")
            else: axis.set_ylim(-.02, 1.04)
    for axis in axes[1]: axis.set_xlabel("Dataset topology / density condition")
    for axis in axes.flat: axis.set_xticks(np.arange(len(cell_ids)), [labels[cell] for cell in cell_ids])
    handles = [plt.Line2D([], [], color="black", marker=["o", "s", "^", "D", "v", "P"][index], lw=.8, label=f"{100*level:g}% noise") for index, level in enumerate(levels)]
    handles += [plt.Line2D([], [], color=COLORS[method], lw=2, label=method) for method in METHODS]
    fig.legend(handles, [item.get_label() for item in handles], loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=16); fig.subplots_adjust(left=.07, right=.98, bottom=.18, top=.91, hspace=.28, wspace=.18)
    fig.savefig(output / f"{filename}.png", dpi=180, bbox_inches="tight"); fig.savefig(output / f"{filename}.pdf", bbox_inches="tight"); plt.close(fig)


def design_figure(config: dict, output: Path) -> None:
    cells = [config["cells"][0], config["cells"][1], config["cells"][2], config["cells"][3], config["cells"][5]]
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.4), sharex=True, sharey=True)
    for axis, cell in zip(axes, cells, strict=True):
        ds = generate(topology=cell["topology"], density_ratio=cell["density_ratio"], noise_fraction=.95, geometry_seed=config["geometry_seed"], sampling_seed=config["sampling_seeds"][0], mass_per_target=config["design"]["mass_per_target"], minimum_local_contrast_at_99pct=config["design"]["minimum_local_contrast_at_99pct"])
        axis.scatter(ds.X[ds.y < 0, 0], ds.X[ds.y < 0, 1], s=.15, c="#c8c8c8", rasterized=True)
        axis.scatter(ds.X[ds.y >= 0, 0], ds.X[ds.y >= 0, 1], s=2, c=ds.y[ds.y >= 0], cmap="tab10", rasterized=True)
        axis.set_title(cell["cell_id"].replace("_", "\n"), fontsize=8); axis.set_aspect("equal")
    fig.suptitle("PILOT dataset design at 95% diffuse background", fontsize=14); fig.tight_layout()
    fig.savefig(output / "topology_design.png", dpi=200, bbox_inches="tight"); fig.savefig(output / "topology_design.pdf", bbox_inches="tight"); plt.close(fig)


def resources(data: pd.DataFrame, output: Path) -> None:
    grouped = data[data.status.eq("ok")].groupby("method", as_index=False).agg(median_runtime_s=("runtime_seconds", "median"), max_runtime_s=("runtime_seconds", "max"), max_rss_mb=("peak_rss_mb", "max"), completed=("status", "size"))
    statuses = data.groupby(["method", "status"]).size().unstack(fill_value=0).reindex(ALL_METHODS, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ordered = grouped.sort_values("median_runtime_s")
    axes[0].barh(ordered.method, ordered.median_runtime_s, color=[COLORS.get(m, "#888888") for m in ordered.method]); axes[0].set_xscale("log"); axes[0].set_xlabel("Median clustering runtime, seconds (log)"); axes[0].set_title("Completed jobs"); axes[0].grid(axis="x", alpha=.2)
    left = np.zeros(len(statuses)); palette = {"ok": "#1a9850", "timeout": "#fdae61", "skipped_resource_limit": "#bdbdbd", "error": "#d73027", "worker_failure": "#d73027"}
    for state in ["ok", "timeout", "skipped_resource_limit", "error", "worker_failure"]:
        if state in statuses:
            axes[1].barh(statuses.index, statuses[state], left=left, label=state, color=palette[state]); left += statuses[state].to_numpy()
    axes[1].set_xlabel("Manifest rows"); axes[1].set_title("Completion / resource status"); axes[1].legend(frameon=False)
    fig.suptitle("PILOT runtime and completion summary", fontsize=15); fig.tight_layout(); fig.savefig(output / "resources_completion.png", dpi=180, bbox_inches="tight"); fig.savefig(output / "resources_completion.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir", type=Path); args = parser.parse_args(); output = args.run_dir.resolve()
    config = json.loads((output / "resolved_config.json").read_text(encoding="utf-8")); data = pd.read_csv(output / "results.csv"); table = summary(data); table.to_csv(output / "summary.csv", index=False)
    status = data.groupby(["cell_id", "topology", "noise_fraction", "method", "status"]).size().rename("jobs").reset_index(); status.to_csv(output / "status.csv", index=False)
    design = []
    for cell in config["cells"]:
        for fraction in cell["noise_fractions"]:
            ds = generate(topology=cell["topology"], density_ratio=cell["density_ratio"], noise_fraction=fraction, geometry_seed=config["geometry_seed"], sampling_seed=config["sampling_seeds"][0], mass_per_target=config["design"]["mass_per_target"], minimum_local_contrast_at_99pct=config["design"]["minimum_local_contrast_at_99pct"])
            m = ds.metadata; design.append({"cell_id": cell["cell_id"], "topology": cell["topology"], "density_ratio": cell["density_ratio"], "noise_fraction": fraction, "total_n": m["total_n"], "signal_count": m["signal_count"], "window_area": m["background_window_area"], "background_density": m["background_density"], "minimum_local_contrast": m["minimum_local_target_background_contrast"], "minimum_edge_gap": m["minimum_edge_gap"], "geometry_instance_id": m["geometry_instance_id"]})
    pd.DataFrame(design).to_csv(output / "design_audit.csv", index=False)
    gaussian_curve(table, output); panel(table, output, cell_ids=["anisotropic_ellipses_screen", "moon_arcs_screen", "rings_screen"], filename="topology_screen", title="PILOT / topology screen across three noise levels"); panel(table, output, cell_ids=["gaussian_noise_curve_r4", "gaussian_density_r16", "gaussian_density_r64"], filename="density_screen", title="PILOT / compact targets with extreme density differences", noise_levels=[.95, .99]); design_figure(config, output); resources(data, output)
    counts = data.status.value_counts().to_dict()
    lines = ["# PILOT / replicated topology and density stress benchmark", "", f"Manifest rows: {len(data)}. Statuses: {counts}.", "", f"Sampling seeds: {config['sampling_seeds']}; four workers; geometry seed: {config['geometry_seed']}.", "", "The Gaussian reference has six noise levels. The topology screens are deliberately separate from the density-ratio screens, so topology is not confounded with the 16x/64x density stress.", "", "All cells retain six 300-point targets and fixed support within topology. The design audit records total n, local target/background contrast, support area, background density, edge separation, and geometry identity. AMD-DBSCAN is an explicit skip in every cell.", "", "Figures are exploratory; metrics are aggregated over four sampling seeds where methods completed. Timeouts and errors remain in status tables and are not treated as missing favorable results."]
    (output / "PILOT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
