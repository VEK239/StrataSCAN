from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from benchmarks.datasets import load_gaia
from benchmarks.evaluation import evaluate
from stratascan import StrataSCAN


FIELDS = ["UBC1170", "UBC1026", "UBC1004", "UBC1306"]
FIELD_NOTES = {
    "UBC1170": "typical: target recovered, large extra component",
    "UBC1026": "low rejection: target recovered, large extra component",
    "UBC1004": "successful separation: rich reference cluster",
    "UBC1306": "failure: reference merged into a large component",
}
COLORS = {
    "noise": "#B8B8B8",
    "largest_extra": "#4477AA",
    "other_cluster": "#228833",
    "matched": "#EE7733",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def raw_coordinates(data_root: Path, field_id: str, ruwe_max: float) -> pd.DataFrame:
    path = data_root / "raw_open_cluster_fields" / "gaia_dr3_cone_fields" / f"gaia_cone_{field_id}.csv"
    frame = pd.read_csv(path)
    columns = ["ra", "dec", "parallax", "pmra", "pmdec", "ruwe", "source_id"]
    valid = np.all(np.isfinite(frame[columns].to_numpy(dtype=np.float64)), axis=1)
    valid &= frame["ruwe"].to_numpy(float) <= ruwe_max
    return frame.loc[valid].reset_index(drop=True)


def categories(labels: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, int | None, int | None]:
    predicted = np.unique(labels[labels >= 0])
    matched = None
    best_f1 = -1.0
    for label in predicted:
        selected = labels == label
        tp = int(np.sum(selected & (truth >= 0)))
        if tp == 0:
            continue
        precision = tp / int(np.sum(selected))
        recall = tp / int(np.sum(truth >= 0))
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            matched = int(label)
    extras = [int(label) for label in predicted if int(label) != matched]
    largest_extra = max(extras, key=lambda label: int(np.sum(labels == label))) if extras else None
    out = np.full(labels.size, "noise", dtype=object)
    out[labels >= 0] = "other_cluster"
    if largest_extra is not None:
        out[labels == largest_extra] = "largest_extra"
    if matched is not None:
        out[labels == matched] = "matched"
    return out, matched, largest_extra


def plot_projection(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    category: np.ndarray,
    reference: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
) -> None:
    rng = np.random.default_rng(20260723)
    for name in ("noise", "largest_extra", "other_cluster", "matched"):
        mask = category == name
        rows = np.flatnonzero(mask)
        if rows.size > 9000:
            rows = np.sort(rng.choice(rows, 9000, replace=False))
        size = 2.3 if name in {"noise", "largest_extra"} else 4.0
        alpha = {"noise": 0.20, "largest_extra": 0.30, "other_cluster": 0.42, "matched": 0.78}[name]
        ax.scatter(x[rows], y[rows], s=size, c=COLORS[name], alpha=alpha, linewidths=0, rasterized=True)
    ax.scatter(
        x[reference], y[reference], s=20, facecolors="none", edgecolors="black",
        linewidths=0.65, marker="o", zorder=8,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.svg")
    fig.savefig(output / f"{stem}.png", dpi=600)
    plt.close(fig)


def make_figure(repo: Path, data_root: Path, output: Path) -> pd.DataFrame:
    fig = plt.figure(figsize=(9.5, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(len(FIELDS), 4, width_ratios=[1.7, 3.0, 3.0, 3.0])
    info_axes: list[plt.Axes] = []
    axes = np.empty((len(FIELDS), 3), dtype=object)
    for row in range(len(FIELDS)):
        info_axes.append(fig.add_subplot(grid[row, 0]))
        info_axes[-1].axis("off")
        for column in range(3):
            axes[row, column] = fig.add_subplot(grid[row, column + 1])
    summaries: list[dict[str, object]] = []
    for row, field_id in enumerate(FIELDS):
        spec = {
            "data_root": str(data_root.relative_to(repo)),
            "field_id": field_id,
            "ruwe_max": 1.6,
            "preprocessing": "unsupervised_field",
        }
        dataset = load_gaia(spec, repo)
        raw = raw_coordinates(data_root, field_id, 1.6)
        if len(raw) != len(dataset.y):
            raise RuntimeError(f"raw and benchmark rows do not align for {field_id}")
        model = StrataSCAN(backend="faiss_flat", n_jobs=1, ambient_dimension=5.0).fit(dataset.X)
        metrics = evaluate(dataset, model.labels_, "gaia")
        category, matched, largest_extra = categories(model.labels_, dataset.y)
        reference = dataset.y >= 0

        dec0 = float(np.median(raw["dec"]))
        sky_x = (raw["ra"].to_numpy(float) - float(np.median(raw["ra"]))) * np.cos(np.deg2rad(dec0))
        sky_y = raw["dec"].to_numpy(float) - dec0
        pca = PCA(n_components=2, svd_solver="full").fit_transform(dataset.X)
        plot_projection(
            axes[row, 0], sky_x, sky_y, category, reference,
            xlabel=r"$\Delta$RA cos(Dec) [deg]", ylabel=r"$\Delta$Dec [deg]",
        )
        plot_projection(
            axes[row, 1], raw["pmra"].to_numpy(float), raw["pmdec"].to_numpy(float),
            category, reference, xlabel=r"$\mu_{\alpha*}$ [mas yr$^{-1}$]", ylabel=r"$\mu_\delta$ [mas yr$^{-1}$]",
        )
        plot_projection(
            axes[row, 2], pca[:, 0], pca[:, 1], category, reference,
            xlabel="PC1 (robust-scaled 5D)", ylabel="PC2",
        )
        if row == 0:
            axes[row, 0].set_title("Sky position")
            axes[row, 1].set_title("Proper motion")
            axes[row, 2].set_title("Joint 5D PCA")

        largest_size = int(np.sum(model.labels_ == largest_extra)) if largest_extra is not None else 0
        largest_reference = int(np.sum((model.labels_ == largest_extra) & reference)) if largest_extra is not None else 0
        info_axes[row].text(
            0.0, 0.55,
            f"{field_id}\n{FIELD_NOTES[field_id]}\n"
            f"n={len(dataset.y):,}, reference={int(reference.sum())}, F1={metrics['best_cluster_f1']:.3f}, "
            f"BG rejection={metrics['background_rejection']:.1%}",
            transform=info_axes[row].transAxes, va="center", fontsize=7.2, fontweight="bold", wrap=True,
        )
        summaries.append(
            {
                "field_id": field_id,
                "n": len(dataset.y),
                "reference_members": int(reference.sum()),
                "best_cluster_f1": metrics["best_cluster_f1"],
                "best_cluster_precision": metrics["best_cluster_precision"],
                "best_cluster_recall": metrics["best_cluster_recall"],
                "background_rejection": metrics["background_rejection"],
                "noise": int(np.sum(model.labels_ < 0)),
                "clusters": model.n_clusters_,
                "matched_cluster": matched,
                "matched_cluster_size": int(np.sum(model.labels_ == matched)) if matched is not None else 0,
                "largest_extra_cluster": largest_extra,
                "largest_extra_size": largest_size,
                "largest_extra_reference_members": largest_reference,
            }
        )

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5, markerfacecolor=COLORS["noise"], alpha=0.5, label="StrataSCAN noise"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5, markerfacecolor=COLORS["largest_extra"], label="Largest extra component"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5, markerfacecolor=COLORS["other_cluster"], label="Other predicted clusters"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5, markerfacecolor=COLORS["matched"], label="Best-matched cluster"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6, markerfacecolor="none", markeredgecolor="black", label="Catalogue reference member"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    save(fig, output, "gaia-stratascan-field-diagnostics")
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize StrataSCAN outputs on representative Gaia fields")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    data_root = repo / "data/raw/mctnc/MCTNC_open_data_release/MCTNC_open_data_release/data"
    configure_style()
    summary = make_figure(repo, data_root, args.output)
    summary.to_csv(args.output / "field-summary.csv", index=False)
    caption = """# Figure caption

**Representative Gaia fields under StrataSCAN.** Rows show two typical fields with a large extra component (UBC1170 and UBC1026), a rich reference cluster that is cleanly separated from the field (UBC1004), and a failure in which the reference population is absorbed by a large component (UBC1306). Columns show sky position, proper-motion space and the first two principal components of the robust-scaled five-dimensional benchmark representation. Filled colors denote StrataSCAN assignments; open black circles denote catalogue reference members. “Background” in the benchmark means reference-negative field stars, not confirmed astrophysical noise, so the blue component may contain genuine field structure or unlabelled populations. Its broad extent across multiple projections, however, distinguishes it from the compact orange reference-matched component in the typical examples.
"""
    (args.output / "CAPTION.md").write_text(caption, encoding="utf-8")
    provenance = {
        "fields": FIELDS,
        "backend": "faiss_flat exact nearest neighbours",
        "stratascan_profile": "0.1.0 defaults, ambient_dimension=5",
        "projections": ["sky", "proper_motion", "PCA of robust-scaled five-dimensional benchmark input"],
        "formats": ["PDF", "SVG", "PNG 600 dpi"],
    }
    (args.output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
