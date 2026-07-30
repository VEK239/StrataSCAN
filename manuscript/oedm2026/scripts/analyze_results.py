from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, wilcoxon


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
SUPPLEMENT_FIGURES = ROOT / "supplement" / "figures"

PUBLICATION = REPO / "results/runs/publication_full_7synthetic_20260722/results.csv"
SAMUSIK_01 = REPO / "results/runs/samusik_01/results.csv"
SAMUSIK_ALL = REPO / "results/runs/samusik_all_samples_02_10/results.csv"
PILOT_BIO = REPO / "results/runs/optimization_dev10_real_pilot_cytometry/results.csv"
PILOT_GAIA = REPO / "results/runs/optimization_dev10_real_pilot_gaia24/results.csv"
LOCKED_SYNTH = REPO / "results/runs/optimization_dev10_locked_n5000_seeds197_251_k1/metrics.csv"

DEV = "MDL-StrataSCAN"
PREDECESSOR = "StrataSCAN 0.1.2"
DISPLAY = {
    "StrataSCAN-Optimization": DEV,
    "StrataSCAN": PREDECESSOR,
    "StrataSCAN-PredictiveMultiscale": PREDECESSOR,
    "StrataSCAN-0.1.2": PREDECESSOR,
    "stratascan_0.1.2": PREDECESSOR,
    "icl": DEV,
    "kNN+Leiden": "kNN + Leiden",
}
METHOD_ORDER = [
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "AMD-DBSCAN",
    "kNN-DBSCAN",
    "kNN + Leiden",
    PREDECESSOR,
    DEV,
]
COMMON_BIO = [
    "DBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "kNN-DBSCAN",
    "kNN + Leiden",
    PREDECESSOR,
    DEV,
]
PALETTE = {
    DEV: "#0072B2",
    PREDECESSOR: "#D55E00",
    "baseline": "#8A93A0",
    "gain": "#009E73",
    "loss": "#CC79A7",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.35,
        }
    )


def read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method"] = frame["method"].replace(DISPLAY)
    return frame


def penalize_failures(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    result = frame.copy()
    failed = ~result["status"].eq("ok")
    for metric in metrics:
        if metric in result:
            result[metric] = pd.to_numeric(result[metric], errors="coerce")
            result.loc[failed, metric] = 0.0
    return result


def collapse(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    frame = penalize_failures(frame, metrics)
    keep = [column for column in metrics if column in frame.columns]
    return (
        frame.groupby(["dataset_id", "method"], as_index=False)[keep]
        .median(numeric_only=True)
    )


def add_target_background_hmean_f1(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    target = result["macro_target_f1"].astype(float)
    noise = result["noise_f1"].astype(float)
    total = target + noise
    result["target_background_hmean_f1"] = np.where(
        total > 0.0, 2.0 * target * noise / total, 0.0
    )
    result["noise_aware_macro_f1"] = result["target_background_hmean_f1"]
    return result


def study_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Average samples within source study before cross-study summaries."""
    result = frame.copy()
    result["study"] = np.where(
        result["dataset_id"].str.startswith("samusik_"),
        "Samusik",
        result["dataset_id"].str.capitalize(),
    )
    numeric = result.select_dtypes(include=[np.number]).columns.tolist()
    return result.groupby(["study", "method"], as_index=False)[numeric].mean()


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260729) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(20_000, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def paired_summary(left: pd.Series, right: pd.Series) -> dict[str, float | int]:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    nonzero = delta[~np.isclose(delta, 0.0)]
    p_wilcoxon = float(wilcoxon(delta, zero_method="pratt").pvalue)
    improved = int(np.sum(delta > 1e-12))
    regressed = int(np.sum(delta < -1e-12))
    sign_p = float(binomtest(improved, improved + regressed, 0.5).pvalue)
    lo, hi = bootstrap_mean_ci(delta)
    return {
        "n": int(delta.size),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci_low": lo,
        "ci_high": hi,
        "improved": improved,
        "tied": int(delta.size - improved - regressed),
        "regressed": regressed,
        "wilcoxon_p": p_wilcoxon,
        "sign_p": sign_p,
        "min_delta": float(delta.min()),
        "max_delta": float(delta.max()),
    }


def load_blocks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_bio = [
        "n",
        "macro_target_f1",
        "noise_f1",
        "pairwise_f1",
        "background_rejection",
        "target_success_rate_f1_080",
        "runtime_seconds",
    ]
    metrics_gaia = [
        "n",
        "best_cluster_f1",
        "pairwise_f1",
        "background_rejection",
        "runtime_seconds",
    ]

    published = read(PUBLICATION)
    canonical = collapse(
        published.loc[published["suite"].eq("cytometry")], metrics_bio
    )
    samusik = pd.concat([read(SAMUSIK_01), read(SAMUSIK_ALL)], ignore_index=True)
    samusik = collapse(samusik, metrics_bio)
    legacy_bio = pd.concat([canonical, samusik], ignore_index=True)

    pilot_bio = collapse(read(PILOT_BIO), metrics_bio)
    predecessor = pilot_bio.loc[pilot_bio["method"].eq(PREDECESSOR)]
    dev = pilot_bio.loc[pilot_bio["method"].eq(DEV)]
    # The seven-method screened Samusik panel has complete 13-dataset coverage.
    shared_legacy = legacy_bio.loc[
        legacy_bio["method"].isin(set(COMMON_BIO) - {PREDECESSOR, DEV})
    ]
    biological = pd.concat([shared_legacy, predecessor, dev], ignore_index=True)
    biological = add_target_background_hmean_f1(biological)

    fields = set(read(PILOT_GAIA)["dataset_id"])
    legacy_gaia = collapse(
        published.loc[
            published["suite"].eq("gaia") & published["dataset_id"].isin(fields)
        ],
        metrics_gaia,
    )
    pilot_gaia = collapse(read(PILOT_GAIA), metrics_gaia)
    legacy_gaia = legacy_gaia.loc[~legacy_gaia["method"].eq(PREDECESSOR)]
    gaia = pd.concat([legacy_gaia, pilot_gaia], ignore_index=True)

    synthetic = read(LOCKED_SYNTH)
    synthetic["method"] = synthetic["method"].replace(DISPLAY)
    synthetic["dataset_id"] = synthetic["case_id"] + "__seed" + synthetic["seed"].astype(str)
    return biological, gaia, synthetic


def method_summary(frame: pd.DataFrame, primary: str, block: str) -> pd.DataFrame:
    rows = []
    for method, part in frame.groupby("method"):
        primary_values = part[primary].astype(float)
        rows.append(
            {
                "block": block,
                "method": method,
                "datasets": int(part["dataset_id"].nunique()),
                "mean_primary": float(primary_values.mean()),
                "median_primary": float(primary_values.median()),
                "success_rate_080": float((primary_values >= 0.8).mean()),
                "mean_pairwise_f1": float(part["pairwise_f1"].mean()),
                "mean_background_rejection": float(part["background_rejection"].mean()),
                "median_runtime_seconds": float(part["runtime_seconds"].median()),
            }
        )
    return pd.DataFrame(rows)


def write_tex_table(summary: pd.DataFrame, path: Path, methods: list[str]) -> None:
    order = {method: index for index, method in enumerate(methods)}
    table = summary.loc[summary["method"].isin(methods)].copy()
    table["order"] = table["method"].map(order)
    table = table.sort_values("order")
    best = table["mean_primary"].max()
    second = table.loc[table["mean_primary"] < best - 1e-12, "mean_primary"].max()
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & $N$ & Score $\uparrow$ & Pairwise $\uparrow$ & BG reject. $\uparrow$ & Time (s) $\downarrow$ \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        name = str(row.method).replace("_", r"\_")
        primary = f"{row.mean_primary:.3f}"
        if np.isclose(row.mean_primary, best):
            primary = r"\textbf{" + primary + "}"
        elif np.isclose(row.mean_primary, second):
            primary = r"\underline{" + primary + "}"
        lines.append(
            f"{name} & {row.datasets:d} & {primary} & {row.mean_pairwise_f1:.3f} & "
            f"{row.mean_background_rejection:.3f} & {row.median_runtime_seconds:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str, directory: Path = FIGURES) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(directory / f"{stem}.{extension}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_method_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.12, 1.75), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.01, 0.55, 0.145, 0.30, "Sparse geometry", "$k$NN graph;\nshell volumes"),
        (0.18, 0.55, 0.145, 0.30, "Density model", "Constrained Gamma\nmixture; ICL"),
        (0.35, 0.55, 0.145, 0.30, "Semantic strata", "Largest log-rate gap;\nexplicit background"),
        (0.52, 0.55, 0.145, 0.30, "Core optimization", "Exact observed-$d_4$\nMDL event sweep"),
        (0.69, 0.55, 0.145, 0.30, "Background scan", "Dyadic windows;\nPoisson/Beta code"),
        (0.845, 0.55, 0.145, 0.30, "Extraction", "Non-merging watershed;\nstrict-radius border"),
    ]
    for index, (x, y, w, h, title, subtitle) in enumerate(boxes):
        face = "#E8F3F8" if index in {1, 3, 4} else "#F2F3F5"
        edge = PALETTE[DEV] if index in {1, 3, 4} else "#6B7280"
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=0.85, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.205, title, ha="center", va="center", fontsize=7.2, fontweight="bold")
        ax.text(x + w / 2, y + 0.09, subtitle, ha="center", va="center", fontsize=6.2, linespacing=1.15)
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.004, y + h / 2), (next_x - 0.004, y + h / 2),
                    arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#4B5563",
                )
            )
    ax.text(0.50, 0.30, "Label-free model and threshold selection", ha="center", va="center", fontsize=7.2, color=PALETTE[DEV], fontweight="bold")
    ax.add_patch(FancyArrowPatch((0.18, 0.37), (0.835, 0.37), arrowstyle="|-|", mutation_scale=4, linewidth=0.75, color=PALETTE[DEV]))
    ax.text(0.50, 0.10, r"Sparse bounded-degree state: $O(nk)$ storage; labels used only by the evaluator", ha="center", va="center", fontsize=6.8)
    save_figure(fig, "fig0_method_pipeline")


def plot_method_matrix(bio: pd.DataFrame, gaia: pd.DataFrame) -> None:
    bio_summary = method_summary(bio, "target_background_hmean_f1", "Biological (13)")
    gaia_summary = method_summary(gaia, "best_cluster_f1", "Gaia (24)")
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.85), constrained_layout=True)
    for ax, summary, title in zip(
        axes,
        [bio_summary, gaia_summary],
        ["(a) Biological target-background H-F1", "(b) Gaia open-cluster fields"],
        strict=True,
    ):
        summary = summary.sort_values("mean_primary")
        colors = [
            PALETTE.get(method, PALETTE["baseline"]) for method in summary["method"]
        ]
        ax.barh(summary["method"], summary["mean_primary"], color=colors, height=0.7)
        for y, value in enumerate(summary["mean_primary"]):
            ax.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=6.5)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("Mean primary F1 (failure-penalized)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax)
    save_figure(fig, "fig1_baseline_matrix")


def paired_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = frame.loc[frame["method"].isin([DEV, PREDECESSOR])].pivot(
        index="dataset_id", columns="method", values=metric
    )
    pivot["delta"] = pivot[DEV] - pivot[PREDECESSOR]
    return pivot.sort_values("delta")


def plot_paired(bio: pd.DataFrame, gaia: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.12, 5.0), constrained_layout=True)
    configs = [
        (bio, "target_background_hmean_f1", "Biological target-background H-F1"),
        (gaia, "best_cluster_f1", "Gaia best-cluster F1"),
    ]
    for col, (frame, metric, label) in enumerate(configs):
        pivot = paired_frame(frame, metric)
        ax = axes[0, col]
        ax.scatter(pivot[PREDECESSOR], pivot[DEV], s=21, color=PALETTE[DEV], alpha=0.82)
        ax.plot([0, 1], [0, 1], color="#444444", lw=0.8, ls="--")
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.set_xlabel(PREDECESSOR)
        ax.set_ylabel(DEV)
        ax.set_title(f"({'a' if col == 0 else 'b'}) {label}", loc="left", fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        sns.despine(ax=ax)

        bg = paired_frame(frame, "background_rejection")
        joined = pivot[["delta"]].rename(columns={"delta": "quality_delta"}).join(
            bg[["delta"]].rename(columns={"delta": "background_delta"})
        )
        ax = axes[1, col]
        quadrant_colors = np.where(
            joined["quality_delta"] >= 0,
            PALETTE["gain"],
            PALETTE["loss"],
        )
        ax.scatter(
            joined["quality_delta"], joined["background_delta"],
            s=24, color=quadrant_colors, alpha=0.82, edgecolor="white", linewidth=0.3,
        )
        ax.axvline(0, color="#555555", lw=0.7)
        ax.axhline(0, color="#555555", lw=0.7)
        ax.set_xlabel(r"$\Delta$ primary F1")
        ax.set_ylabel(r"$\Delta$ background rejection")
        ax.set_title(
            f"({'c' if col == 0 else 'd'}) Quality-noise trade-off",
            loc="left", fontweight="bold",
        )
        sns.despine(ax=ax)
    save_figure(fig, "fig2_paired_tradeoffs")


def summarize_synthetic(synthetic: pd.DataFrame) -> pd.DataFrame:
    pivot = synthetic.pivot_table(
        index=["case_id", "seed"], columns="method",
        values=["macro_target_f1", "pairwise_f1", "noise_f1"],
    )
    rows = []
    for case_id, part in pivot.groupby(level=0):
        row = {"case_id": case_id}
        for metric in ("macro_target_f1", "pairwise_f1", "noise_f1"):
            delta = part[(metric, DEV)] - part[(metric, PREDECESSOR)]
            row[f"{metric}_delta"] = float(delta.mean())
            row[f"{metric}_dev"] = float(part[(metric, DEV)].mean())
            row[f"{metric}_predecessor"] = float(part[(metric, PREDECESSOR)].mean())
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("macro_target_f1_delta")
    return result


def plot_secondary_diagnostics(
    biological: pd.DataFrame,
    biological_studies: pd.DataFrame,
    gaia: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> None:
    """Consolidate secondary rank, paired-effect, and scaling diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(7.12, 5.1), constrained_layout=True)

    selected = ["kNN + Leiden", PREDECESSOR, DEV, "SNN-DBSCAN"]
    study_means = (
        biological_studies.loc[biological_studies["method"].isin(selected)]
        .groupby("method")["target_background_hmean_f1"]
        .mean()
        .reindex(selected)
    )
    ax = axes[0, 0]
    colors = [PALETTE.get(method, PALETTE["baseline"]) for method in selected]
    ax.barh(selected, study_means, color=colors, height=0.65)
    for y, value in enumerate(study_means):
        ax.text(value + 0.004, y, f"{value:.3f}", va="center", fontsize=6.5)
    ax.set_xlim(0, 0.19)
    ax.set_xlabel("Study-weighted post-hoc TB-HF1")
    ax.set_title("(a) Biological rank sensitivity", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax)

    gaia_delta = paired_frame(gaia, "best_cluster_f1")["delta"].sort_values()
    ax = axes[0, 1]
    gaia_colors = np.where(
        gaia_delta > 1e-12,
        PALETTE["gain"],
        np.where(gaia_delta < -1e-12, PALETTE["loss"], PALETTE["baseline"]),
    )
    ax.bar(np.arange(len(gaia_delta)), gaia_delta, color=gaia_colors, width=0.82)
    ax.axhline(0, color="#333333", lw=0.75)
    ax.set_xlabel("Prespecified Gaia field (sorted)")
    ax.set_ylabel(r"$\Delta$ best-cluster F1")
    ax.set_title("(b) Gaia: 6 improve, 10 tie, 8 regress", loc="left", fontweight="bold")
    sns.despine(ax=ax)

    labels = {
        "multidensity_2d": "Multi-density",
        "ultrasparse_16d": "Ultra-sparse",
        "overlapping_density_16d": "Overlap-density",
        "moons_2d": "Moons",
        "rings_2d": "Rings",
        "overlap_8d": "Gaussian overlap",
        "imbalanced_16d": "Imbalanced",
    }
    ax = axes[1, 0]
    x = np.arange(len(synthetic))
    width = 0.24
    specs = [
        ("macro_target_f1_delta", "Target F1", "#0072B2"),
        ("pairwise_f1_delta", "Pairwise F1", "#009E73"),
        ("noise_f1_delta", "Noise F1", "#CC79A7"),
    ]
    for offset, (column, label, color) in zip((-width, 0.0, width), specs, strict=True):
        ax.bar(x + offset, synthetic[column], width=width, label=label, color=color)
    ax.axhline(0, color="#333333", lw=0.75)
    ax.set_xticks(
        x,
        [labels[value] for value in synthetic["case_id"]],
        rotation=28,
        ha="right",
    )
    ax.set_ylabel(r"Mean paired $\Delta$ (MDL $-$ 0.1.2)")
    ax.set_title("(c) Locked synthetic effects", loc="left", fontweight="bold")
    ax.legend(ncol=3, loc="upper left", frameon=False, fontsize=6.3)
    sns.despine(ax=ax)

    current_bio = biological.loc[biological["method"].eq(DEV)].copy()
    current_gaia = gaia.loc[gaia["method"].eq(DEV)].copy()
    ax = axes[1, 1]
    ax.scatter(
        current_gaia["n"], current_gaia["runtime_seconds"],
        s=18, alpha=0.72, color="#56B4E9", label="Gaia fields",
    )
    ax.scatter(
        current_bio["n"], current_bio["runtime_seconds"],
        s=22, alpha=0.82, color=PALETTE[DEV], label="Cytometry datasets",
    )
    mosmann = current_bio.loc[current_bio["dataset_id"].eq("mosmann")].iloc[0]
    ax.annotate(
        "Mosmann\n396,460; 517 s",
        (mosmann["n"], mosmann["runtime_seconds"]),
        xytext=(-48, -12), textcoords="offset points", fontsize=6.2,
        arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "#444444"},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observations per evaluated dataset")
    ax.set_ylabel("End-to-end runtime (s)")
    ax.set_title("(d) Fixed-range snapshot: direct pilots", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=6.3)
    sns.despine(ax=ax)

    save_figure(fig, "figS1_secondary_diagnostics", SUPPLEMENT_FIGURES)


def write_synthetic_tex(table: pd.DataFrame) -> None:
    labels = {
        "multidensity_2d": "Multi-density 2D",
        "ultrasparse_16d": "Ultra-sparse 16D",
        "overlapping_density_16d": "Overlap-density 16D",
        "moons_2d": "Moons 2D",
        "rings_2d": "Rings 2D",
        "overlap_8d": "Gaussian overlap 8D",
        "imbalanced_16d": "Imbalanced 16D",
    }
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Family & 0.1.2 macro & MDL macro & $\Delta$ macro & $\Delta$ pairwise \\",
        r"\midrule",
    ]
    for row in table.sort_values("case_id").itertuples(index=False):
        lines.append(
            f"{labels[row.case_id]} & {row.macro_target_f1_predecessor:.3f} & "
            f"{row.macro_target_f1_dev:.3f} & {row.macro_target_f1_delta:+.3f} & "
            f"{row.pairwise_f1_delta:+.3f} \\\\"
        )
    lines.extend([r"\midrule", r"Mean over 14 pairs & 0.830 & 0.907 & +0.077 & +0.060 \\", r"\bottomrule", r"\end{tabular}"])
    (TABLES / "table_locked_synthetic.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    configure_style()
    biological, gaia, synthetic = load_blocks()

    bio_summary = method_summary(biological, "target_background_hmean_f1", "Biological (13)")
    bio_studies = study_aggregate(biological)
    bio_study_summary = method_summary(
        bio_studies.rename(columns={"study": "dataset_id"}),
        "target_background_hmean_f1",
        "Biological studies (4)",
    )
    gaia_summary = method_summary(gaia, "best_cluster_f1", "Gaia (24)")
    combined = pd.concat([bio_summary, bio_study_summary, gaia_summary], ignore_index=True)
    combined.to_csv(TABLES / "method_comparison_matrix.csv", index=False)
    biological.to_csv(TABLES / "biological_dataset_results.csv", index=False)
    gaia.to_csv(TABLES / "gaia24_field_results.csv", index=False)
    write_tex_table(bio_summary, TABLES / "table_biological.tex", COMMON_BIO)
    write_tex_table(gaia_summary, TABLES / "table_gaia24.tex", METHOD_ORDER)

    plot_method_pipeline()
    synthetic_table = summarize_synthetic(synthetic)
    plot_secondary_diagnostics(biological, bio_studies, gaia, synthetic_table)
    synthetic_table.to_csv(TABLES / "locked_synthetic_by_family.csv", index=False)
    write_synthetic_tex(synthetic_table)

    bio_primary = paired_frame(biological, "target_background_hmean_f1")
    bio_study_primary = paired_frame(
        bio_studies.rename(columns={"study": "dataset_id"}),
        "target_background_hmean_f1",
    )
    bio_bg = paired_frame(biological, "background_rejection")
    gaia_primary = paired_frame(gaia, "best_cluster_f1")
    gaia_bg = paired_frame(gaia, "background_rejection")
    stats = {
        "biological_posthoc_dataset_composite": paired_summary(
            bio_primary[DEV], bio_primary[PREDECESSOR]
        ),
        "biological_posthoc_study_composite": paired_summary(
            bio_study_primary[DEV], bio_study_primary[PREDECESSOR]
        ),
        "biological_background": paired_summary(bio_bg[DEV], bio_bg[PREDECESSOR]),
        "gaia_primary": paired_summary(gaia_primary[DEV], gaia_primary[PREDECESSOR]),
        "gaia_background": paired_summary(gaia_bg[DEV], gaia_bg[PREDECESSOR]),
        "source_rows": {
            "biological": int(len(biological)),
            "gaia": int(len(gaia)),
            "synthetic": int(len(synthetic)),
        },
        "protocol_note": (
            "Legacy baselines are collapsed by dataset using the median over declared stochastic "
            "seeds; failed jobs contribute zero. MDL-StrataSCAN and its 0.1.2 predecessor use the "
            "direct seed-42 pilot runs."
        ),
    }
    (TABLES / "statistical_summary.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
