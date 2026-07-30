from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
PUBLICATION = REPO / "results/runs/publication_full_7synthetic_20260722/results.csv"
PILOT_BIO = REPO / "results/runs/optimization_dev10_real_pilot_cytometry/results.csv"
LOCKED_SYNTH = REPO / "results/runs/optimization_dev10_locked_n5000_seeds197_251_k1/metrics.csv"
RUNTIME_20K = REPO / "results/runs/optimization_dev10_runtime_n20000_seed197/metrics.csv"
DEV9_SCALING = REPO / "results/runs/optimization_v020_dev9_scaling/results.csv"
PREDECESSOR_SCALING = (
    REPO
    / "results/predictive_scaling_5m_20260727/analysis/extended-scaling-summary.csv"
)

DEV = "MDL-StrataSCAN"
OLD = "StrataSCAN 0.1.2"
LEGACY = "StrataSCAN 0.1.0"
DISPLAY = {
    "StrataSCAN-Optimization": DEV,
    "StrataSCAN": OLD,
    "icl": DEV,
    "stratascan_0.1.2": OLD,
    "kNN+Leiden": "kNN + Leiden",
}
BIO_METHOD_ORDER = [
    "DBSCAN", "OPTICS", "SNN-DBSCAN", "VDBSCAN-2007",
    "kNN-DBSCAN", "kNN + Leiden", OLD, DEV,
]
SYN_METHOD_ORDER = [
    "DBSCAN", "HDBSCAN", "OPTICS", "SNN-DBSCAN", "VDBSCAN-2007",
    "AMD-DBSCAN", "kNN-DBSCAN", "kNN + Leiden", LEGACY,
]
DATASET_LABELS = {
    "levine": "Levine",
    "mosmann": "Mosmann",
    "nilsson": "Nilsson",
    **{f"samusik_{i:02d}": f"Samusik {i:02d}" for i in range(1, 11)},
}
CASE_LABELS = {
    "multidensity_2d": "Multi-density\n2D",
    "ultrasparse_16d": "Ultra-sparse\n16D",
    "overlapping_density_16d": "Overlap-density\n16D",
    "moons_2d": "Moons\n2D",
    "rings_2d": "Rings\n2D",
    "overlap_8d": "Gaussian overlap\n8D",
    "imbalanced_16d": "Imbalanced\n16D",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.2,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.6,
        "figure.dpi": 160,
        "savefig.dpi": 450,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def biological_outputs() -> None:
    matrix = pd.read_csv(TABLES / "biological_dataset_results.csv")
    matrix["dataset_label"] = matrix["dataset_id"].map(DATASET_LABELS)
    macro = matrix.pivot(
        index="dataset_label", columns="method", values="target_background_hmean_f1"
    )
    dataset_order = [DATASET_LABELS[x] for x in DATASET_LABELS]
    macro = macro.reindex(index=dataset_order, columns=BIO_METHOD_ORDER)

    comparison = pd.read_csv(TABLES / "method_comparison_matrix.csv")
    weighting = comparison.loc[
        comparison["block"].isin(["Biological (13)", "Biological studies (4)"])
    ].pivot(index="method", columns="block", values="mean_primary")
    weighting_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Dataset mean & Study mean \\",
        r"\midrule",
    ]
    for method in ["SNN-DBSCAN", OLD, DEV, "kNN + Leiden"]:
        row = weighting.loc[method]
        name = method.replace("kNN + Leiden", r"$k$NN+Leiden")
        weighting_lines.append(
            f"{name} & {row['Biological (13)']:.3f} & "
            f"{row['Biological studies (4)']:.3f} \\\\"
        )
    weighting_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "table_biological_weighting.tex").write_text(
        "\n".join(weighting_lines) + "\n", encoding="utf-8"
    )

    direct = pd.read_csv(PILOT_BIO)
    direct["method"] = direct["method"].replace(DISPLAY)
    direct["dataset_label"] = direct["dataset_id"].map(DATASET_LABELS)
    metrics = [
        "n", "truth_clusters", "n_clusters", "macro_target_f1", "noise_f1", "pairwise_f1",
        "signal_coverage", "background_rejection", "extra_signal_fragments",
        "merged_predicted_clusters", "spurious_background_majority_clusters",
        "runtime_seconds", "incremental_rss_mb", "process_peak_rss_mb",
    ]
    pivot = direct.pivot(index="dataset_label", columns="method", values=metrics).reindex(dataset_order)
    rows = []
    for label in dataset_order:
        row = {"dataset": label}
        for metric in metrics:
            row[f"{metric}_old"] = float(pivot.loc[label, (metric, OLD)])
            row[f"{metric}_dev"] = float(pivot.loc[label, (metric, DEV)])
        row["macro_delta"] = row["macro_target_f1_dev"] - row["macro_target_f1_old"]
        for suffix in ("old", "dev"):
            target = row[f"macro_target_f1_{suffix}"]
            noise = row[f"noise_f1_{suffix}"]
            row[f"target_background_hmean_f1_{suffix}"] = (
                2.0 * target * noise / (target + noise) if target + noise else 0.0
            )
            row[f"noise_aware_macro_f1_{suffix}"] = row[
                f"target_background_hmean_f1_{suffix}"
            ]
        row["target_background_hmean_delta"] = (
            row["target_background_hmean_f1_dev"]
            - row["target_background_hmean_f1_old"]
        )
        row["noise_aware_delta"] = row["target_background_hmean_delta"]
        row["noise_delta"] = row["noise_f1_dev"] - row["noise_f1_old"]
        row["pairwise_delta"] = row["pairwise_f1_dev"] - row["pairwise_f1_old"]
        row["coverage_delta"] = row["signal_coverage_dev"] - row["signal_coverage_old"]
        row["background_delta"] = row["background_rejection_dev"] - row["background_rejection_old"]
        row["old_cluster_ratio_log2"] = np.log2(row["n_clusters_old"] / row["truth_clusters_old"])
        row["dev_cluster_ratio_log2"] = np.log2(row["n_clusters_dev"] / row["truth_clusters_dev"])
        rows.append(row)
    detail = pd.DataFrame(rows)
    detail.to_csv(TABLES / "biological_segmentation_by_dataset.csv", index=False)

    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Dataset & $n$ (k) & $K^*$ & $\hat K$ & Target F1 & BG F1 & TB-HF1 & Pairwise & BG recall & Time (s) \\",
        r"\midrule",
    ]
    for row in detail.itertuples(index=False):
        name = row.dataset.replace(" ", r"\ ")
        lines.append(
            f"{name} & {row.n_old / 1000:.1f} & {int(row.truth_clusters_old)} & "
            f"{int(row.n_clusters_old)}/{int(row.n_clusters_dev)} & "
            f"{row.macro_target_f1_old:.3f}/{row.macro_target_f1_dev:.3f} & "
            f"{row.noise_f1_old:.3f}/{row.noise_f1_dev:.3f} & "
            f"{row.target_background_hmean_f1_old:.3f}/{row.target_background_hmean_f1_dev:.3f} & "
            f"{row.pairwise_f1_old:.3f}/{row.pairwise_f1_dev:.3f} & "
            f"{row.background_rejection_old:.3f}/{row.background_rejection_dev:.3f} & "
            f"{row.runtime_seconds_old:.1f}/{row.runtime_seconds_dev:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "table_biological_by_dataset.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig = plt.figure(figsize=(7.12, 5.15), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[2.5, 1.30, 0.75, 0.78], wspace=0.04)
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(macro, ax=ax0, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2f",
                annot_kws={"fontsize": 5.6}, cbar_kws={"label": "Target-background H-F1", "shrink": 0.72})
    ax0.set(xlabel="", ylabel="")
    ax0.set_title("(a) TB-HF1", loc="left", fontweight="bold")
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=48, ha="right")

    delta = detail.set_index("dataset")[["target_background_hmean_delta", "macro_delta", "noise_delta", "background_delta"]]
    delta.columns = [r"$\Delta$TB", r"$\Delta$Target", r"$\Delta$BG F1", r"$\Delta$BG recall"]
    ax1 = fig.add_subplot(gs[0, 1])
    sns.heatmap(delta, ax=ax1, vmin=-0.3, vmax=0.3, center=0, cmap="vlag", annot=True, fmt="+.2f",
                annot_kws={"fontsize": 5.5}, yticklabels=False,
                cbar_kws={"label": "MDL $-$ 0.1.2", "shrink": 0.72})
    ax1.set(xlabel="", ylabel="")
    ax1.set_title("(b) Paired effects", loc="left", fontweight="bold")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=48, ha="right")

    ratios = detail.set_index("dataset")[["old_cluster_ratio_log2", "dev_cluster_ratio_log2"]]
    ratios.columns = ["0.1.2", "MDL"]
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(ratios, ax=ax2, vmin=-4, vmax=6, center=0, cmap="coolwarm", annot=True, fmt="+.1f",
                annot_kws={"fontsize": 5.6}, yticklabels=False,
                cbar_kws={"label": r"$\log_2(\hat K/K^*)$", "shrink": 0.72})
    ax2.set(xlabel="", ylabel="")
    ax2.set_title("(c) Granularity", loc="left", fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=48, ha="right")

    counts = detail.set_index("dataset")[["extra_signal_fragments_dev", "merged_predicted_clusters_dev"]]
    counts.columns = ["Extra\nfragments", "Merged\nsegments"]
    ax3 = fig.add_subplot(gs[0, 3])
    sns.heatmap(counts, ax=ax3, vmin=0, cmap="Blues", annot=True, fmt=".0f",
                annot_kws={"fontsize": 5.8}, yticklabels=False,
                cbar_kws={"label": "Count", "shrink": 0.72})
    ax3.set(xlabel="", ylabel="")
    ax3.set_title("(d) MDL errors", loc="left", fontweight="bold")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=48, ha="right")
    save(fig, "fig4_biological_by_dataset")


def synthetic_outputs() -> None:
    frame = pd.read_csv(PUBLICATION)
    frame = frame.loc[(frame["suite"] == "synthetic") & (frame["n"] == 5000)].copy()
    frame["method"] = frame["method"].replace({"StrataSCAN": LEGACY, "kNN+Leiden": "kNN + Leiden"})
    quality = ["macro_target_f1", "pairwise_f1", "noise_f1", "signal_coverage"]
    failed = frame["status"] != "ok"
    frame.loc[failed, quality] = 0.0
    frame["case_id"] = frame["dataset_id"].str.replace("__quality__n5000", "", regex=False)
    rows = []
    for method, part in frame.groupby("method"):
        truth = part["truth_clusters"].astype(float)
        predicted = part["n_clusters"].astype(float)
        rows.append({
            "method": method,
            "jobs": len(part),
            "successful_jobs": int((part["status"] == "ok").sum()),
            "mean_macro_target_f1": part["macro_target_f1"].mean(),
            "mean_pairwise_f1": part["pairwise_f1"].mean(),
            "mean_noise_f1": part["noise_f1"].mean(),
            "mean_signal_coverage": part["signal_coverage"].mean(),
            "median_abs_log2_cluster_ratio": np.median(
                np.abs(np.log2((predicted + 0.5) / (truth + 0.5)))
            ),
            "median_runtime_seconds": part["runtime_seconds"].median(),
            "median_incremental_rss_mb": part["incremental_rss_mb"].median(),
            "median_process_peak_rss_mb": part["process_peak_rss_mb"].median(),
        })
    summary = pd.DataFrame(rows)
    summary["order"] = summary["method"].map({m: i for i, m in enumerate(SYN_METHOD_ORDER)})
    summary = summary.sort_values("order").drop(columns="order")
    summary.to_csv(TABLES / "synthetic_five_seed_resource_summary.csv", index=False)

    lines = [
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Method & Jobs & Macro & Pairwise & Noise F1 & Coverage & Time (s) & $\Delta$RSS (MiB) \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        name = row.method.replace("kNN + Leiden", r"$k$NN+Leiden")
        lines.append(
            f"{name} & {row.successful_jobs}/{row.jobs} & {row.mean_macro_target_f1:.3f} & "
            f"{row.mean_pairwise_f1:.3f} & {row.mean_noise_f1:.3f} & "
            f"{row.mean_signal_coverage:.3f} & {row.median_runtime_seconds:.3f} & "
            f"{row.median_incremental_rss_mb:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "table_synthetic_five_seed_resources.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    case_order = list(CASE_LABELS)
    macro = frame.pivot_table(index="method", columns="case_id", values="macro_target_f1", aggfunc="mean")
    macro = macro.reindex(index=SYN_METHOD_ORDER, columns=case_order)
    clusters = frame.pivot_table(index="method", columns="case_id", values="n_clusters", aggfunc="median")
    truth = frame.groupby("case_id")["truth_clusters"].median().reindex(case_order)
    log_ratio = np.log2(clusters.reindex(index=SYN_METHOD_ORDER, columns=case_order).divide(truth, axis=1))
    macro.rename(columns=CASE_LABELS).to_csv(TABLES / "synthetic_five_seed_family_macro_f1.csv")
    log_ratio.rename(columns=CASE_LABELS).to_csv(TABLES / "synthetic_five_seed_cluster_ratio_log2.csv")

    fig, axes = plt.subplots(2, 1, figsize=(7.12, 5.25), constrained_layout=True)
    labels = [CASE_LABELS[x] for x in case_order]
    sns.heatmap(macro, ax=axes[0], vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2f",
                annot_kws={"fontsize": 5.8}, cbar_kws={"label": "Macro target F1", "shrink": 0.78})
    axes[0].set(xlabel="", ylabel="")
    axes[0].set_xticklabels(labels, rotation=0)
    axes[0].set_title("(a) Recovery on the original five-seed synthetic matrix", loc="left", fontweight="bold")
    sns.heatmap(log_ratio, ax=axes[1], vmin=-4, vmax=7, center=0, cmap="coolwarm", annot=True, fmt="+.1f",
                annot_kws={"fontsize": 5.8}, cbar_kws={"label": r"$\log_2(\hat K/K^*)$", "shrink": 0.78})
    axes[1].set(xlabel="", ylabel="")
    axes[1].set_xticklabels(labels, rotation=0)
    axes[1].set_title("(b) Segmentation granularity: 0 is the reference cluster count", loc="left", fontweight="bold")
    save(fig, "fig5_synthetic_baseline_segmentation")

    locked = pd.read_csv(LOCKED_SYNTH)
    locked["runtime_seconds"] = locked["fit_seconds_without_graph"] + locked["graph_seconds"]
    locked["method"] = locked["method"].replace(DISPLAY)
    locked.groupby("method").agg(
        mean_macro_target_f1=("macro_target_f1", "mean"),
        mean_pairwise_f1=("pairwise_f1", "mean"),
        mean_noise_f1=("noise_f1", "mean"),
        median_runtime_seconds=("runtime_seconds", "median"),
        median_clusters=("n_clusters", "median"),
    ).to_csv(TABLES / "locked_synthetic_method_summary.csv")


def scalability_outputs() -> None:
    """State exactly which scale claims belong to which algorithm snapshot."""
    locked = pd.read_csv(LOCKED_SYNTH)
    locked["runtime_seconds"] = locked["fit_seconds_without_graph"] + locked["graph_seconds"]
    locked_current = locked.loc[locked["method"].eq("icl")]

    runtime_20k = pd.read_csv(RUNTIME_20K)
    runtime_20k["runtime_seconds"] = (
        runtime_20k["fit_seconds_without_graph"] + runtime_20k["graph_seconds"]
    )
    runtime_20k_current = runtime_20k.loc[runtime_20k["method"].eq("icl")]

    biological = pd.read_csv(PILOT_BIO)
    biological_current = biological.loc[
        biological["method"].eq("StrataSCAN-Optimization")
        & biological["status"].eq("ok")
    ]
    largest_biological = biological_current.sort_values("n").iloc[-1]

    dev9 = pd.read_csv(DEV9_SCALING)
    dev9_million = dev9.loc[
        dev9["method"].eq("StrataSCAN-Optimization") & dev9["n"].eq(1_000_000)
    ].iloc[0]

    predecessor = pd.read_csv(PREDECESSOR_SCALING)
    predecessor_5m = predecessor.loc[predecessor["n"].eq(5_000_000)].iloc[0]

    evidence = pd.DataFrame(
        [
            {
                "snapshot": "Fixed-range MDL",
                "scope": "14 locked synthetic pairs",
                "n": 5_000,
                "runtime_seconds": float(locked_current["runtime_seconds"].median()),
                "memory_record": "not monitored",
                "claim_boundary": "evaluated snapshot; paired-process engineering time",
            },
            {
                "snapshot": "Fixed-range MDL",
                "scope": "7 synthetic families",
                "n": 20_000,
                "runtime_seconds": float(runtime_20k_current["runtime_seconds"].median()),
                "memory_record": "not monitored",
                "claim_boundary": "evaluated snapshot; one seed",
            },
            {
                "snapshot": "Fixed-range MDL",
                "scope": str(largest_biological["dataset_id"]).capitalize(),
                "n": int(largest_biological["n"]),
                "runtime_seconds": float(largest_biological["runtime_seconds"]),
                "memory_record": f"{float(largest_biological['peak_rss_mb']):.1f} peak",
                "claim_boundary": "largest directly evaluated snapshot job",
            },
            {
                "snapshot": "Earlier dev9",
                "scope": "Moons smoke test",
                "n": int(dev9_million["n"]),
                "runtime_seconds": float(dev9_million["runtime_seconds"]),
                "memory_record": f"{float(dev9_million['peak_rss_mb']):.1f} peak",
                "claim_boundary": "different algorithm; sparse-lineage evidence only",
            },
            {
                "snapshot": "Predecessor 0.1.2",
                "scope": "7-family median",
                "n": int(predecessor_5m["n"]),
                "runtime_seconds": float(predecessor_5m["new_median_seconds"]),
                "memory_record": "endpoint only",
                "claim_boundary": "different algorithm; not MDL evidence",
            },
        ]
    )
    evidence.to_csv(TABLES / "scalability_evidence_boundaries.csv", index=False)

    lines = [
        r"\begin{tabular}{@{}lrrl@{}}",
        r"\toprule",
        r"Evidence & $n$ & Time (s) & Memory (MiB) \\",
        r"\midrule",
    ]
    for row in evidence.itertuples(index=False):
        label = f"{row.snapshot}: {row.scope}"
        lines.append(
            f"{label} & {row.n:,} & {row.runtime_seconds:.1f} & {row.memory_record} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "table_scalability_boundaries.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    configure_style()
    biological_outputs()
    synthetic_outputs()
    scalability_outputs()


if __name__ == "__main__":
    main()
