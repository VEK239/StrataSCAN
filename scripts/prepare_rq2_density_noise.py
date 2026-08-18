from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DENSITY_SEEDS = [211, 223, 227]
NOISE_SEEDS = [211, 223, 227, 229, 233]
NOISE_LEVELS = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
METHODS = [
    "AMD-DBSCAN",
    "DBSCAN",
    "HDBSCAN",
    "kNN-DBSCAN",
    "kNN+Leiden",
    "OPTICS",
    "SNN-DBSCAN",
    "StrataSCAN",
    "VDBSCAN-2007",
]

DENSITY_PATTERN = re.compile(
    r"^density_r(?P<ratio>16|64|256)_(?P<shape>gaussian|anisotropic)_"
    r"(?P<dimension>2|8)d__quality__n10000$"
)
NOISE_PATTERN = re.compile(
    r"^global_contamination_noise(?P<noise>25|50|75|90|95|99)"
    r"__quality__signal300__n(?P<n>400|600|1200|3000|6000|30000)$"
)

EXPECTED_DENSITY_CASES = {
    (16, "gaussian", 2),
    (64, "gaussian", 2),
    (256, "gaussian", 2),
    (64, "anisotropic", 2),
    (256, "anisotropic", 2),
    (16, "gaussian", 8),
    (64, "gaussian", 8),
    (256, "gaussian", 8),
    (256, "anisotropic", 8),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_locked_evaluation(frame: pd.DataFrame) -> None:
    required = {
        "job_id",
        "status",
        "dataset_id",
        "method",
        "seed",
        "n",
        "evaluation_protocol_version",
        "matching_strategy",
        "matching_objective",
        "discovery_purity_threshold",
        "discovery_coverage_threshold",
        "macro_target_f1",
        "macro_target_purity",
        "macro_target_coverage",
        "target_discovery_rate",
        "noise_evidence_f1",
        "predicted_cluster_count",
        "unmatched_predicted_cluster_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing locked evaluation columns: {missing}")
    if frame["job_id"].duplicated().any():
        raise ValueError("duplicate job IDs")
    if set(frame["evaluation_protocol_version"].dropna()) != {"target-discovery-v1"}:
        raise ValueError("evaluation protocol is not target-discovery-v1")
    if set(frame["matching_strategy"].dropna()) != {"hungarian"}:
        raise ValueError("matching strategy is not Hungarian")
    if set(frame["matching_objective"].dropna()) != {"pairwise_f1"}:
        raise ValueError("matching objective is not pairwise F1")
    if set(pd.to_numeric(frame["discovery_purity_threshold"], errors="coerce").dropna()) != {0.9}:
        raise ValueError("discovery purity threshold differs from 0.90")
    if set(pd.to_numeric(frame["discovery_coverage_threshold"], errors="coerce").dropna()) != {0.1}:
        raise ValueError("discovery coverage threshold differs from 0.10")


def summarize_density(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    _assert_locked_evaluation(frame)
    if frame.shape[0] != 27:
        raise ValueError(f"expected 27 density cells, found {frame.shape[0]}")
    if set(frame["method"]) != {"StrataSCAN"}:
        raise ValueError("density evidence is not the StrataSCAN-only protocol slice")
    if set(frame["seed"]) != set(DENSITY_SEEDS):
        raise ValueError("density seeds differ from the frozen design")
    if set(frame["status"]) != {"ok"}:
        raise ValueError("density matrix is not 27/27 successful")

    parsed = frame["dataset_id"].str.extract(DENSITY_PATTERN)
    if parsed.isna().any(axis=None):
        raise ValueError("unexpected density dataset identifier")
    data = frame.copy()
    data["density_ratio"] = parsed["ratio"].astype(int)
    data["shape"] = parsed["shape"].replace({"gaussian": "isotropic"})
    data["dimension"] = parsed["dimension"].astype(int)
    cases = set(zip(data["density_ratio"], data["shape"].replace({"isotropic": "gaussian"}), data["dimension"]))
    if cases != EXPECTED_DENSITY_CASES:
        raise ValueError("density case matrix differs from the frozen design")
    observed = set(zip(data["density_ratio"], data["shape"], data["dimension"], data["seed"]))
    expected = {
        (ratio, "isotropic" if shape == "gaussian" else shape, dimension, seed)
        for ratio, shape, dimension in EXPECTED_DENSITY_CASES
        for seed in DENSITY_SEEDS
    }
    if observed != expected:
        raise ValueError("density case-by-seed matrix is incomplete")

    rows: list[dict[str, float | int | str]] = []
    for (ratio, shape, dimension), block in data.groupby(
        ["density_ratio", "shape", "dimension"], sort=True
    ):
        row: dict[str, float | int | str] = {
            "density_ratio": int(ratio),
            "shape": str(shape),
            "dimension": int(dimension),
            "n_seeds": int(block.shape[0]),
            "n_successful": int((block["status"] == "ok").sum()),
        }
        for metric in (
            "macro_target_f1",
            "macro_target_purity",
            "macro_target_coverage",
            "target_discovery_rate",
            "predicted_cluster_count",
            "unmatched_predicted_cluster_count",
        ):
            values = pd.to_numeric(block[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["dimension", "shape", "density_ratio"])
    audit = {
        "attempted": int(data.shape[0]),
        "successful": int((data["status"] == "ok").sum()),
        "minimum_cell_target_f1": float(data["macro_target_f1"].min()),
        "minimum_cell_target_discovery_rate": float(data["target_discovery_rate"].min()),
        "minimum_case_median_target_f1": float(summary["macro_target_f1_median"].min()),
        "maximum_case_median_target_f1": float(summary["macro_target_f1_median"].max()),
        "median_predicted_clusters": float(data["predicted_cluster_count"].median()),
        "median_unmatched_clusters": float(data["unmatched_predicted_cluster_count"].median()),
        "maximum_unmatched_clusters": int(data["unmatched_predicted_cluster_count"].max()),
    }
    return summary, audit


def summarize_noise(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    _assert_locked_evaluation(frame)
    if frame.shape[0] != 270:
        raise ValueError(f"expected 270 rare-dense cells, found {frame.shape[0]}")
    if set(frame["method"]) != set(METHODS):
        raise ValueError("rare-dense method matrix differs from the frozen design")
    if set(frame["seed"]) != set(NOISE_SEEDS):
        raise ValueError("rare-dense seeds differ from the frozen design")
    parsed = frame["dataset_id"].str.extract(NOISE_PATTERN)
    if parsed.isna().any(axis=None):
        raise ValueError("unexpected rare-dense dataset identifier")
    data = frame.copy()
    data["noise_fraction"] = parsed["noise"].astype(float) / 100.0
    data["declared_total_n"] = parsed["n"].astype(int)
    observed_n = pd.to_numeric(data["n"], errors="coerce")
    if not (
        observed_n.loc[observed_n.notna()].astype(int)
        == data.loc[observed_n.notna(), "declared_total_n"]
    ).all():
        raise ValueError("dataset ID and n column disagree")
    expected = {
        (method, noise, seed)
        for method in METHODS
        for noise in NOISE_LEVELS
        for seed in NOISE_SEEDS
    }
    observed = set(zip(data["method"], data["noise_fraction"], data["seed"]))
    if observed != expected:
        raise ValueError("rare-dense method-by-noise-by-seed matrix is incomplete")
    statuses = data["status"].value_counts().to_dict()
    if statuses != {"ok": 268, "error": 2}:
        raise ValueError(f"unexpected completion profile: {statuses}")
    failures = data.loc[data["status"] != "ok", ["method", "noise_fraction", "seed"]]
    if set(failures["method"]) != {"AMD-DBSCAN"} or set(failures["noise_fraction"]) != {0.99}:
        raise ValueError("unexpected rare-dense failure identity")

    rows: list[dict[str, float | int | str]] = []
    for (method, noise), block in data.groupby(["method", "noise_fraction"], sort=True):
        ok = block.loc[block["status"] == "ok"]
        row: dict[str, float | int | str] = {
            "method": str(method),
            "noise_fraction": float(noise),
            "total_n": int(block["declared_total_n"].iloc[0]),
            "n_attempted": int(block.shape[0]),
            "n_successful": int(ok.shape[0]),
            "completion_rate": float(ok.shape[0] / block.shape[0]),
        }
        for metric in (
            "macro_target_f1",
            "macro_target_purity",
            "macro_target_coverage",
            "target_discovery_rate",
            "noise_evidence_f1",
            "predicted_cluster_count",
            "unmatched_predicted_cluster_count",
        ):
            values = pd.to_numeric(ok[metric], errors="coerce").dropna()
            for suffix, func in (("median", values.median), ("min", values.min), ("max", values.max)):
                row[f"{metric}_{suffix}"] = float(func()) if not values.empty else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["method", "noise_fraction"])
    strata = summary.loc[summary["method"] == "StrataSCAN"].copy()
    endpoint = summary.loc[summary["noise_fraction"] == 0.99].copy()
    endpoint = endpoint.sort_values(
        ["noise_evidence_f1_median", "macro_target_f1_median"], ascending=False
    )
    raw_strata = data.loc[data["method"] == "StrataSCAN"]
    audit = {
        "attempted": int(data.shape[0]),
        "successful": int((data["status"] == "ok").sum()),
        "stratascan_attempted": int(raw_strata.shape[0]),
        "stratascan_successful": int((raw_strata["status"] == "ok").sum()),
        "stratascan_minimum_cell_discovery_rate": float(raw_strata["target_discovery_rate"].min()),
        "stratascan_minimum_level_median_target_f1": float(strata["macro_target_f1_median"].min()),
        "stratascan_maximum_level_median_target_f1": float(strata["macro_target_f1_median"].max()),
        "stratascan_99_target_f1": float(
            strata.loc[strata["noise_fraction"] == 0.99, "macro_target_f1_median"].iloc[0]
        ),
        "stratascan_99_noise_f1": float(
            strata.loc[strata["noise_fraction"] == 0.99, "noise_evidence_f1_median"].iloc[0]
        ),
        "amd_failures": int((data["status"] != "ok").sum()),
    }
    return summary, strata, endpoint, audit


def plot_figure(density: pd.DataFrame, strata_noise: pd.DataFrame, output: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.48))

    ax = axes[0]
    styles = {
        (2, "isotropic"): ("#8c1d40", "o", "-", "2D isotropic"),
        (2, "anisotropic"): ("#8c1d40", "s", "--", "2D anisotropic"),
        (8, "isotropic"): ("#3f5d7d", "^", "-", "8D isotropic"),
        (8, "anisotropic"): ("#3f5d7d", "D", "--", "8D anisotropic"),
    }
    for (dimension, shape), (color, marker, linestyle, label) in styles.items():
        block = density.loc[
            (density["dimension"] == dimension) & (density["shape"] == shape)
        ].sort_values("density_ratio")
        if block.empty:
            continue
        x = block["density_ratio"].to_numpy(float)
        y = block["macro_target_f1_median"].to_numpy(float)
        low = block["macro_target_f1_min"].to_numpy(float)
        high = block["macro_target_f1_max"].to_numpy(float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.25,
            markersize=3.8,
            capsize=2,
            elinewidth=0.75,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 64, 256], ["16", "64", "256"])
    ax.set_xlabel("Peak-density ratio")
    ax.set_ylabel("Macro target F1")
    ax.set_ylim(0.68, 1.015)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=2.2, columnspacing=0.8)
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    ax = axes[1]
    block = strata_noise.sort_values("noise_fraction")
    x = np.arange(block.shape[0], dtype=float)
    specs = [
        ("macro_target_f1", "Target F1", "#8c1d40", "o", "-"),
        ("noise_evidence_f1", "Background F1", "#176d8a", "s", "--"),
    ]
    for metric, label, color, marker, linestyle in specs:
        y = block[f"{metric}_median"].to_numpy(float)
        low = block[f"{metric}_min"].to_numpy(float)
        high = block[f"{metric}_max"].to_numpy(float)
        ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=3.8,
            label=label,
        )
    ax.set_xticks(x, ["25", "50", "75", "90", "95", "99"])
    ax.set_xlabel("Low-density background (%)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.015)
    ax.set_yticks([0.0, 0.5, 0.8, 1.0])
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.01, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    ax.annotate(
        "99%: .979 / 1.000",
        xy=(5, float(block.loc[block["noise_fraction"] == 0.99, "macro_target_f1_median"].iloc[0])),
        xytext=(3.30, 0.69),
        fontsize=6.6,
        color="#444444",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#777777"},
    )

    for ax in axes:
        ax.grid(axis="y", color="#dedede", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.985, wspace=0.28)

    paths: list[Path] = []
    for suffix, dpi in (("pdf", None), ("svg", None), ("png", 400)):
        path = output / f"rq2_density_noise.{suffix}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def write_endpoint_table(endpoint: pd.DataFrame, output: Path) -> Path:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Recovery at 99\% low-density background. Medians are over successful paired seeds; completion retains failures in the denominator.}",
        r"\label{tab:rare-dense-99}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Method & Target F1 & Background F1 & Done \\",
        r"\hline",
    ]
    preferred_order = [
        "StrataSCAN",
        "OPTICS",
        "kNN-DBSCAN",
        "DBSCAN",
        "SNN-DBSCAN",
        "VDBSCAN-2007",
        "HDBSCAN",
        "kNN+Leiden",
        "AMD-DBSCAN",
    ]
    for method in preferred_order:
        row = endpoint.loc[endpoint["method"] == method].iloc[0]
        name = r"\textbf{StrataSCAN}" if method == "StrataSCAN" else method.replace("kNN", r"$k$NN")
        target = f"{row['macro_target_f1_median']:.3f}"
        noise = f"{row['noise_evidence_f1_median']:.3f}"
        if method == "StrataSCAN":
            target = rf"\textbf{{{target}}}"
            noise = rf"\textbf{{{noise}}}"
        lines.append(
            f"{name} & {target} & {noise} & {int(row['n_successful'])}/{int(row['n_attempted'])} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path = output / "table_rq2_noise99.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_subsection(
    density_audit: dict[str, float | int],
    noise_audit: dict[str, float | int],
    output: Path,
) -> Path:
    text = f"""## Density adaptation and rare-target recovery

**RQ2. Can StrataSCAN recover clusters with strongly varying densities and preserve rare dense targets under extreme background contamination?**

We first varied the peak density among six equal-mass Gaussian targets while holding the target count (1,250 points each), separation, and uniform-background fraction (25%) fixed. The matrix covered determinant-defined peak-density ratios of 16, 64, and 256 in 2D and 8D, including rotated anisotropic targets. The frozen StrataSCAN slice completed {density_audit['successful']}/{density_audit['attempted']} cells. Every target satisfied the prespecified discovery rule (purity $\\geq 0.90$, coverage $\\geq 0.10$) in every cell. Across the nine geometry--contrast settings, the median macro target F1 ranged from {density_audit['minimum_case_median_target_f1']:.3f} to {density_audit['maximum_case_median_target_f1']:.3f} [Fig. 2(a)]. This establishes capability across the declared density range, including 256-fold contrast; it is not an all-method superiority result because the corresponding 243-cell baseline matrix was not run.

We then separated background prevalence from local density contrast. Three fixed dense targets (100 points each) were embedded in progressively larger low-density backgrounds, increasing the known background fraction from 25% to 99% while preserving an empirical 16-neighbour signal/background density ratio of approximately 8.4--17.2. The all-method design contained 270 attempted cells (nine methods, six background fractions, five paired seeds), of which {noise_audit['successful']}/{noise_audit['attempted']} completed. StrataSCAN completed {noise_audit['stratascan_successful']}/{noise_audit['stratascan_attempted']} cells and discovered all three targets in every cell. Its level-wise median target F1 remained between {noise_audit['stratascan_minimum_level_median_target_f1']:.3f} and {noise_audit['stratascan_maximum_level_median_target_f1']:.3f}; at 99% background it retained target F1={noise_audit['stratascan_99_target_f1']:.3f} and achieved background F1={noise_audit['stratascan_99_noise_f1']:.4f} [Fig. 2(b)]. At this endpoint StrataSCAN had the highest median background F1 among the nine methods, and no comparator exceeded it on both target F1 and background F1 (Table II). Thus, when rare targets remain distinctly denser than their surroundings, StrataSCAN combines target recovery with explicit rejection of a dominant background.

The density-only run also exposed a cost: the median cell returned {density_audit['median_predicted_clusters']:.0f} predicted clusters for six targets, including {density_audit['median_unmatched_clusters']:.0f} unmatched clusters (maximum {density_audit['maximum_unmatched_clusters']}). We therefore describe the result as recovery with residual fragmentation, rather than exact partition reconstruction. Two AMD-DBSCAN cells at 99% background ended in memory-allocation errors; their quality values remain missing rather than being replaced by zero.

**One-sentence answer.** StrataSCAN discovered every target across the 16--256 density-contrast matrix and, with strong local target--background separation preserved, recovered all rare targets through 99% background while providing the best background rejection at that endpoint.

**Figure 2 caption.** *Density adaptation and rare-target recovery.* (a) StrataSCAN macro target F1 across determinant-defined peak-density ratios; points are medians over three seeds and bars show the full seed range. Only StrataSCAN was executed for the complete density-contrast matrix. (b) StrataSCAN target and known-background F1 as low-density background rises around three fixed 100-point targets; points are medians over five paired seeds and bands show the full seed range. The full nine-method 99% endpoint is reported in Table II.

### Evidence boundary and exclusions

- The density-contrast evidence is StrataSCAN-only (27/27), so it supports capability but not comparative superiority.
- The rare-dense evidence is the complete nine-method matrix (270 attempted; 268 successful). Failed AMD-DBSCAN executions remain in the completion denominator and are not zero-filled.
- The result is scoped to global low-density background at preserved local contrast. The older locally overlapping 99% moons construction simultaneously reduced each target to 100 points and collapsed the local density ratio to about 1.18:1; that failure belongs in Limitations, not in this advantage-focused Results subsection.
- Gaia and the abandoned optimization study are excluded.
"""
    path = output / "rq2_subsection.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-results", type=Path, required=True)
    parser.add_argument("--noise-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    density_frame = pd.read_csv(args.density_results)
    noise_frame = pd.read_csv(args.noise_results)
    density, density_audit = summarize_density(density_frame)
    noise, strata_noise, endpoint, noise_audit = summarize_noise(noise_frame)

    density.to_csv(args.output / "rq2_density_summary.csv", index=False)
    noise.to_csv(args.output / "rq2_rare_dense_all_methods_summary.csv", index=False)
    strata_noise.to_csv(args.output / "rq2_rare_dense_stratascan_summary.csv", index=False)
    endpoint.to_csv(args.output / "rq2_noise99_all_methods.csv", index=False)
    plot_figure(density, strata_noise, args.output)
    write_endpoint_table(endpoint, args.output)
    write_subsection(density_audit, noise_audit, args.output)

    provenance = {
        "schema_version": 1,
        "sources": {
            str(args.density_results): sha256(args.density_results),
            str(args.noise_results): sha256(args.noise_results),
        },
        "density_audit": density_audit,
        "noise_audit": noise_audit,
        "claim_scope": {
            "density": "StrataSCAN capability only; no all-method comparison",
            "rare_dense": "complete nine-method paired comparison",
            "excluded": ["Gaia", "abandoned optimization study", "locally overlapping 99% moons from main Results"],
        },
    }
    (args.output / "rq2_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
