from __future__ import annotations

"""Generate figures only from the canonical target-discovery-v1 freezes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "final_figures"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_manuscript_statistics import (  # noqa: E402
    EVIDENCE_ROOT,
    EvidenceDesign,
    EvidencePaths,
    STRATASCAN,
    SYNTHETIC_DIAGNOSTICS,
    TARGET_METRICS,
    _case,
    _ensure_n,
    _validate_sources,
)


METHOD_ORDER = list(EvidenceDesign.canonical().methods)
SYNTHETIC_CASES = list(EvidenceDesign.canonical().cases)
CASE_LABELS = {
    "multidensity_2d": "Multi-density", "ultrasparse_16d": "Ultra-sparse",
    "overlapping_density_16d": "Overlap-density", "moons_2d": "Moons",
    "rings_2d": "Rings", "overlap_8d": "Gaussian overlap",
    "imbalanced_16d": "Imbalanced",
}
METHOD_LABEL = {
    "AMD-DBSCAN": "AMD-inspired",
    "kNN+Leiden": r"$k$NN+Leiden",
}
METHOD_COLORS = {method: "#777777" for method in METHOD_ORDER} | {STRATASCAN: "#0072B2"}
COMPARISON_MIN_COMPLETION = 0.80
QUALITY_METRICS = [*TARGET_METRICS, *SYNTHETIC_DIAGNOSTICS]
CANONICAL_PROTOCOLS = (
    REPO / "benchmarks/evaluation_protocol.v1.json",
    REPO / "benchmarks/protocol.v0.2.4-target-discovery-v1-synthetic.json",
    REPO / "benchmarks/protocol.v0.2.4-target-discovery-v1-scaling.json",
    REPO / "benchmarks/protocol.v0.2.4-target-discovery-v1-noise-sweep.json",
    REPO / "benchmarks/protocol.v0.2.4-target-discovery-v1-biological.json",
    REPO / "benchmarks/protocol.v0.2.4-target-discovery-v1-gaia-stratascan.json",
)
CANONICAL_DATA_FILES = (
    "fig2_synthetic_validation-data.csv",
    "fig3_execution_envelope-scaling-data.csv",
    "fig4_biological_validation-data.csv",
    "figS3_synthetic_metric_matrix-data.csv",
    "figS4_biological_metric_matrix-data.csv",
    "figS5_gaia_validation-data.csv",
    "figS7_noise_factorial-data.csv",
)


def configure_ieee_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "legend.fontsize": 6.3,
        "axes.linewidth": .65, "lines.linewidth": 1.0, "figure.dpi": 160,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def method_label(method: str) -> str:
    return METHOD_LABEL.get(method, method)


def ensure_n(frame: pd.DataFrame) -> pd.DataFrame:
    return _ensure_n(frame)


def successful_quality(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Coerce quality fields and preserve missingness for failed executions."""
    result = frame.copy()
    successful = result["status"].eq("ok")
    for metric in metrics:
        if metric in result:
            result[metric] = pd.to_numeric(result[metric], errors="coerce")
            result.loc[~successful, metric] = np.nan
    return result


def penalize_quality_failures(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility name: the release policy leaves failed quality missing."""
    return successful_quality(frame, [metric for metric in QUALITY_METRICS if metric in frame])


def comparison_eligibility(frame: pd.DataFrame) -> pd.Series:
    return frame.assign(ok=frame["status"].eq("ok")).groupby("method", observed=True)["ok"].mean()


def noise_rate_from_id(values: pd.Series) -> pd.Series:
    encoded = pd.to_numeric(values.astype(str).str.extract(r"noise(50|75|90|95|975|99)", expand=False))
    return encoded.where(encoded.le(99), encoded / 10) / 100.0


def joint_success_delta(frame: pd.DataFrame, baseline: str, metric: str = "macro_target_f1") -> pd.DataFrame:
    keys = ["dataset_id", "seed"]
    subset = frame.loc[frame["method"].isin([STRATASCAN, baseline]), [*keys, "method", "status", metric]].copy()
    values = subset.pivot(index=keys, columns="method", values=metric)
    statuses = subset.assign(ok=subset["status"].eq("ok")).pivot(index=keys, columns="method", values="ok")
    joint = statuses[STRATASCAN] & statuses[baseline]
    result = pd.DataFrame(index=values.index)
    result["joint_success"] = joint
    result["delta"] = (values[STRATASCAN] - values[baseline]).where(joint)
    return result.reset_index()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finish_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#BBBBBB", alpha=.35, linewidth=.45)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-.10, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=8.5)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".svg", ".png"):
        path = output / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=600 if suffix == ".png" else None)
        written.append(path)
    plt.close(fig)
    return written


def figure_method_overview(output: Path) -> tuple[list[Path], list[str]]:
    fig, ax = plt.subplots(figsize=(7.16, 1.65))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    boxes = [
        (.01, "Sparse kNN graph", "fixed ranks"), (.21, "Shell volumes", "Gamma mixture"),
        (.41, "Density strata", "semantic background"),
        (.61, "Exact event sweep", r"threshold code $\log |V_s|$"),
        (.81, "Non-merging growth", "clusters or abstention"),
    ]
    for index, (x, title, note) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, .27), .17, .48, boxstyle="round,pad=.012", fc="#F3F6F8", ec="#4A5568"))
        ax.text(x+.085, .57, title, ha="center", va="center", fontweight="bold", fontsize=7)
        ax.text(x+.085, .41, note, ha="center", va="center", fontsize=6.4, color="#374151")
        if index < len(boxes)-1:
            ax.add_patch(FancyArrowPatch((x+.17, .51), (x+.20, .51), arrowstyle="->", mutation_scale=8, color="#0072B2"))
    ax.text(.01, .91, "StrataSCAN: blockwise MDL-guided extraction", fontsize=8.5, fontweight="bold")
    return save_figure(fig, output, "fig1_method_pipeline"), []


def figure_synthetic_validation(frame: pd.DataFrame, output: Path) -> tuple[list[Path], list[str]]:
    selected = ensure_n(frame)
    selected = successful_quality(selected.loc[selected["n"].eq(20_000)], TARGET_METRICS)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), gridspec_kw={"width_ratios": [1.55, 1]})
    baselines = [method for method in METHOD_ORDER if method != STRATASCAN]
    for stage, offset, color in (("development", -.17, "#0072B2"), ("fresh", .17, "#D55E00")):
        part = selected.loc[selected["evidence_stage"].eq(stage)]
        medians, lows, highs, labels = [], [], [], []
        for baseline in baselines:
            paired = joint_success_delta(part, baseline)
            clean = paired["delta"].dropna()
            medians.append(float(clean.median())); lows.append(float(clean.quantile(.25))); highs.append(float(clean.quantile(.75)))
            labels.append(f"{method_label(baseline)}\n{len(clean)}/{len(paired)}")
            rows.append({"panel": "contrast", "evidence_stage": stage, "baseline": baseline,
                         "joint_successful_pairs": len(clean), "declared_pairs": len(paired),
                         "median_delta": medians[-1], "q1_delta": lows[-1], "q3_delta": highs[-1]})
        x = np.arange(len(baselines)) + offset
        axes[0].errorbar(x, medians, yerr=[np.array(medians)-np.array(lows), np.array(highs)-np.array(medians)], fmt="o", color=color, capsize=2, label=stage.capitalize())
        axes[0].set_xticks(np.arange(len(baselines)), labels, rotation=35, ha="right")
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set_ylabel(r"StrataSCAN $-$ baseline target F1")
    axes[0].set_title("Paired effects; joint/declared pairs", loc="left")
    axes[0].legend(frameon=False, ncol=2); finish_axis(axes[0]); panel_label(axes[0], "a")

    anchors = selected.loc[selected["method"].eq(STRATASCAN)].groupby("evidence_stage", observed=True)[["macro_target_f1", "target_discovery_rate"]].mean()
    for index, (stage, color) in enumerate((("development", "#0072B2"), ("fresh", "#D55E00"))):
        values = anchors.loc[stage, ["macro_target_f1", "target_discovery_rate"]].to_numpy(float)
        axes[1].bar(np.arange(2)+(-.18 if index == 0 else .18), values, .36, color=color, label=stage.capitalize())
        for metric, value in zip(("macro_target_f1", "target_discovery_rate"), values, strict=True):
            rows.append({"panel": "anchor", "evidence_stage": stage, "metric": metric, "mean": value})
    axes[1].set_xticks(range(2), ["Target F1", "Discovery rate"]); axes[1].set_ylim(0, 1)
    axes[1].set_title("Absolute StrataSCAN anchors", loc="left"); axes[1].legend(frameon=False)
    finish_axis(axes[1]); panel_label(axes[1], "b"); fig.tight_layout()
    pd.DataFrame(rows).to_csv(output / "fig2_synthetic_validation-data.csv", index=False)
    return save_figure(fig, output, "fig2_synthetic_validation"), []


def figure_execution_envelope(scaling: pd.DataFrame, output: Path) -> tuple[list[Path], list[str]]:
    scaled = successful_quality(ensure_n(scaling), [*TARGET_METRICS, *SYNTHETIC_DIAGNOSTICS])
    successful = scaled.loc[scaled["status"].eq("ok")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    grouped = successful.groupby("n", observed=True)
    for ax, metric, title, ylabel in (
        (axes[0], "runtime_seconds", "Runtime envelope", "Runtime (s)"),
        (axes[1], "process_peak_rss_mb", "Per-process memory envelope", "Peak RSS (MiB)"),
    ):
        median = grouped[metric].median(); low = grouped[metric].min(); high = grouped[metric].max()
        ax.plot(median.index, median.values, marker="o", color="#0072B2")
        ax.fill_between(median.index, low.reindex(median.index), high.reindex(median.index), color="#0072B2", alpha=.18)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Observations"); ax.set_ylabel(ylabel)
        ax.set_xticks([500_000, 1_000_000, 2_000_000, 5_000_000], ["0.5M", "1M", "2M", "5M"])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        compact_title = "Runtime; median and range" if ax is axes[0] else "Memory; median and range"
        ax.set_title(compact_title, loc="left", fontsize=7.2); finish_axis(ax); panel_label(ax, "a" if ax is axes[0] else "b")
    for metric, label, style in (("macro_target_f1", "Target F1", "-"), ("target_discovery_rate", "Discovery", "--"), ("noise_evidence_f1", "True-noise F1", ":"), ("pairwise_f1", "Pairwise F1", "-.")):
        curve = grouped[metric].median(); axes[2].plot(curve.index, curve.values, style, marker="o", markersize=3, label=label)
    axes[2].set_xscale("log"); axes[2].set_ylim(0, 1); axes[2].set_xlabel("Observations")
    axes[2].set_xticks([500_000, 1_000_000, 2_000_000, 5_000_000], ["0.5M", "1M", "2M", "5M"])
    axes[2].xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axes[2].set_title("Quality envelope", loc="left", fontsize=7.2); axes[2].legend(frameon=False, fontsize=5.7)
    finish_axis(axes[2]); panel_label(axes[2], "c"); fig.tight_layout()
    scaled.to_csv(output / "fig3_execution_envelope-scaling-data.csv", index=False)
    return save_figure(fig, output, "fig3_execution_envelope"), []


def _study(dataset: str) -> str:
    return "samusik" if dataset.startswith("samusik_") else dataset


def _biological_aggregates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = successful_quality(frame, TARGET_METRICS).loc[lambda x: x["status"].eq("ok")].copy()
    equal_dataset = ok.groupby("method", observed=True)[list(TARGET_METRICS)].mean()
    ok["study"] = ok["dataset_id"].map(_study)
    equal_study = ok.groupby(["method", "study"], observed=True)[list(TARGET_METRICS)].mean().groupby("method").mean()
    return equal_dataset, equal_study


def figure_biological_validation(frame: pd.DataFrame, output: Path) -> tuple[list[Path], list[str]]:
    expected = len(EvidenceDesign.canonical().biological_datasets) * len(METHOD_ORDER)
    observed = len(frame[["dataset_id", "method"]].drop_duplicates())
    if observed != expected:
        return [], [f"biological matrix missing {expected-observed} cells"]
    equal_dataset, equal_study = _biological_aggregates(frame)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55)); methods = METHOD_ORDER; y = np.arange(len(methods))
    equal_dataset = equal_dataset.reindex(methods)
    equal_study = equal_study.reindex(methods)
    axes[0].barh(y-.18, equal_dataset["macro_target_f1"], .36, label="Equal dataset", color="#0072B2")
    axes[0].barh(y+.18, equal_study["macro_target_f1"], .36, label="Equal study", color="#D55E00")
    unavailable = equal_dataset["macro_target_f1"].isna() & equal_study["macro_target_f1"].isna()
    for index in np.flatnonzero(unavailable.to_numpy()):
        axes[0].plot(.015, y[index], marker="x", color="#9B2226", clip_on=False)
        axes[0].text(.035, y[index], "NA", va="center", fontsize=5.4, color="#9B2226")
    axes[0].set_yticks(y, [method_label(m) for m in methods]); axes[0].set_xlim(0, 1)
    axes[0].set_title("One-to-one target F1", loc="left"); axes[0].legend(frameon=False, fontsize=5.8)
    finish_axis(axes[0], "x"); panel_label(axes[0], "a")
    for metric, marker, label in (("macro_target_purity", "o", "Purity"), ("macro_target_coverage", "s", "Coverage"), ("target_discovery_rate", "^", "Discovery")):
        axes[1].plot(equal_dataset[metric], y, marker=marker, linestyle="none", label=label)
    for index in np.flatnonzero(equal_dataset[list(TARGET_METRICS)].isna().all(axis=1).to_numpy()):
        axes[1].plot(.015, y[index], marker="x", color="#9B2226", clip_on=False)
        axes[1].text(.035, y[index], "NA", va="center", fontsize=5.4, color="#9B2226")
    axes[1].set_yticks(y, []); axes[1].set_xlim(0, 1); axes[1].set_title("Equal-dataset components", loc="left")
    axes[1].legend(frameon=False); finish_axis(axes[1], "x"); panel_label(axes[1], "b")
    successful = frame.loc[frame["status"].eq("ok")]
    completion = frame.assign(ok=frame["status"].eq("ok")).groupby("method", observed=True)["ok"].sum().reindex(methods, fill_value=0)
    declared = frame.groupby("method", observed=True).size().reindex(methods, fill_value=0)
    burden = successful.groupby("method", observed=True)["candidate_clusters_per_target"].median().reindex(methods)
    axes[2].barh(y, completion, color=[METHOD_COLORS[m] for m in methods])
    for index, method in enumerate(methods):
        burden_text = "burden NA" if pd.isna(burden.loc[method]) else f"burden {burden.loc[method]:.1f}"
        axes[2].text(max(.1, completion.loc[method] + .15), y[index], f"{completion.loc[method]}/{declared.loc[method]}; {burden_text}", va="center", fontsize=4.8)
    axes[2].set_yticks(y, []); axes[2].set_xlim(0, max(14.5, float(completion.max()) + 4))
    axes[2].set_xlabel("Successful datasets"); axes[2].set_title("Completion; burden uses successes only", loc="left")
    finish_axis(axes[2], "x"); panel_label(axes[2], "c"); fig.tight_layout()
    successful_quality(frame, TARGET_METRICS).to_csv(output / "fig4_biological_validation-data.csv", index=False)
    return save_figure(fig, output, "fig4_biological_validation"), []


def figure_synthetic_metric_matrix(frame: pd.DataFrame, output: Path) -> tuple[list[Path], list[str]]:
    selected = successful_quality(ensure_n(frame), [*TARGET_METRICS, *SYNTHETIC_DIAGNOSTICS])
    selected = selected.loc[selected["evidence_stage"].eq("development") & selected["n"].eq(20_000)].copy(); selected["case"] = _case(selected["dataset_id"])
    metrics = [("macro_target_f1", "Target F1"), ("target_discovery_rate", "Discovery rate"), ("noise_evidence_f1", "True-noise F1")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        table = selected.pivot_table(index="method", columns="case", values=metric, aggfunc="mean").reindex(index=METHOD_ORDER, columns=SYNTHETIC_CASES)
        image = ax.imshow(table, vmin=0, vmax=1, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(SYNTHETIC_CASES)), [CASE_LABELS[c] for c in SYNTHETIC_CASES], rotation=55, ha="right")
        ax.set_yticks(range(len(METHOD_ORDER)), [method_label(m) for m in METHOD_ORDER] if ax is axes[0] else []); ax.set_title(title, loc="left")
    fig.colorbar(image, ax=axes, fraction=.018, pad=.02)
    selected.to_csv(output / "figS3_synthetic_metric_matrix-data.csv", index=False)
    return save_figure(fig, output, "figS3_synthetic_metric_matrix"), []


def figure_biological_metric_matrix(frame: pd.DataFrame, output: Path) -> tuple[list[Path], list[str]]:
    successful = successful_quality(frame, TARGET_METRICS).loc[lambda x: x["status"].eq("ok")]
    datasets = list(EvidenceDesign.canonical().biological_datasets)
    table = successful.pivot(index="method", columns="dataset_id", values="macro_target_f1").reindex(index=METHOD_ORDER, columns=datasets)
    fig, ax = plt.subplots(figsize=(7.16, 2.65)); image = ax.imshow(table, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(datasets)), datasets, rotation=55, ha="right"); ax.set_yticks(range(len(METHOD_ORDER)), [method_label(m) for m in METHOD_ORDER])
    ax.set_title("Biological one-to-one target F1 by dataset", loc="left"); fig.colorbar(image, ax=ax, fraction=.02, pad=.02)
    successful.to_csv(output / "figS4_biological_metric_matrix-data.csv", index=False)
    return save_figure(fig, output, "figS4_biological_metric_matrix"), []


def figure_gaia_validation(frame: pd.DataFrame, output: Path) -> tuple[list[Path], list[str]]:
    successful = successful_quality(frame, TARGET_METRICS).loc[lambda x: x["status"].eq("ok")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45)); methods = METHOD_ORDER
    discovery = successful.groupby("method", observed=True)["target_discovery_rate"].mean().reindex(methods)
    axes[0].barh(range(len(methods)), discovery, color=[METHOD_COLORS[m] for m in methods]); axes[0].set_yticks(range(len(methods)), [method_label(m) for m in methods]); axes[0].set_xlim(0, 1)
    axes[0].set_title(r"Discovery ($P\geq.90,Q\geq.10$)", loc="left"); finish_axis(axes[0], "x"); panel_label(axes[0], "a")
    for metric, marker, label in (("macro_target_f1", "o", "F1"), ("macro_target_purity", "s", "Purity"), ("macro_target_coverage", "^", "Coverage")):
        median = successful.groupby("method", observed=True)[metric].median().reindex(methods); axes[1].plot(median, range(len(methods)), marker=marker, linestyle="none", label=label)
    axes[1].set_yticks(range(len(methods)), []); axes[1].set_xlim(0, 1); axes[1].set_title("Successful-field medians", loc="left"); axes[1].legend(frameon=False)
    finish_axis(axes[1], "x"); panel_label(axes[1], "b")
    burden = successful.groupby("method", observed=True)["unmatched_predicted_cluster_count"].median().reindex(methods)
    completion = frame.assign(ok=frame["status"].eq("ok")).groupby("method", observed=True)["ok"].sum().reindex(methods)
    axes[2].scatter(burden, completion, color=[METHOD_COLORS[m] for m in methods], s=18); axes[2].set_xlabel("Median unmatched candidates"); axes[2].set_ylabel("Successful fields / 359")
    axes[2].set_title("Burden and completion", loc="left"); finish_axis(axes[2]); panel_label(axes[2], "c"); fig.tight_layout()
    successful.to_csv(output / "figS5_gaia_validation-data.csv", index=False)
    return save_figure(fig, output, "figS5_gaia_validation"), []


def figure_noise_factorial(frame: pd.DataFrame, output: Path) -> tuple[list[Path], list[str]]:
    prepared = successful_quality(frame, [*TARGET_METRICS, *SYNTHETIC_DIAGNOSTICS]).copy(); prepared["noise_fraction"] = noise_rate_from_id(prepared["dataset_id"])
    metrics = [("macro_target_f1", "Target F1"), ("target_discovery_rate", "Discovery"), ("noise_evidence_f1", "True-noise F1"), ("pairwise_f1", "Pairwise F1")]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.2), sharey=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        summary = prepared.loc[prepared["status"].eq("ok")].groupby(["method", "noise_fraction"], observed=True)[metric].mean()
        for method in METHOD_ORDER:
            curve = summary.xs(method, level="method"); ax.plot(curve.index, curve.values, marker="o", markersize=2.5, label=method.replace("StrataSCAN-", ""))
        ax.set_title(title, loc="left"); ax.set_xlabel("Known noise fraction"); ax.set_ylim(0, 1); finish_axis(ax)
    axes[0].set_ylabel("Score"); axes[-1].legend(frameon=False, fontsize=4.7, loc="lower left"); fig.tight_layout()
    prepared.to_csv(output / "figS7_noise_factorial-data.csv", index=False)
    return save_figure(fig, output, "figS7_noise_factorial"), []


def write_manifest(output: Path, figures: Iterable[Path], paths: EvidencePaths) -> None:
    sources = []
    for field in EvidencePaths.__dataclass_fields__:
        source = getattr(paths, field); sources.extend([source, source.with_name(source.stem + "-provenance.json")])
    for protocol in CANONICAL_PROTOCOLS:
        if not protocol.is_file():
            raise FileNotFoundError(f"required canonical protocol is missing: {protocol}")
        sources.append(protocol)
    ancillary = [output / name for name in CANONICAL_DATA_FILES]
    missing_ancillary = [path for path in ancillary if not path.is_file()]
    if missing_ancillary:
        raise FileNotFoundError(f"required canonical figure data are missing: {missing_ancillary}")
    manifest = {
        "schema_version": 2, "evaluation_protocol_version": "target-discovery-v1",
        "matching_strategy": "hungarian", "matching_objective": "pairwise_f1",
        "discovery_thresholds": {"purity": .90, "coverage": .10},
        "failure_policy": "failed executions retain missing quality; paired effects use joint successes",
        "non_synthetic_background_policy": "background abstention is diagnostic, not true-noise evidence",
        "sources": [{"path": path.relative_to(REPO).as_posix(), "exists": True, "sha256": sha256(path)} for path in sources],
        "figures": sorted(path.name for path in figures),
        "ancillary_files": sorted(path.name for path in ancillary),
        "placeholders": [],
        "excluded_historical_claim_blocks": ["model specification", "solver predecessor"],
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-discovery-v1 manuscript figures")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    args = parser.parse_args(); configure_ieee_style(); args.output.mkdir(parents=True, exist_ok=True)
    paths = EvidencePaths.canonical(args.evidence_root)
    frames = _validate_sources(paths, EvidenceDesign.canonical())
    figures = []
    obsolete_outputs = (
        list(args.output.glob("figS1a_algorithm_behavior_grid*"))
        + list(args.output.glob("figS1b_algorithm_behavior_grid*"))
        + list(args.output.glob("figS2_*"))
        + list(args.output.glob("figS6_*"))
        + [args.output / "fig3_execution_envelope-data.csv"]
    )
    for obsolete in obsolete_outputs:
        if obsolete.is_file(): obsolete.unlink()
    written, issues = figure_method_overview(args.output); figures.extend(written)
    jobs = [
        (figure_synthetic_validation, (frames["synthetic_release"], args.output)),
        (figure_execution_envelope, (frames["synthetic_scaling"], args.output)),
        (figure_biological_validation, (frames["cytometry"], args.output)),
        (figure_synthetic_metric_matrix, (frames["synthetic_release"], args.output)),
        (figure_biological_metric_matrix, (frames["cytometry"], args.output)),
        (figure_gaia_validation, (frames["gaia"], args.output)),
        (figure_noise_factorial, (frames["noise_factorial"], args.output)),
    ]
    for builder, builder_args in jobs:
        written, issues = builder(*builder_args)
        if issues: raise SystemExit("; ".join(issues))
        figures.extend(written)
    write_manifest(args.output, figures, paths)
    print(json.dumps({"output": str(args.output), "files": len(figures), "placeholders": []}, indent=2))


if __name__ == "__main__":
    main()
