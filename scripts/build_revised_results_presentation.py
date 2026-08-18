from __future__ import annotations

"""Build the compact, AMD-free Results tables and IEEE-sized figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript/oedm2026"
FIGURES = MANUSCRIPT / "final_figures"
TABLES = MANUSCRIPT / "tables"
RQ2 = ROOT / "outputs/rq2_density_noise"
RQ4 = ROOT / "outputs/rq4_cytometry"
SYNTHETIC = ROOT / "results/published/v0.2.4/target-discovery-v1/synthetic_release.csv"
CYTOMETRY = ROOT / "results/published/v0.2.4/target-discovery-v1/cytometry.csv"

METHODS = [
    "StrataSCAN", "HDBSCAN", "kNN+Leiden", "SNN-DBSCAN", "DBSCAN",
    "kNN-DBSCAN", "OPTICS", "VDBSCAN-2007",
]
TABLE_METHODS = [
    "StrataSCAN", "HDBSCAN", "VDBSCAN-2007", "kNN+Leiden",
    "DBSCAN", "kNN-DBSCAN", "OPTICS", "SNN-DBSCAN",
]
COLOR = "#8b1e3f"
BLUE = "#24758a"


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def table_i() -> None:
    frame = pd.read_csv(SYNTHETIC, low_memory=False)
    frame = frame.loc[
        frame["status"].eq("ok")
        & frame["seed"].isin([211, 223, 227])
        & ~frame["method"].eq("AMD-DBSCAN")
    ]
    summary = frame.groupby("method", observed=True).agg(
        mean_f1=("macro_target_f1", "mean"),
        sd_f1=("macro_target_f1", "std"),
        discovery=("target_discovery_rate", "mean"),
        cells=("macro_target_f1", "size"),
    ).reindex(TABLE_METHODS)
    if summary["cells"].ne(21).any():
        raise ValueError("Table I requires 21 successful fresh cells per displayed method")
    lines = [
        r"\begin{table}[t]", r"\centering", r"\caption{Synthetic target recovery on three fresh seeds across seven families.}",
        r"\label{tab:synthetic-fresh}", r"\setlength{\tabcolsep}{4.2pt}", r"\begin{tabular}{lcc}", r"\toprule",
        r"Method & Target F1 & Discovery \\", r"\midrule",
    ]
    for method, row in summary.iterrows():
        label = method.replace("kNN", r"$k$NN")
        f1 = rf"${row.mean_f1:.3f}\pm{row.sd_f1:.3f}$"
        discovery = f"{row.discovery:.3f}"
        if method == "StrataSCAN":
            label, f1, discovery = rf"\textbf{{{label}}}", rf"\textbf{{{f1}}}", rf"\textbf{{{discovery}}}"
        lines.append(f"{label} & {f1} & {discovery} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{0.98\columnwidth}\footnotesize Target F1 is the mean $\pm$ sample SD across 21 fresh cells. Discovery is the mean fraction of reference targets matched at purity $\geq0.90$ and coverage $\geq0.10$.\end{minipage}",
        r"\end{table}",
    ]
    (TABLES / "table_rq1_synthetic_fresh.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_ii() -> None:
    frame = pd.read_csv(RQ2 / "rq2_noise99_all_methods.csv")
    frame = frame.loc[~frame["method"].eq("AMD-DBSCAN")].set_index("method").reindex(TABLE_METHODS)
    if frame[["macro_target_f1_median", "noise_evidence_f1_median"]].isna().any().any():
        raise ValueError("Table II contains missing displayed endpoint scores")
    lines = [
        r"\begin{table}[t]", r"\centering", r"\caption{Recovery at 99\% low-density background. Medians over five paired seeds.}",
        r"\label{tab:rare-dense-99}", r"\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{lcc}", r"\toprule",
        r"Method & Target F1 & Background F1 \\", r"\midrule",
    ]
    for method, row in frame.iterrows():
        label = method.replace("kNN", r"$k$NN")
        a, b = f"{row.macro_target_f1_median:.3f}", f"{row.noise_evidence_f1_median:.3f}"
        if method == "StrataSCAN":
            label, a, b = rf"\textbf{{{label}}}", rf"\textbf{{{a}}}", rf"\textbf{{{b}}}"
        lines.append(f"{label} & {a} & {b} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "table_rq2_noise99.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure_2() -> None:
    density = pd.read_csv(RQ2 / "rq2_density_summary.csv")
    noise = pd.read_csv(RQ2 / "rq2_rare_dense_stratascan_summary.csv")
    row_specs = [("2-D isotropic", 2, "isotropic"), ("8-D isotropic", 8, "isotropic"),
                 ("2-D anisotropic", 2, "anisotropic"), ("8-D anisotropic", 8, "anisotropic")]
    ratios = [16, 64, 256]
    matrix = np.full((len(row_specs), len(ratios)), np.nan)
    for i, (_, dim, shape) in enumerate(row_specs):
        for j, ratio in enumerate(ratios):
            hit = density.loc[(density.dimension == dim) & (density["shape"] == shape) & (density.density_ratio == ratio)]
            if len(hit) == 1:
                matrix[i, j] = float(hit.iloc[0].macro_target_f1_median)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35), gridspec_kw={"width_ratios": [1.05, 1.15]})
    ax = axes[0]
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, vmin=.70, vmax=1.0, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(3), [r"$16\times$", r"$64\times$", r"$256\times$"])
    ax.set_yticks(range(4), [x[0] for x in row_specs])
    ax.set_xlabel("Peak-density contrast")
    ax.set_title("(a) Six targets; 25% background", loc="left", fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = "not tested" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.3f}"
            color = "#666666" if np.isnan(matrix[i, j]) else ("white" if matrix[i, j] < .82 else "black")
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color=color)
    cb = fig.colorbar(image, ax=ax, fraction=.046, pad=.03)
    cb.set_label("Median target F1", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ax = axes[1]
    x = noise["noise_fraction"].to_numpy(float) * 100
    for prefix, label, color, marker, ls in [
        ("macro_target_f1", "Target F1", COLOR, "o", "-"),
        ("noise_evidence_f1", "Known-background F1", BLUE, "s", "--"),
    ]:
        med = noise[f"{prefix}_median"].to_numpy(float)
        lo = noise[f"{prefix}_min"].to_numpy(float)
        hi = noise[f"{prefix}_max"].to_numpy(float)
        ax.errorbar(x, med, yerr=np.vstack([med - lo, hi - med]), color=color, marker=marker,
                    linestyle=ls, lw=1.3, ms=3.5, capsize=2, label=label)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(x)
    ax.set_xlabel("Low-density background (%)")
    ax.set_ylabel("F1")
    ax.set_title("(b) Three fixed dense targets", loc="left", fontsize=8)
    ax.text(.02, .62, "Discovery: 3/3 targets in every run", transform=ax.transAxes, fontsize=6.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.grid(axis="y", color="#dddddd", lw=.5)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig2_density_noise")


def _merge_aware_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    run = ROOT / "results/runs/v0.2.4-target-discovery-v1-cytometry"
    rows: list[dict[str, object]] = []
    for _, row in frame.loc[frame.status.eq("ok")].iterrows():
        path = run / str(row.evaluation_artifact_path)
        with np.load(path, allow_pickle=False) as state:
            contingency = state["target_predicted_contingency"].astype(float)
            target_sizes = state["target_sizes"].astype(float)
            predicted_sizes = state["predicted_sizes"].astype(float)
        denominator = target_sizes[:, None] + predicted_sizes[None, :]
        pair_f1 = np.divide(2 * contingency, denominator, out=np.zeros_like(contingency), where=denominator > 0)
        owners = np.argmax(pair_f1, axis=0)
        target_scores = []
        for target in range(len(target_sizes)):
            use = (owners == target) & (contingency[target] > 0)
            tp = float(contingency[target, use].sum())
            predicted = float(predicted_sizes[use].sum())
            purity = tp / predicted if predicted else 0.0
            coverage = tp / target_sizes[target] if target_sizes[target] else 0.0
            target_scores.append(2 * purity * coverage / (purity + coverage) if purity + coverage else 0.0)
        rows.append({"dataset_id": row.dataset_id, "method": row.method,
                     "one_to_one_f1": float(row.macro_target_f1),
                     "merge_aware_f1": float(np.mean(target_scores))})
    return pd.DataFrame(rows)


def figure_4() -> None:
    frame = pd.read_csv(CYTOMETRY, low_memory=False)
    frame = frame.loc[~frame.method.eq("AMD-DBSCAN")].copy()
    datasets = ["levine", "mosmann", "nilsson"] + [f"samusik_{i:02d}" for i in range(1, 11)]
    values = np.full((len(METHODS), len(datasets)), np.nan)
    status = np.empty_like(values, dtype=object)
    for i, method in enumerate(METHODS):
        for j, dataset in enumerate(datasets):
            row = frame.loc[(frame.method == method) & (frame.dataset_id == dataset)]
            if len(row) != 1:
                raise ValueError(f"cytometry cell is not unique: {method}/{dataset}")
            status[i, j] = row.iloc[0].status
            if row.iloc[0].status == "ok":
                values[i, j] = float(row.iloc[0].macro_target_f1)

    fig, ax = plt.subplots(figsize=(7.05, 2.65))
    image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    labels = ["Levine", "Mosmann", "Nilsson"] + [f"S{i}" for i in range(1, 11)]
    ax.set_xticks(range(len(datasets)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(METHODS)), [m.replace("kNN", r"$k$NN") for m in METHODS])
    ax.axvline(2.5, color="white", lw=2)
    ax.add_patch(plt.Rectangle((-.5, -.5), len(datasets), 1, fill=False, edgecolor=COLOR, lw=1.8))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                ax.text(j, i, r"$\times$", ha="center", va="center", color="#b2182b", fontsize=7)
            else:
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        color="white" if values[i, j] > .55 else "black", fontsize=5.5)
    ax.set_xlabel("Cytometry dataset (S1--S10: Samusik samples)")
    ax.set_title("One-to-one macro target F1; absolute values", fontsize=8)
    cb = fig.colorbar(image, ax=ax, fraction=.025, pad=.02)
    cb.set_label("Target F1", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    _save(fig, "fig4_cytometry")

    sensitivity = _merge_aware_sensitivity(frame)
    sensitivity.to_csv(RQ4 / "merge_aware_sensitivity.csv", index=False)
    strata = sensitivity.loc[sensitivity.method.eq("StrataSCAN")]
    audit = {
        "schema_version": 1,
        "purpose": "exploratory oversegmentation-tolerant sensitivity; not a replacement primary endpoint",
        "constraint": "each predicted component is assigned to at most one target before components are unioned",
        "stratascan_one_to_one_median": float(strata.one_to_one_f1.median()),
        "stratascan_merge_aware_median": float(strata.merge_aware_f1.median()),
        "decision": "retain the preregistered one-to-one target F1 because the sensitivity does not resolve the low absolute biological scores",
    }
    (RQ4 / "merge_aware_sensitivity.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    table_i()
    table_ii()
    figure_2()
    figure_4()
    print("built revised Results tables and figures (AMD excluded from displayed comparisons)")


if __name__ == "__main__":
    main()
