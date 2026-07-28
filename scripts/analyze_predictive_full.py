from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


QUALITY_METRICS = ("macro_target_f1", "pairwise_f1", "binary_macro_f1")
NEW_METHOD = "PredictiveMultiscale"


def numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def select_new(raw: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.loc[raw["status"].eq("ok")].copy()
    numeric(
        raw,
        QUALITY_METRICS
        + (
            "q",
            "n",
            "seed",
            "full_primary_seconds",
            "graph_wall_seconds",
            "stratification_seconds",
            "core_seconds",
            "best_cluster_f1",
        ),
    )
    raw["runtime_seconds"] = (
        raw["graph_wall_seconds"] + raw["stratification_seconds"] + raw["core_seconds"]
    )
    sweep = raw.loc[raw["dataset_id"].isin(["mosmann", "nilsson"])].copy()
    sweep.to_csv(output / "q-sweep-mosmann-nilsson.csv", index=False)
    sweep_targets = []
    for row in sweep.itertuples():
        for target in json.loads(row.target_matches_json):
            sweep_targets.append(
                {
                    "dataset_id": row.dataset_id,
                    "q": row.q,
                    **target,
                }
            )
    pd.DataFrame(sweep_targets).to_csv(output / "q-sweep-targets.csv", index=False)
    chosen_q = {}
    for dataset_id, part in sweep.groupby("dataset_id"):
        best_value = part["macro_target_f1"].max()
        candidates = part.loc[np.isclose(part["macro_target_f1"], best_value)].copy()
        candidates["q_distance"] = (candidates["q"] - 0.95).abs()
        chosen_q[dataset_id] = float(
            candidates.sort_values(["pairwise_f1", "q_distance"], ascending=[False, True]).iloc[0]["q"]
        )
    selected = raw.loc[
        raw.apply(
            lambda row: np.isclose(row["q"], chosen_q.get(row["dataset_id"], 0.95)), axis=1
        )
    ].copy()
    selected["method"] = NEW_METHOD
    selected["canonical_id"] = selected["dataset_id"]
    gaia = selected["suite"].eq("gaia")
    selected.loc[gaia, "canonical_id"] = (
        selected.loc[gaia, "dataset_id"].astype(str) + "__unsupervised_field"
    )
    synthetic = selected["suite"].eq("synthetic")
    selected.loc[synthetic, "canonical_id"] = selected.loc[synthetic].apply(
        lambda row: (
            f"{row['dataset_id']}__"
            f"{'scaling' if row['tier'] == 'scalability' else row['tier']}__n{int(row['n'])}"
        ),
        axis=1,
    )
    selection = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "selected_q": q,
                "selection_endpoint": "macro_target_f1",
                "selection_scope": "same-dataset supervised sensitivity analysis",
            }
            for dataset_id, q in sorted(chosen_q.items())
        ]
    )
    selection.to_csv(output / "selected-q.csv", index=False)
    selected.to_csv(output / "selected-new-version-results.csv", index=False)
    real_targets = []
    for row in selected.loc[~selected["suite"].eq("synthetic")].itertuples():
        for target in json.loads(row.target_matches_json):
            real_targets.append(
                {
                    "suite": row.suite,
                    "dataset_id": row.dataset_id,
                    "q": row.q,
                    **target,
                }
            )
    pd.DataFrame(real_targets).to_csv(output / "real-target-results.csv", index=False)
    return selected, selection


def prepare_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    numeric(frame, QUALITY_METRICS + ("runtime_seconds", "best_cluster_f1", "seed"))
    for metric in QUALITY_METRICS + ("best_cluster_f1",):
        if metric in frame:
            frame.loc[~frame["status"].eq("ok"), metric] = 0.0
    frame["canonical_id"] = frame["dataset_id"].astype(str)
    return frame


def collapse(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = [
        column
        for column in QUALITY_METRICS + ("best_cluster_f1", "runtime_seconds")
        if column in frame
    ]
    return frame.groupby(keys, as_index=False)[columns].median()


def build_tables(
    selected: pd.DataFrame,
    reference: pd.DataFrame,
    samusik_reference: pd.DataFrame,
    output: Path,
) -> dict[str, pd.DataFrame]:
    new_collapsed = collapse(selected, ["suite", "canonical_id", "method"])
    ref_collapsed = collapse(reference, ["suite", "canonical_id", "method"])
    combined = pd.concat([ref_collapsed, new_collapsed], ignore_index=True, sort=False)

    synthetic = combined.loc[combined["suite"].eq("synthetic")].copy()
    synthetic["tier"] = synthetic["canonical_id"].str.extract(r"__(quality|scaling)__")
    synthetic["n"] = pd.to_numeric(
        synthetic["canonical_id"].str.extract(r"__n(\d+)$")[0]
    )
    synthetic["case"] = synthetic["canonical_id"].str.replace(
        r"__(quality|scaling)__n\d+$", "", regex=True
    )
    synthetic.to_csv(output / "synthetic-condition-results.csv", index=False)
    synthetic_quality = (
        synthetic.loc[synthetic["tier"].eq("quality")]
        .groupby("method", as_index=False)
        .agg(
            conditions=("canonical_id", "size"),
            mean_macro_target_f1=("macro_target_f1", "mean"),
            median_macro_target_f1=("macro_target_f1", "median"),
            mean_pairwise_f1=("pairwise_f1", "mean"),
            median_pairwise_f1=("pairwise_f1", "median"),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .sort_values("mean_macro_target_f1", ascending=False)
    )
    synthetic_quality.to_csv(output / "synthetic-quality-summary.csv", index=False)
    scalability = (
        synthetic.loc[synthetic["tier"].eq("scaling")]
        .groupby(["method", "n"], as_index=False)
        .agg(
            completed_conditions=("runtime_seconds", "count"),
            median_runtime_seconds=("runtime_seconds", "median"),
            median_macro_target_f1=("macro_target_f1", "median"),
        )
    )
    scalability.to_csv(output / "synthetic-scalability-summary.csv", index=False)

    cytometry = combined.loc[
        combined["suite"].eq("cytometry")
        & ~combined["canonical_id"].str.startswith("samusik_")
    ].copy()
    cytometry.to_csv(output / "cytometry-results.csv", index=False)
    cytometry_summary = (
        cytometry.groupby("method", as_index=False)
        .agg(
            datasets=("canonical_id", "size"),
            mean_macro_target_f1=("macro_target_f1", "mean"),
            median_macro_target_f1=("macro_target_f1", "median"),
            median_pairwise_f1=("pairwise_f1", "median"),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .sort_values("mean_macro_target_f1", ascending=False)
    )
    cytometry_summary.to_csv(output / "cytometry-summary.csv", index=False)

    gaia = combined.loc[combined["suite"].eq("gaia")].copy()
    gaia.to_csv(output / "gaia-field-results.csv", index=False)
    gaia_summary = (
        gaia.groupby("method", as_index=False)
        .agg(
            fields=("canonical_id", "size"),
            mean_best_cluster_f1=("best_cluster_f1", "mean"),
            median_best_cluster_f1=("best_cluster_f1", "median"),
            success_rate_f1_080=("best_cluster_f1", lambda x: np.mean(x >= 0.80)),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .sort_values("mean_best_cluster_f1", ascending=False)
    )
    gaia_summary.to_csv(output / "gaia-summary.csv", index=False)

    sam_new = selected.loc[selected["dataset_id"].str.startswith("samusik_")].copy()
    sam_new = collapse(sam_new, ["canonical_id", "method"])
    sam_ref = collapse(samusik_reference, ["canonical_id", "method"])
    samusik = pd.concat([sam_ref, sam_new], ignore_index=True, sort=False)
    samusik.to_csv(output / "samusik-sample-results.csv", index=False)
    samusik_summary = (
        samusik.groupby("method", as_index=False)
        .agg(
            samples=("canonical_id", "size"),
            mean_macro_target_f1=("macro_target_f1", "mean"),
            median_macro_target_f1=("macro_target_f1", "median"),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .sort_values("mean_macro_target_f1", ascending=False)
    )
    samusik_summary.to_csv(output / "samusik-summary.csv", index=False)
    return {
        "synthetic": synthetic,
        "synthetic_quality": synthetic_quality,
        "scalability": scalability,
        "cytometry": cytometry,
        "cytometry_summary": cytometry_summary,
        "gaia": gaia,
        "gaia_summary": gaia_summary,
        "samusik": samusik,
        "samusik_summary": samusik_summary,
    }


def paired_comparisons(tables: dict[str, pd.DataFrame], output: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    specifications = (
        (
            "synthetic_quality",
            tables["synthetic"].loc[tables["synthetic"]["tier"].eq("quality")],
            "macro_target_f1",
        ),
        (
            "synthetic_scaling",
            tables["synthetic"].loc[tables["synthetic"]["tier"].eq("scaling")],
            "macro_target_f1",
        ),
        ("cytometry", tables["cytometry"], "macro_target_f1"),
        ("samusik", tables["samusik"], "macro_target_f1"),
        ("gaia", tables["gaia"], "best_cluster_f1"),
    )
    rows = []
    detail_frames = []
    for suite, frame, metric in specifications:
        new = frame.loc[frame["method"].eq(NEW_METHOD), ["canonical_id", metric]].rename(
            columns={metric: "new"}
        )
        old = frame.loc[frame["method"].eq("StrataSCAN"), ["canonical_id", metric]].rename(
            columns={metric: "old"}
        )
        paired = new.merge(old, on="canonical_id", validate="one_to_one")
        delta = paired["new"].to_numpy(float) - paired["old"].to_numpy(float)
        paired["suite"] = suite
        paired["endpoint"] = metric
        paired["delta"] = delta
        detail_frames.append(paired)
        draws = rng.choice(delta, size=(20_000, len(delta)), replace=True).mean(axis=1)
        rows.append(
            {
                "suite": suite,
                "endpoint": metric,
                "conditions": len(delta),
                "new_mean": float(paired["new"].mean()),
                "old_mean": float(paired["old"].mean()),
                "mean_delta": float(np.mean(delta)),
                "mean_delta_ci025": float(np.quantile(draws, 0.025)),
                "mean_delta_ci975": float(np.quantile(draws, 0.975)),
                "median_delta": float(np.median(delta)),
                "improved": int(np.sum(delta > 1e-12)),
                "degraded": int(np.sum(delta < -1e-12)),
                "unchanged": int(np.sum(np.abs(delta) <= 1e-12)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "paired-comparisons.csv", index=False)
    pd.concat(detail_frames, ignore_index=True).to_csv(
        output / "paired-condition-deltas.csv", index=False
    )
    return result


def plot_summary(raw: pd.DataFrame, tables: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    qdata = raw.loc[raw["dataset_id"].isin(["mosmann", "nilsson"])].copy()
    numeric(qdata, ("q", "macro_target_f1"))
    for dataset_id, part in qdata.groupby("dataset_id"):
        axes[0, 0].plot(part["q"], part["macro_target_f1"], marker="o", label=dataset_id)
    axes[0, 0].set(xlabel="q", ylabel="macro target F1", title="Targeted sparse-data q sweep")
    axes[0, 0].legend(frameon=False)

    quality = tables["synthetic_quality"].head(10).sort_values("mean_macro_target_f1")
    axes[0, 1].barh(quality["method"], quality["mean_macro_target_f1"])
    axes[0, 1].set(xlabel="mean macro target F1", title="Synthetic quality (21 conditions)")

    scaling = tables["scalability"]
    for method in (NEW_METHOD, "StrataSCAN", "HDBSCAN", "kNN+Leiden"):
        part = scaling.loc[scaling["method"].eq(method)].sort_values("n")
        if len(part):
            axes[1, 0].plot(
                part["n"], part["median_runtime_seconds"], marker="o", label=method
            )
    axes[1, 0].set(
        xlabel="n", ylabel="median runtime, s", title="Synthetic scalability", yscale="log"
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    real = pd.DataFrame(
        [
            {
                "suite": "Cytometry",
                "score": tables["cytometry_summary"].set_index("method").get(
                    "mean_macro_target_f1", pd.Series()
                ).get(NEW_METHOD, np.nan),
            },
            {
                "suite": "Gaia",
                "score": tables["gaia_summary"].set_index("method").get(
                    "mean_best_cluster_f1", pd.Series()
                ).get(NEW_METHOD, np.nan),
            },
            {
                "suite": "Samusik",
                "score": tables["samusik_summary"].set_index("method").get(
                    "mean_macro_target_f1", pd.Series()
                ).get(NEW_METHOD, np.nan),
            },
        ]
    )
    axes[1, 1].bar(real["suite"], real["score"], color="#E67E22")
    axes[1, 1].set(ylim=(0, 1), ylabel="mean target F1", title="New version on real data")
    fig.savefig(output / "benchmark-overview.png", dpi=190)
    plt.close(fig)


def value(table: pd.DataFrame, method: str, column: str) -> float:
    row = table.loc[table["method"].eq(method)]
    return float(row.iloc[0][column]) if len(row) else float("nan")


def write_report(
    selected: pd.DataFrame,
    selection: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    paired: pd.DataFrame,
    output: Path,
) -> None:
    sq = tables["synthetic_quality"]
    cy = tables["cytometry_summary"]
    ga = tables["gaia_summary"]
    sa = tables["samusik_summary"]
    scaling = tables["scalability"]
    q_lines = [
        f"- {row.dataset_id}: selected q={row.selected_q:.2f} by maximum same-dataset macro target F1."
        for row in selection.itertuples()
    ]
    scale_new = scaling.loc[scaling["method"].eq(NEW_METHOD)].sort_values("n")
    paired_index = paired.set_index("suite")
    syn_pair = paired_index.loc["synthetic_quality"]
    sam_pair = paired_index.loc["samusik"]
    gaia_pair = paired_index.loc["gaia"]
    gaia_runtime_ratio = value(ga, NEW_METHOD, "median_runtime_seconds") / value(
        ga, "StrataSCAN", "median_runtime_seconds"
    )
    synthetic_runtime_ratio = value(sq, NEW_METHOD, "median_runtime_seconds") / value(
        sq, "StrataSCAN", "median_runtime_seconds"
    )
    cytometry_lines = []
    for dataset_id in ("levine", "mosmann", "nilsson"):
        part = tables["cytometry"].loc[tables["cytometry"]["canonical_id"].eq(dataset_id)]
        cytometry_lines.append(
            f"- {dataset_id}: {value(part, NEW_METHOD, 'macro_target_f1'):.3f} versus "
            f"{value(part, 'StrataSCAN', 'macro_target_f1'):.3f}."
        )
    scale_lines = [
        f"- n={int(row.n):,}: median {row.median_runtime_seconds:.2f} s; "
        f"median macro target F1 {row.median_macro_target_f1:.3f}."
        for row in scale_new.itertuples()
    ]
    report = f"""# Full benchmark analysis — predictive multiscale StrataSCAN

## Scope

- Synthetic quality: seven scenarios, n=5k/20k/50k, five data seeds (105 runs).
- Synthetic scalability: the same seven scenarios at n=100k/200k (14 runs).
- Real cytometry: Levine, Mosmann and Nilsson.
- Samusik: all ten independently processed mouse samples.
- Gaia: all {int(selected.loc[selected['suite'].eq('gaia'), 'dataset_id'].nunique())} available benchmark fields.
- New-version default is q=0.95. Only Mosmann and Nilsson receive the requested q sweep from 0.30 to 0.99.

## Targeted q sensitivity

{chr(10).join(q_lines)}

This selection is a supervised sensitivity analysis on the same annotations used for scoring; it is evidence for a dataset-specific setting, not an unbiased generalization estimate.

## Synthetic quality

Across the 21 scenario-size conditions, the new version has mean macro target F1 **{value(sq, NEW_METHOD, 'mean_macro_target_f1'):.3f}** and median pairwise F1 **{value(sq, NEW_METHOD, 'median_pairwise_f1'):.3f}**. The current StrataSCAN reference has **{value(sq, 'StrataSCAN', 'mean_macro_target_f1'):.3f}** and **{value(sq, 'StrataSCAN', 'median_pairwise_f1'):.3f}**, respectively.

The paired mean macro-F1 delta is {syn_pair.mean_delta:+.3f} (condition-bootstrap 95% interval {syn_pair.mean_delta_ci025:+.3f} to {syn_pair.mean_delta_ci975:+.3f}); {int(syn_pair.improved)} conditions improve, {int(syn_pair.degraded)} degrade, and {int(syn_pair.unchanged)} are unchanged.

The median quality-run runtime is {synthetic_runtime_ratio:.2f}x the current StrataSCAN runtime.

## Scalability

{chr(10).join(scale_lines)}

Runtime is end-to-end estimator time excluding CSV loading. The predictive model-selection stage is included.

## Real datasets

- Cytometry (Levine/Mosmann/Nilsson): mean macro target F1 {value(cy, NEW_METHOD, 'mean_macro_target_f1'):.3f} for the new version versus {value(cy, 'StrataSCAN', 'mean_macro_target_f1'):.3f} for current StrataSCAN.
{chr(10).join(cytometry_lines)}
- Samusik (10 samples): mean macro target F1 {value(sa, NEW_METHOD, 'mean_macro_target_f1'):.3f} versus {value(sa, 'StrataSCAN', 'mean_macro_target_f1'):.3f}.
- Gaia: mean best-cluster F1 {value(ga, NEW_METHOD, 'mean_best_cluster_f1'):.3f}, median {value(ga, NEW_METHOD, 'median_best_cluster_f1'):.3f}, and F1>=0.80 success rate {value(ga, NEW_METHOD, 'success_rate_f1_080'):.3f}; current StrataSCAN is {value(ga, 'StrataSCAN', 'mean_best_cluster_f1'):.3f}, {value(ga, 'StrataSCAN', 'median_best_cluster_f1'):.3f}, and {value(ga, 'StrataSCAN', 'success_rate_f1_080'):.3f}.

Samusik's paired mean delta is {sam_pair.mean_delta:+.3f} (bootstrap 95% interval {sam_pair.mean_delta_ci025:+.3f} to {sam_pair.mean_delta_ci975:+.3f}); {int(sam_pair.improved)}/10 samples improve. Gaia's paired mean delta is {gaia_pair.mean_delta:+.3f} (bootstrap 95% interval {gaia_pair.mean_delta_ci025:+.3f} to {gaia_pair.mean_delta_ci975:+.3f}); {int(gaia_pair.improved)} fields improve, {int(gaia_pair.degraded)} degrade, and {int(gaia_pair.unchanged)} are unchanged.

Gaia median runtime is {gaia_runtime_ratio:.1f}x current StrataSCAN.

## Decision

Do not promote this predictive version as the universal default yet. Its synthetic mean gain is concentrated in a minority of conditions and the paired interval includes zero; Gaia has unchanged aggregate success, rare near-unit gains and losses, and a large runtime penalty. The strongest actionable leads are the rings improvement, the Nilsson q=0.30 recovery, and the modest Samusik gain. Mosmann is not solved by q tuning, and the moons/Gaia failure flips need a stability guard before release.

## Interpretation guardrails

- Failed reference jobs are retained with zero quality where the frozen benchmark recorded a controlled failure.
- Synthetic seeds change the generated datasets; deterministic real-data predictions are run once.
- Mosmann/Nilsson q values are tuned on their truth labels and must be validated out of sample before becoming automatic defaults.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--samusik-01", type=Path, required=True)
    parser.add_argument("--samusik-all", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.new_results)
    selected, selection = select_new(raw, args.output)
    reference = prepare_reference(args.reference)
    samusik_reference = pd.concat(
        [prepare_reference(args.samusik_01), prepare_reference(args.samusik_all)],
        ignore_index=True,
    )
    tables = build_tables(selected, reference, samusik_reference, args.output)
    paired = paired_comparisons(tables, args.output)
    plot_summary(raw, tables, args.output)
    write_report(selected, selection, tables, paired, args.output)
    print((args.output / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
