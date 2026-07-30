from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_gaia
from benchmarks.evaluation import evaluate
from stratascan.experimental import MultiscaleStrataSCAN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--protected-results", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.results)
    baseline = raw.loc[
        raw["suite"].eq("gaia")
        & raw["status"].eq("ok")
        & raw["method"].eq("StrataSCAN")
        & raw["seed"].eq(42)
    ].copy()
    baseline = baseline.loc[baseline["best_cluster_f1"] < 0.5].sort_values("dataset_id")
    if len(baseline) != 28:
        raise ValueError(f"expected 28 bad fields, found {len(baseline)}")
    protected = None
    if args.protected_results is not None:
        protected = pd.read_csv(args.protected_results).set_index("dataset_id")
    data_root = REPO / "data/raw/mctnc/MCTNC_open_data_release/MCTNC_open_data_release/data"
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "gaia-multiscale-bad-fields.csv"
    if checkpoint.exists():
        output = pd.read_csv(checkpoint)
        rows = output.to_dict("records")
        completed = set(output["dataset_id"].astype(str))
    else:
        rows = []
        completed = set()
    pending = baseline.loc[~baseline["dataset_id"].isin(completed)]
    for index, row in enumerate(pending.itertuples(index=False), start=1):
        metadata = json.loads(row.dataset_metadata_json)
        field_id = metadata["field_id"]
        dataset = load_gaia(
            {
                "data_root": str(data_root.relative_to(REPO)),
                "field_id": field_id,
                "ruwe_max": 1.6,
                "preprocessing": "unsupervised_field",
            },
            REPO,
        )
        model = MultiscaleStrataSCAN(
            backend="faiss_hnsw",
            n_jobs=1,
            ambient_dimension=5.0,
        ).fit(dataset.X)
        metrics = evaluate(dataset, model.labels_, "gaia")
        protected_f1 = np.nan
        if protected is not None:
            protected_f1 = float(protected.loc[row.dataset_id, "variant_f1"])
        rows.append(
            {
                "dataset_id": row.dataset_id,
                "field_id": field_id,
                "n": len(dataset.y),
                "reference_members": int(np.sum(dataset.y >= 0)),
                "baseline_f1": float(row.best_cluster_f1),
                "baseline_precision": float(row.best_cluster_precision),
                "baseline_recall": float(row.best_cluster_recall),
                "multiscale_f1": metrics["best_cluster_f1"],
                "multiscale_precision": metrics["best_cluster_precision"],
                "multiscale_recall": metrics["best_cluster_recall"],
                "multiscale_clusters": metrics["n_clusters"],
                "multiscale_noise_fraction": metrics["noise_fraction"],
                "protected_contrast_f1": protected_f1,
                "profile_json": json.dumps(model.profile_),
            }
        )
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        print(f"processed {index}/{len(pending)} remaining bad fields: {field_id}", flush=True)
    output = pd.DataFrame(rows)
    summary = {
        "fields": int(len(output)),
        "baseline_mean_f1": float(output["baseline_f1"].mean()),
        "multiscale_mean_f1": float(output["multiscale_f1"].mean()),
        "protected_contrast_mean_f1": float(output["protected_contrast_f1"].mean()),
        "multiscale_improved": int((output["multiscale_f1"] > output["baseline_f1"] + 1e-12).sum()),
        "multiscale_unchanged": int((np.abs(output["multiscale_f1"] - output["baseline_f1"]) <= 1e-12).sum()),
        "multiscale_worsened": int((output["multiscale_f1"] < output["baseline_f1"] - 1e-12).sum()),
        "multiscale_f1_ge_080": int((output["multiscale_f1"] >= 0.8).sum()),
        "multiscale_f1_ge_090": int((output["multiscale_f1"] >= 0.9).sum()),
        "multiscale_f1_lt_050": int((output["multiscale_f1"] < 0.5).sum()),
    }
    (args.output / "gaia-multiscale-bad-fields-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
