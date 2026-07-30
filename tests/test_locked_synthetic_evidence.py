from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analyze_locked_synthetic_evidence import (
    condition_summary,
    load_and_validate,
    reversal_summary,
    seed_level_deltas,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "protocol.v0.2.2-synthetic-evidence.json"


def test_locked_protocol_expands_to_complete_seed_matrix() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert len(protocol["methods"]) == 13
    assert len(protocol["synthetic"]["cases"]) == 12
    assert len(protocol["synthetic"]["quality_seeds"]) == 5


def test_evidence_analysis_retains_seed_deltas_and_failures(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = []
    for case in protocol["synthetic"]["cases"]:
        for method in protocol["methods"]:
            for seed in protocol["synthetic"]["quality_seeds"]:
                score = 0.7 if method == "StrataSCAN" else 0.6
                rows.append({
                    "suite": "synthetic",
                    "dataset_id": f"{case['id']}__quality__n5000",
                    "method": method,
                    "seed": seed,
                    "status": "ok",
                    "macro_target_f1": score,
                    "noise_f1": score,
                    "pairwise_f1": score,
                    "target_background_structure_hmean_f1": score,
                    "target_success_rate_f1_080": score,
                    "mean_target_fragments": 1.0,
                    "spurious_background_majority_clusters": 0,
                    "merged_predicted_clusters": 0,
                })
    # A controlled failure must remain in the input and be zeroed only for
    # quality metrics.
    rows[0]["status"] = "timeout"
    rows[0]["pairwise_f1"] = 0.9
    pd.DataFrame(rows).to_csv(tmp_path / "results.csv", index=False)
    (tmp_path / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )

    frame, _ = load_and_validate(tmp_path, PROTOCOL)
    assert len(frame) == 780
    assert frame.loc[frame["status"].eq("timeout"), "pairwise_f1"].item() == 0.0
    assert pd.isna(frame.loc[frame["status"].eq("timeout"), "mean_target_fragments"].item())
    summary = condition_summary(frame)
    deltas = seed_level_deltas(frame)
    reversals = reversal_summary(deltas)
    assert len(summary) == 156
    assert len(deltas) == 720
    assert not reversals["reversal_observed"].any()
