from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plot_rare_dense_noise_comparison.py"
SPEC = importlib.util.spec_from_file_location("plot_rare_dense_noise_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _matrix() -> pd.DataFrame:
    rows = []
    for method in MODULE.METHODS:
        for noise in MODULE.NOISE_LEVELS:
            for seed in MODULE.SEEDS:
                rows.append(
                    {
                        "job_id": f"{method}-{noise}-{seed}",
                        "status": "ok",
                        "dataset_id": f"global_contamination_noise{int(noise * 100)}__quality__signal300",
                        "method": method,
                        "seed": seed,
                        "n": int(round(300 / (1 - noise))),
                        "evaluation_protocol_version": "target-discovery-v1",
                        "matching_strategy": "hungarian",
                        "matching_objective": "pairwise_f1",
                        "discovery_purity_threshold": 0.9,
                        "discovery_coverage_threshold": 0.1,
                        "macro_target_f1": 0.9,
                        "target_discovery_rate": 1.0,
                        "noise_evidence_f1": 0.95,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_retains_failures_without_zero_filling_quality() -> None:
    frame = _matrix()
    mask = (
        (frame["method"] == "AMD-DBSCAN")
        & (frame["dataset_id"].str.contains("noise99"))
        & (frame["seed"] == 233)
    )
    frame.loc[mask, ["status", "macro_target_f1", "target_discovery_rate", "noise_evidence_f1"]] = [
        "error",
        float("nan"),
        float("nan"),
        float("nan"),
    ]
    summary = MODULE.validate_and_summarize(frame)
    row = summary.loc[
        (summary["method"] == "AMD-DBSCAN") & (summary["noise_fraction"] == 0.99)
    ].iloc[0]
    assert row["n_attempted"] == 5
    assert row["n_successful"] == 4
    assert row["macro_target_f1_median"] == pytest.approx(0.9)


def test_summary_fails_closed_on_incomplete_matrix() -> None:
    with pytest.raises(ValueError, match="expected 270"):
        MODULE.validate_and_summarize(_matrix().iloc[:-1])
