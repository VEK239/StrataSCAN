from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "manuscript/oedm2026/scripts/discovery_threshold_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("discovery_threshold_sensitivity", SCRIPT)
assert SPEC and SPEC.loader
sensitivity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sensitivity
SPEC.loader.exec_module(sensitivity)


def encoded_matches() -> str:
    return json.dumps([
        {"target": "a", "matched_predicted_cluster": 2, "purity": .96, "coverage": .20},
        {"target": "b", "matched_predicted_cluster": 3, "precision": .92, "recall": .08},
        {"target": "c", "matched_predicted_cluster": None, "purity": 0., "coverage": 0.},
    ])


def test_parse_matches_supports_current_and_compatibility_names() -> None:
    assert sensitivity.parse_target_matches(encoded_matches()) == [(.96, .20), (.92, .08), (0., 0.)]


def test_unmatched_target_is_retained_as_zero_and_never_discovered() -> None:
    matches = sensitivity.parse_target_matches(encoded_matches())
    assert len(matches) == 3
    assert sensitivity.discovery_rate(matches, .90, .10) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="unmatched target"):
        sensitivity.parse_target_matches(json.dumps([
            {"matched_predicted_cluster": None, "purity": .9, "coverage": .2}
        ]))


def test_threshold_rates_are_monotone() -> None:
    frame = pd.DataFrame([{
        "method": "StrataSCAN", "status": "ok", "target_matches_json": encoded_matches(),
        "target_discovery_rate": 1 / 3,
    }])
    result = sensitivity.summarize_block(frame, "synthetic_fresh")
    rates = {(row.purity_threshold, row.coverage_threshold): row.mean_execution_discovery_rate for row in result.itertuples()}
    assert rates[(.90, .05)] >= rates[(.90, .10)] >= rates[(.90, .25)]
    assert rates[(.90, .10)] >= rates[(.95, .10)]


def test_locked_threshold_must_reproduce_recorded_rate() -> None:
    frame = pd.DataFrame([{
        "method": "StrataSCAN", "status": "ok", "target_matches_json": encoded_matches(),
        "target_discovery_rate": .9,
    }])
    with pytest.raises(ValueError, match="locked discovery mismatch"):
        sensitivity.summarize_block(frame, "gaia")
