from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rq2_density_noise.py"
SPEC = importlib.util.spec_from_file_location("prepare_rq2_density_noise", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


ROOT = Path(__file__).resolve().parents[1]
DENSITY = ROOT / "results" / "runs" / "v0.2.4-density-contrast-stratascan-full" / "results.csv"
NOISE = ROOT / "results" / "runs" / "v0.2.4-rare-dense-noise-all-methods-lean" / "results.csv"


def test_canonical_rq2_matrices_validate_and_reproduce_claims() -> None:
    density, density_audit = MODULE.summarize_density(pd.read_csv(DENSITY))
    noise, strata, endpoint, noise_audit = MODULE.summarize_noise(pd.read_csv(NOISE))

    assert density.shape[0] == 9
    assert density_audit["successful"] == 27
    assert density_audit["minimum_cell_target_discovery_rate"] == pytest.approx(1.0)
    assert density_audit["minimum_case_median_target_f1"] == pytest.approx(0.8288, abs=5e-4)
    assert density_audit["maximum_case_median_target_f1"] == pytest.approx(0.9290, abs=5e-4)

    assert noise.shape[0] == 54
    assert noise_audit["successful"] == 268
    assert noise_audit["stratascan_successful"] == 30
    assert noise_audit["stratascan_minimum_cell_discovery_rate"] == pytest.approx(1.0)
    assert noise_audit["stratascan_99_target_f1"] == pytest.approx(0.9794670445)
    assert noise_audit["stratascan_99_noise_f1"] == pytest.approx(0.9995624074)
    strata_99 = endpoint.loc[endpoint["method"] == "StrataSCAN"].iloc[0]
    assert strata_99["noise_evidence_f1_median"] == endpoint["noise_evidence_f1_median"].max()


def test_density_validation_fails_closed_on_missing_cell() -> None:
    frame = pd.read_csv(DENSITY).iloc[:-1]
    with pytest.raises(ValueError, match="expected 27"):
        MODULE.summarize_density(frame)


def test_noise_validation_retains_controlled_failures() -> None:
    frame = pd.read_csv(NOISE)
    _, _, endpoint, audit = MODULE.summarize_noise(frame)
    amd = endpoint.loc[endpoint["method"] == "AMD-DBSCAN"].iloc[0]
    assert audit["attempted"] == 270
    assert audit["successful"] == 268
    assert amd["n_attempted"] == 5
    assert amd["n_successful"] == 3
