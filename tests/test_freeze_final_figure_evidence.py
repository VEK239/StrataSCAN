from __future__ import annotations

import importlib.util
from pathlib import Path
import json

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_final_figure_evidence.py"
SPEC = importlib.util.spec_from_file_location("freeze_final_figure_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
freeze_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze_module)


def evidence_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = [("case_a__quality__n20000", 211), ("case_b__quality__n20000", 223)]
    baseline_rows = []
    for method in sorted(freeze_module.BASELINE_METHODS):
        for dataset_id, seed in cells:
            baseline_rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "seed": seed,
                    "status": "ok",
                    "suite": "synthetic",
                    "package_version": "0.2.2",
                }
            )
    for dataset_id, seed in cells:
        baseline_rows.append(
            {
                "dataset_id": dataset_id,
                "method": "StrataSCAN",
                "seed": seed,
                "status": "ok",
                "suite": "synthetic",
                "package_version": "0.2.2",
            }
        )
    current = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "method": "StrataSCAN",
                "seed": seed,
                "status": "ok",
                "suite": "synthetic",
                "package_version": "0.2.3",
            }
            for dataset_id, seed in cells
        ]
    )
    return pd.DataFrame(baseline_rows), current


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    baseline, current = evidence_frames()
    baseline_path = tmp_path / "baseline" / "results.csv"
    current_path = tmp_path / "current" / "results.csv"
    output_path = tmp_path / "published" / "combined.csv"
    baseline_path.parent.mkdir()
    current_path.parent.mkdir()
    baseline.to_csv(baseline_path, index=False)
    current.to_csv(current_path, index=False)
    baseline_path.with_name("manifest.json").write_text("{}", encoding="utf-8")
    current_path.with_name("manifest.json").write_text("{}", encoding="utf-8")
    baseline_path.with_name("validation.json").write_text(
        json.dumps({"complete": True, "expected": 18, "found": 18}), encoding="utf-8"
    )
    current_path.with_name("validation.json").write_text(
        json.dumps({"complete": True, "expected": 2, "found": 2}), encoding="utf-8"
    )
    return baseline_path, current_path, output_path


def test_freeze_requires_matching_locked_cells(tmp_path: Path, monkeypatch) -> None:
    baseline_path, current_path, output_path = write_inputs(tmp_path)
    monkeypatch.setattr(freeze_module, "EXPECTED_CELLS_PER_METHOD", 2)
    current = pd.read_csv(current_path)
    current.loc[1, "seed"] = 227
    current.to_csv(current_path, index=False)
    with pytest.raises(ValueError, match="locked cells differ"):
        freeze_module.freeze(baseline_path, current_path, output_path)


def test_freeze_rejects_wrong_current_package_version(tmp_path: Path, monkeypatch) -> None:
    baseline_path, current_path, output_path = write_inputs(tmp_path)
    monkeypatch.setattr(freeze_module, "EXPECTED_CELLS_PER_METHOD", 2)
    current = pd.read_csv(current_path)
    current["package_version"] = "0.2.2"
    current.to_csv(current_path, index=False)
    with pytest.raises(ValueError, match="package version 0.2.3"):
        freeze_module.freeze(baseline_path, current_path, output_path)


def test_freeze_writes_validated_combined_evidence(tmp_path: Path, monkeypatch) -> None:
    baseline_path, current_path, output_path = write_inputs(tmp_path)
    monkeypatch.setattr(freeze_module, "EXPECTED_CELLS_PER_METHOD", 2)
    provenance = freeze_module.freeze(baseline_path, current_path, output_path)
    combined = pd.read_csv(output_path)
    assert len(combined) == 18
    assert set(combined["method"]) == {*freeze_module.BASELINE_METHODS, "StrataSCAN"}
    assert provenance["rows"] == 18
    assert output_path.with_name("combined-provenance.json").exists()
