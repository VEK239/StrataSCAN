from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/make_baseline_parameter_profiles.py"
SPEC = importlib.util.spec_from_file_location("make_baseline_parameter_profiles", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
profiles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiles)


def test_value_summary_is_deterministic_and_preserves_exact_value_hash() -> None:
    first = profiles.summarize_values([3.0, 1.0, 3.0], total_records=4)
    second = profiles.summarize_values([1.0, 3.0, 3.0], total_records=4)
    assert first == second
    assert first["minimum"] == 1.0
    assert first["maximum"] == 3.0
    assert first["n_distinct"] == 2
    assert first["missing_records"] == 1


def test_build_profiles_splits_fixed_rules_from_realized_values(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.csv"
    frame = pd.DataFrame(
        [
            {
                "method": "DBSCAN",
                "status": "ok",
                "dimension": 2,
                "parameters_json": json.dumps(
                    {"geometry_profile": "low_dim", "min_samples": 5, "eps": 0.2}
                ),
                "profile_json": "{}",
            },
            {
                "method": "DBSCAN",
                "status": "ok",
                "dimension": 2,
                "parameters_json": json.dumps(
                    {"geometry_profile": "low_dim", "min_samples": 5, "eps": 0.4}
                ),
                "profile_json": "{}",
            },
            {
                "method": "DBSCAN",
                "status": "timeout",
                "dimension": None,
                "dataset_id": "overlap_8d__quality__n200000",
                "parameters_json": "{}",
                "profile_json": "{}",
            },
            {
                "method": "StrataSCAN",
                "status": "ok",
                "dimension": 16,
                "parameters_json": json.dumps(
                    {"geometry_profile": "high_dim", "k": 32}
                ),
                "profile_json": json.dumps({"mdl_gamma_selected_components": 7}),
            },
        ]
    )
    frame.to_csv(evidence, index=False)
    table = profiles.build_profiles(
        frame,
        source_path=evidence,
        resource_profiles={"profiles": [], "protocols": []},
        repo=Path(__file__).resolve().parents[1],
    )

    low = table.loc[
        table["method"].eq("DBSCAN") & table["dimension_regime"].eq("low_dim")
    ].iloc[0]
    fixed = json.loads(low["fixed_parameter_rules_json"])
    adaptive = json.loads(low["adaptive_or_realized_parameters_json"])
    assert fixed == {"geometry_profile": "low_dim", "min_samples": 5}
    assert adaptive["eps"]["minimum"] == 0.2
    assert adaptive["eps"]["maximum"] == 0.4
    assert adaptive["eps"]["n_distinct"] == 2

    failed = table.loc[
        table["method"].eq("DBSCAN") & table["dimension_regime"].eq("high_dim")
    ].iloc[0]
    assert failed["parameter_records_available"] == 0
    assert failed["evidence_rows"] == 1

    strata = table.loc[table["method"].eq("StrataSCAN")].iloc[0]
    realized = json.loads(strata["adaptive_or_realized_parameters_json"])
    assert realized["profile.mdl_gamma_selected_components"]["distinct_values"] == [7]


def test_write_profiles_emits_csv_markdown_and_source_hash(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.csv"
    protocol = tmp_path / "protocol.json"
    output_csv = tmp_path / "tables" / "profiles.csv"
    output_markdown = tmp_path / "tables" / "profiles.md"
    pd.DataFrame(
        [
            {
                "method": "HDBSCAN",
                "status": "ok",
                "dimension": 2,
                "parameters_json": json.dumps(
                    {"geometry_profile": "low_dim", "min_cluster_size": 40}
                ),
                "profile_json": "{}",
            }
        ]
    ).to_csv(evidence, index=False)
    protocol.write_text(
        json.dumps(
            {
                "protocol_version": "test-v1",
                "methods": ["HDBSCAN"],
                "resources": {"threads": 1, "timeout_seconds": 10, "memory_limit_mb": 64},
            }
        ),
        encoding="utf-8",
    )

    table = profiles.write_profiles(
        evidence,
        output_csv,
        output_markdown,
        protocol_paths=[protocol],
        repo=Path(__file__).resolve().parents[1],
    )
    assert len(table) == 1
    assert output_csv.exists()
    markdown = output_markdown.read_text(encoding="utf-8")
    assert "| HDBSCAN | low dim |" in markdown
    assert profiles.sha256_file(evidence) in markdown
    assert "timeout_seconds" in output_csv.read_text(encoding="utf-8")


def test_invalid_parameter_json_is_rejected(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.csv"
    frame = pd.DataFrame(
        [
            {
                "method": "DBSCAN",
                "status": "ok",
                "dimension": 2,
                "parameters_json": "{broken",
                "profile_json": "{}",
            }
        ]
    )
    frame.to_csv(evidence, index=False)
    with pytest.raises(ValueError, match="invalid parameters_json JSON"):
        profiles.build_profiles(
            frame,
            source_path=evidence,
            resource_profiles={"profiles": [], "protocols": []},
        )
