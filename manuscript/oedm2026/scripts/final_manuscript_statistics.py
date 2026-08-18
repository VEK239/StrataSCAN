from __future__ import annotations

"""Validate target-discovery-v1 evidence and build the manuscript source of truth.

No unrescored legacy table is accepted. Every successful row must identify the
Hungarian evaluator, locked discovery thresholds, and evaluation generation.
Gaia alone permits exact single-target-rescored legacy external predictions;
all other canonical rows must be native target-discovery-v1. Synthetic
noise evidence is suite-specific; biological and Gaia reference negatives are
reported only through explicitly diagnostic background-abstention fields.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
DEFAULT_JSON = TABLES / "final_statistical_summary.json"
DEFAULT_REPORT = TABLES / "final_statistical_summary.md"
EVIDENCE_ROOT = REPO / "results/published/v0.2.4/target-discovery-v1"

PROTOCOL_VERSION = "target-discovery-v1"
MATCHING_STRATEGY = "hungarian"
MATCHING_OBJECTIVE = "pairwise_f1"
PURITY_THRESHOLD = 0.90
COVERAGE_THRESHOLD = 0.10
TERMINAL_STATUSES = {"ok", "error", "timeout", "memory_limit"}
STRATASCAN = "StrataSCAN"
METHOD_DISPLAY = {"AMD-DBSCAN": "AMD-inspired"}

TARGET_METRICS = (
    "macro_target_f1",
    "macro_target_purity",
    "macro_target_coverage",
    "target_discovery_rate",
)
SYNTHETIC_DIAGNOSTICS = ("noise_evidence_f1", "pairwise_f1")
ABSTENTION_DIAGNOSTICS = (
    "background_abstention_precision_diagnostic",
    "background_abstention_fraction",
    "background_abstention_f1_diagnostic",
    "predicted_abstention_fraction",
)
BURDEN_METRICS = (
    "predicted_cluster_count",
    "unmatched_predicted_cluster_count",
    "background_majority_predicted_clusters",
    "candidate_clusters_per_target",
)


@dataclass(frozen=True)
class EvidencePaths:
    synthetic_release: Path
    synthetic_scaling: Path
    cytometry: Path
    gaia: Path
    noise_factorial: Path

    @classmethod
    def canonical(cls, root: Path = EVIDENCE_ROOT) -> "EvidencePaths":
        return cls(
            synthetic_release=root / "synthetic_release.csv",
            synthetic_scaling=root / "synthetic_scaling.csv",
            cytometry=root / "cytometry.csv",
            gaia=root / "gaia.csv",
            noise_factorial=root / "noise_factorial.csv",
        )


@dataclass(frozen=True)
class EvidenceDesign:
    methods: tuple[str, ...]
    cases: tuple[str, ...]
    development_sizes: tuple[int, ...]
    development_seeds: tuple[int, ...]
    fresh_seeds: tuple[int, ...]
    comparison_size: int
    scaling_sizes: tuple[int, ...]
    biological_datasets: tuple[str, ...]
    noise_methods: tuple[str, ...]
    noise_cases: tuple[str, ...]
    noise_seeds: tuple[int, ...]

    @classmethod
    def canonical(cls) -> "EvidenceDesign":
        return cls(
            methods=(
                "DBSCAN", "HDBSCAN", "OPTICS", "SNN-DBSCAN", "VDBSCAN-2007",
                "AMD-DBSCAN", "kNN-DBSCAN", "kNN+Leiden", STRATASCAN,
            ),
            cases=(
                "multidensity_2d", "ultrasparse_16d", "overlapping_density_16d",
                "moons_2d", "rings_2d", "overlap_8d", "imbalanced_16d",
            ),
            development_sizes=(20_000,),
            development_seeds=(23, 42, 73, 101, 151),
            fresh_seeds=(211, 223, 227),
            comparison_size=20_000,
            scaling_sizes=(500_000, 1_000_000, 2_000_000, 5_000_000),
            biological_datasets=(
                "levine", "mosmann", "nilsson",
                *(f"samusik_{index:02d}" for index in range(1, 11)),
            ),
            noise_methods=(
                "DBSCAN", "HDBSCAN", "OPTICS", "SNN-DBSCAN", "VDBSCAN-2007",
                "AMD-DBSCAN", "kNN-DBSCAN", "kNN+Leiden", STRATASCAN,
            ),
            noise_cases=(
                "moons_noise50", "moons_noise75", "moons_noise90",
                "moons_noise95", "moons_noise975", "moons_noise99",
            ),
            noise_seeds=(211, 223, 227, 229, 233),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"required target-discovery-v1 {label} freeze is missing: {path}"
        )
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"required target-discovery-v1 {label} freeze is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _provenance_path(path: Path) -> Path:
    return path.with_name(path.stem + "-provenance.json")


def _check_provenance(path: Path) -> dict[str, object]:
    provenance_path = _provenance_path(path)
    if not provenance_path.is_file():
        raise FileNotFoundError(f"required evidence provenance is missing: {provenance_path}")
    value = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evidence provenance is not an object: {provenance_path}")
    expected = {
        "sha256": _sha256(path),
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "matching_strategy": MATCHING_STRATEGY,
        "matching_objective": MATCHING_OBJECTIVE,
        "discovery_purity_threshold": PURITY_THRESHOLD,
        "discovery_coverage_threshold": COVERAGE_THRESHOLD,
    }
    for key, expected_value in expected.items():
        observed = value.get(key)
        if isinstance(expected_value, float):
            valid = isinstance(observed, (int, float)) and np.isclose(observed, expected_value)
        else:
            valid = observed == expected_value
        if not valid:
            raise ValueError(
                f"evidence provenance has wrong {key}: expected {expected_value!r}, got {observed!r}"
            )
    return value


def _check_semantics(frame: pd.DataFrame, suite: str, label: str) -> None:
    common = [
        "suite", "dataset_id", "method", "seed", "status", "package_version",
        "evaluation_generation",
        "evaluation_protocol_version", "evaluation_mode", "background_semantics",
        "matching_strategy", "matching_objective", "primary_target_aggregation",
        "discovery_purity_threshold", "discovery_coverage_threshold",
        "runtime_seconds", "process_peak_rss_mb", *TARGET_METRICS,
    ]
    diagnostics = SYNTHETIC_DIAGNOSTICS if suite == "synthetic" else ABSTENTION_DIAGNOSTICS
    _require_columns(frame, [*common, *diagnostics, *BURDEN_METRICS], label)
    if set(frame["suite"].astype(str)) != {suite}:
        raise ValueError(f"{label} contains a suite other than {suite}")
    unknown = set(frame["status"].astype(str)) - TERMINAL_STATUSES
    if unknown:
        raise ValueError(f"{label} contains nonterminal statuses: {sorted(unknown)}")
    successful = frame.loc[frame["status"].eq("ok")]
    if successful.empty:
        raise ValueError(f"{label} contains no successful rows")
    release_rows = successful["method"].astype(str).str.startswith(STRATASCAN)
    observed_release_versions = set(successful.loc[release_rows, "package_version"].astype(str))
    if observed_release_versions != {"0.2.4"}:
        raise ValueError(
            f"{label} has wrong StrataSCAN release version: {sorted(observed_release_versions)}"
        )
    if suite == "gaia":
        native = successful["method"].astype(str).eq(STRATASCAN)
        if set(successful.loc[native, "evaluation_generation"].astype(str)) != {
            "native_target_discovery_v1"
        }:
            raise ValueError("gaia StrataSCAN rows are not native target-discovery-v1")
        if set(successful.loc[~native, "evaluation_generation"].astype(str)) != {
            "legacy_single_target_exact_rescore"
        }:
            raise ValueError("gaia external rows are not exact single-target legacy rescores")
    elif set(successful["evaluation_generation"].astype(str)) != {
        "native_target_discovery_v1"
    }:
        raise ValueError(f"{label} contains a non-native evaluation generation")
    exact = {
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "matching_strategy": MATCHING_STRATEGY,
        "matching_objective": MATCHING_OBJECTIVE,
        "primary_target_aggregation": "macro",
    }
    for column, expected in exact.items():
        observed = set(successful[column].dropna().astype(str))
        if observed != {expected}:
            raise ValueError(f"{label} has wrong {column}: {sorted(observed)}")
    for column, expected in (
        ("discovery_purity_threshold", PURITY_THRESHOLD),
        ("discovery_coverage_threshold", COVERAGE_THRESHOLD),
    ):
        values = pd.to_numeric(successful[column], errors="coerce")
        if values.isna().any() or not np.allclose(values, expected):
            raise ValueError(f"{label} has wrong {column}; expected {expected}")
    expected_background = {
        "synthetic": "known_synthetic_noise",
        "cytometry": "heterogeneous_biological_background",
        "gaia": "reference_negative_background",
    }[suite]
    if set(successful["background_semantics"].astype(str)) != {expected_background}:
        raise ValueError(f"{label} has wrong background semantics")
    if suite in {"synthetic", "gaia"}:
        expected_mode = "full_partition" if suite == "synthetic" else "target_discovery"
        if set(successful["evaluation_mode"].astype(str)) != {expected_mode}:
            raise ValueError(f"{label} has wrong evaluation mode")
    else:
        expected_modes = successful["dataset_id"].astype(str).map(
            lambda dataset: "target_discovery" if dataset in {"mosmann", "nilsson"} else "full_partition"
        )
        if not successful["evaluation_mode"].astype(str).reset_index(drop=True).equals(
            expected_modes.reset_index(drop=True)
        ):
            raise ValueError("cytometry evaluation modes do not match the locked dataset overrides")
    metric_columns = [*TARGET_METRICS, *diagnostics]
    for column in metric_columns:
        values = pd.to_numeric(successful[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"successful {label} rows have invalid {column}")
    forbidden = {"noise_f1", "noise_precision", "noise_recall"}
    if suite != "synthetic" and forbidden & set(frame.columns):
        raise ValueError(f"{label} exposes synthetic-noise names for non-synthetic references")


def _case(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"__(?:quality|scaling)__n\d+$", "", regex=True)


def _ensure_n(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    parsed = pd.to_numeric(
        result["dataset_id"].astype(str).str.extract(r"__n(\d+)$", expand=False),
        errors="coerce",
    )
    current = pd.to_numeric(result.get("n", np.nan), errors="coerce")
    if not isinstance(current, pd.Series):
        current = pd.Series(current, index=result.index)
    result["n"] = current.fillna(parsed)
    return result


def _identities(cases: Sequence[str], sizes: Sequence[int], seeds: Sequence[int], methods: Sequence[str]) -> set[tuple[str, str, int]]:
    return {
        (f"{case}__quality__n{size}", method, seed)
        for case in cases for size in sizes for seed in seeds for method in methods
    }


def _check_exact(frame: pd.DataFrame, expected: set[tuple[str, str, int]], label: str) -> None:
    keys = frame[["dataset_id", "method", "seed"]].copy()
    keys["seed"] = pd.to_numeric(keys["seed"], errors="raise").astype(int)
    if keys.duplicated().any():
        raise ValueError(f"{label} contains duplicate identities")
    observed = set(map(tuple, keys.itertuples(index=False, name=None)))
    if observed != expected:
        raise ValueError(
            f"{label} identity mismatch: missing={sorted(expected-observed)[:3]}, "
            f"extra={sorted(observed-expected)[:3]}"
        )


def _validate_sources(paths: EvidencePaths, design: EvidenceDesign) -> dict[str, pd.DataFrame]:
    frames = {
        field: _read(getattr(paths, field), field.replace("_", " "))
        for field in EvidencePaths.__dataclass_fields__
    }
    for field, frame in frames.items():
        _check_provenance(getattr(paths, field))
        suite = "synthetic" if field.startswith("synthetic") or field == "noise_factorial" else field
        _check_semantics(frame, suite, field.replace("_", " "))

    release = _ensure_n(frames["synthetic_release"])
    _require_columns(release, ["evidence_stage"], "synthetic release")
    development = release.loc[release["evidence_stage"].eq("development")]
    fresh = release.loc[release["evidence_stage"].eq("fresh")]
    if len(development) + len(fresh) != len(release):
        raise ValueError("synthetic release contains an unknown evidence_stage")
    _check_exact(
        development,
        _identities(design.cases, design.development_sizes, design.development_seeds, design.methods),
        "synthetic development",
    )
    _check_exact(
        fresh,
        _identities(design.cases, (design.comparison_size,), design.fresh_seeds, design.methods),
        "synthetic fresh",
    )
    frames["synthetic_release"] = release

    scaling_expected = {
        (f"{case}__scaling__n{size}", STRATASCAN, 42)
        for case in design.cases for size in design.scaling_sizes
    }
    _check_exact(frames["synthetic_scaling"], scaling_expected, "synthetic scaling")
    noise_expected = _identities(
        design.noise_cases, (20_000,), design.noise_seeds, design.noise_methods
    )
    _check_exact(frames["noise_factorial"], noise_expected, "noise factorial")

    for field, datasets in (("cytometry", design.biological_datasets), ("gaia", None)):
        frame = frames[field]
        keys = frame[["dataset_id", "method", "seed"]]
        if keys.duplicated().any():
            raise ValueError(f"{field} contains duplicate execution identities")
        observed_methods = set(frame["method"].astype(str))
        if observed_methods != set(design.methods):
            raise ValueError(f"{field} method panel is incomplete")
        observed_datasets = set(frame["dataset_id"].astype(str))
        if datasets is not None and observed_datasets != set(datasets):
            raise ValueError("cytometry dataset panel is incomplete")
        if field == "gaia" and len(observed_datasets) != 359:
            raise ValueError("Gaia freeze must contain exactly 359 fields")
        matrix = set(map(tuple, frame[["dataset_id", "method"]].drop_duplicates().itertuples(index=False, name=None)))
        expected_matrix = {(dataset, method) for dataset in observed_datasets for method in design.methods}
        if matrix != expected_matrix:
            raise ValueError(f"{field} dataset-method matrix is incomplete")
    return frames


def _distribution(values: Sequence[float] | pd.Series) -> dict[str, float | int | None]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not len(array):
        return {key: None if key != "n" else 0 for key in ("n", "min", "q1", "median", "q3", "max", "mean")}
    return {
        "n": int(len(array)), "min": float(array.min()), "q1": float(np.quantile(array, .25)),
        "median": float(np.median(array)), "q3": float(np.quantile(array, .75)),
        "max": float(array.max()), "mean": float(array.mean()),
    }


def _effect(values: pd.Series) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    tolerance = 1e-12
    return _distribution(clean) | {
        "sign_counts": {
            "positive": int((clean > tolerance).sum()),
            "tie": int((clean.abs() <= tolerance).sum()),
            "negative": int((clean < -tolerance).sum()),
            "pairs": int(len(clean)),
        }
    }


def _contrast_block(frame: pd.DataFrame, design: EvidenceDesign, stage: str) -> dict[str, object]:
    selected = frame.loc[
        frame["evidence_stage"].eq(stage) & frame["n"].eq(design.comparison_size)
    ].copy()
    selected["case"] = _case(selected["dataset_id"])
    table = selected.pivot(index=["case", "seed"], columns="method", values="macro_target_f1")
    status = selected.assign(success=selected["status"].eq("ok")).pivot(
        index=["case", "seed"], columns="method", values="success"
    )
    comparisons: dict[str, object] = {}
    for baseline in (method for method in design.methods if method != STRATASCAN):
        joint = status[STRATASCAN] & status[baseline]
        delta = (table[STRATASCAN] - table[baseline]).where(joint)
        families = {
            case: {
                "delta_joint_success": _effect(delta.xs(case, level="case")),
                "joint_successful_pairs": int(joint.xs(case, level="case").sum()),
                "declared_pairs": int(len(joint.xs(case, level="case"))),
            }
            for case in design.cases
        }
        comparisons[baseline] = {
            "overall_delta_joint_success": _effect(delta),
            "joint_successful_pairs": int(joint.sum()),
            "declared_pairs": int(len(joint)),
            "families": families,
        }
    strata = selected.loc[selected["method"].eq(STRATASCAN)]
    anchors = {
        case: {
            metric: _distribution(part.loc[part["status"].eq("ok"), metric])
            for metric in TARGET_METRICS
        }
        for case, part in strata.groupby("case", observed=True)
    }
    return {
        "stage": stage,
        "n": design.comparison_size,
        "cells": int(len(selected)),
        "successful_cells": int(selected["status"].eq("ok").sum()),
        "failure_policy": "quality remains missing on failed executions; effects use joint successes",
        "comparisons": comparisons,
        "stratascan_anchors": anchors,
    }


def _resource_summary(frame: pd.DataFrame) -> dict[str, object] | None:
    successful = frame.loc[frame["status"].eq("ok")]
    if successful.empty:
        return None
    return {
        "successful_cells": int(len(successful)),
        "runtime_seconds": _distribution(successful["runtime_seconds"]),
        "process_peak_rss_mb": _distribution(successful["process_peak_rss_mb"]),
    }


def _scaling(frame: pd.DataFrame, design: EvidenceDesign) -> dict[str, object]:
    prepared = _ensure_n(frame)
    successful = prepared.loc[prepared["status"].eq("ok")]
    by_size = {
        str(size): {
            "successful_cells": int(part["status"].eq("ok").sum()),
            "prespecified_cells": int(len(part)),
            "resources_successful_only": _resource_summary(part),
            "quality_successful_only": {
                metric: _distribution(part.loc[part["status"].eq("ok"), metric])
                for metric in (*TARGET_METRICS, *SYNTHETIC_DIAGNOSTICS)
            },
        }
        for size in design.scaling_sizes
        for part in [prepared.loc[prepared["n"].eq(size)]]
    }
    endpoint = prepared.loc[prepared["n"].eq(max(design.scaling_sizes))]
    return {
        "successful_cells": int(len(successful)), "prespecified_cells": int(len(prepared)),
        "endpoint_size": max(design.scaling_sizes),
        "endpoint_resources_successful_only": _resource_summary(endpoint), "by_size": by_size,
        "claim_boundary": "StrataSCAN-only execution envelope",
    }


def _study(dataset: str) -> str:
    return "Samusik" if dataset.startswith("samusik_") else dataset.title()


def _collapse(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, method), part in frame.groupby(["dataset_id", "method"], observed=True):
        ok = part.loc[part["status"].eq("ok")]
        row: dict[str, object] = {
            "dataset_id": str(dataset), "method": str(method), "attempts": int(len(part)),
            "successful_attempts": int(len(ok)), "completion_fraction": float(len(ok) / len(part)),
        }
        for metric in metrics:
            row[metric] = float(pd.to_numeric(ok[metric], errors="coerce").median()) if len(ok) else np.nan
        for resource in ("runtime_seconds", "process_peak_rss_mb"):
            row[resource] = float(pd.to_numeric(ok[resource], errors="coerce").median()) if len(ok) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_payload(table: pd.DataFrame, methods: Sequence[str], metrics: Sequence[str]) -> dict[str, object]:
    return {
        method: {
            metric: (None if pd.isna(table.loc[method, metric]) else float(table.loc[method, metric]))
            for metric in metrics
        }
        for method in methods
    }


def _biological(frame: pd.DataFrame, design: EvidenceDesign) -> dict[str, object]:
    metrics = (*TARGET_METRICS, *ABSTENTION_DIAGNOSTICS, *BURDEN_METRICS)
    collapsed = _collapse(frame, metrics)
    collapsed["study"] = collapsed["dataset_id"].map(_study)
    dataset = collapsed.groupby("method", observed=True)[list(metrics)].mean().reindex(design.methods)
    study = (
        collapsed.groupby(["method", "study"], observed=True)[list(metrics)].mean()
        .groupby("method", observed=True).mean().reindex(design.methods)
    )
    ranking = [str(value) for value in study["macro_target_f1"].sort_values(ascending=False).index]
    effects: dict[str, object] = {}
    pivot = collapsed.pivot(index="dataset_id", columns="method", values="macro_target_f1")
    for dataset_id in design.biological_datasets:
        baseline = pivot.loc[dataset_id].drop(STRATASCAN).dropna()
        effects[dataset_id] = {
            "stratascan": None if pd.isna(pivot.loc[dataset_id, STRATASCAN]) else float(pivot.loc[dataset_id, STRATASCAN]),
            "delta_vs_each_baseline": {
                str(method): float(pivot.loc[dataset_id, STRATASCAN] - value)
                for method, value in baseline.items() if pd.notna(pivot.loc[dataset_id, STRATASCAN])
            },
        }
    coverage = {
        method: {
            "successful_datasets": int(part["successful_attempts"].gt(0).sum()),
            "declared_datasets": len(design.biological_datasets),
            "successful_attempts": int(part["successful_attempts"].sum()),
            "attempts": int(part["attempts"].sum()),
        }
        for method, part in collapsed.groupby("method", observed=True)
    }
    return {
        "primary_metric": "macro_target_f1_hungarian_one_to_one",
        "discovery_rule": {"purity": PURITY_THRESHOLD, "coverage": COVERAGE_THRESHOLD},
        "dataset_weighted": _metric_payload(dataset, design.methods, metrics),
        "equal_study_weighted": _metric_payload(study, design.methods, metrics),
        "equal_study_ranking_by_target_f1": ranking,
        "per_dataset_target_f1_effects": effects,
        "coverage": coverage,
        "background_interpretation": "reference-background abstention agreement is diagnostic, not known noise",
    }


def _gaia(frame: pd.DataFrame, design: EvidenceDesign) -> dict[str, object]:
    metrics = (*TARGET_METRICS, *ABSTENTION_DIAGNOSTICS, *BURDEN_METRICS)
    collapsed = _collapse(frame, metrics)
    methods: dict[str, object] = {}
    for method in design.methods:
        part = collapsed.loc[collapsed["method"].eq(method)]
        successful = part.loc[part["successful_attempts"].gt(0)]
        methods[method] = {
            "fields": int(len(part)),
            "successful_fields": int(len(successful)),
            "discovered_fields": int(pd.to_numeric(successful["target_discovery_rate"], errors="coerce").sum()),
            "target_metrics_successful_only": {
                metric: _distribution(successful[metric]) for metric in TARGET_METRICS
            },
            "abstention_diagnostics_successful_only": {
                metric: _distribution(successful[metric]) for metric in ABSTENTION_DIAGNOSTICS
            },
            "candidate_burden_successful_only": {
                metric: _distribution(successful[metric]) for metric in BURDEN_METRICS
            },
            "runtime_seconds_successful_only": _distribution(successful["runtime_seconds"]),
        }
    return {
        "fields": 359, "methods": methods,
        "discovery_rule": {"purity": PURITY_THRESHOLD, "coverage": COVERAGE_THRESHOLD},
        "background_interpretation": "reference-negative abstention agreement is diagnostic, not known noise",
    }


def _noise(frame: pd.DataFrame, design: EvidenceDesign) -> dict[str, object]:
    prepared = frame.copy()
    prepared["case"] = _case(prepared["dataset_id"])
    encoded_fraction = pd.to_numeric(
        prepared["case"].str.extract(r"_noise(\d+)$", expand=False), errors="raise"
    )
    prepared["declared_noise_fraction"] = encoded_fraction.where(
        encoded_fraction.le(99), encoded_fraction / 10
    ) / 100
    trajectories: dict[str, object] = {}
    for method in design.noise_methods:
        trajectories[method] = {}
        for fraction, part in prepared.loc[prepared["method"].eq(method)].groupby("declared_noise_fraction", observed=True):
            ok = part.loc[part["status"].eq("ok")]
            trajectories[method][str(float(fraction))] = {
                metric: _distribution(ok[metric])
                for metric in (*TARGET_METRICS, *SYNTHETIC_DIAGNOSTICS)
            } | {"successful_cells": int(len(ok)), "prespecified_cells": int(len(part))}
    return {"cells": int(len(prepared)), "trajectories": trajectories}


def _source(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    provenance = _provenance_path(path)
    return {
        "path": str(path.resolve()), "sha256": _sha256(path), "rows": int(len(frame)),
        "provenance_path": str(provenance.resolve()), "provenance_sha256": _sha256(provenance),
    }


def generate_statistics(paths: EvidencePaths, design: EvidenceDesign | None = None) -> dict[str, object]:
    design = design or EvidenceDesign.canonical()
    frames = _validate_sources(paths, design)
    release = frames["synthetic_release"]
    summary = {
        "schema_version": 2,
        "evaluation_contract": {
            "protocol_version": PROTOCOL_VERSION,
            "matching": "Hungarian one-to-one maximum pairwise F1",
            "primary_target_aggregation": "macro over reference targets",
            "discovery_rule": {"purity": PURITY_THRESHOLD, "coverage": COVERAGE_THRESHOLD},
            "synthetic_background": "known true-noise evidence",
            "biological_gaia_background": "reference-background abstention diagnostic only",
            "failure_policy": "failed quality remains missing; completion is always reported",
        },
        "sources": {field: _source(getattr(paths, field), frame) for field, frame in frames.items()},
        "synthetic": {
            "development": _contrast_block(release, design, "development"),
            "fresh": _contrast_block(release, design, "fresh"),
        },
        "scaling": _scaling(frames["synthetic_scaling"], design),
        "biological": _biological(frames["cytometry"], design),
        "gaia": _gaia(frames["gaia"], design),
        "noise_factorial": _noise(frames["noise_factorial"], design),
    }
    json.dumps(summary, allow_nan=False)
    return summary


def _fmt(value: object, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _display_method(method: object) -> str:
    value = str(method)
    return METHOD_DISPLAY.get(value, value)


def render_markdown(summary: Mapping[str, object]) -> str:
    contract = summary["evaluation_contract"]  # type: ignore[index]
    biological = summary["biological"]  # type: ignore[index]
    gaia = summary["gaia"]  # type: ignore[index]
    lines = [
        "# Final target-discovery-v1 statistical summary", "",
        "This generated report is accepted only after exact identity and evaluator-semantic checks.", "",
        "## Locked evaluation contract", "",
        f"- {contract['matching']}; unmatched targets score zero.",
        "- Primary target score: equal-target macro F1.",
        "- Discovery: purity >= 0.90 and coverage >= 0.10.",
        "- Synthetic true-noise evidence is separate from biological/Gaia abstention diagnostics.", "",
        "## Biological validation", "",
        "| Method | Equal-study target F1 | Purity | Coverage | Discovery rate | Abstention F1 diagnostic | Successful datasets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in biological["equal_study_weighted"].items():  # type: ignore[index,union-attr]
        coverage = biological["coverage"][method]  # type: ignore[index]
        lines.append(
            f"| {_display_method(method)} | {_fmt(values['macro_target_f1'])} | {_fmt(values['macro_target_purity'])} | "
            f"{_fmt(values['macro_target_coverage'])} | {_fmt(values['target_discovery_rate'])} | "
            f"{_fmt(values['background_abstention_f1_diagnostic'])} | "
            f"{coverage['successful_datasets']}/{coverage['declared_datasets']} |"
        )
    lines.extend(["", "## Gaia validation", ""])
    for method, values in gaia["methods"].items():  # type: ignore[index,union-attr]
        target = values["target_metrics_successful_only"]
        burden = values["candidate_burden_successful_only"]
        lines.append(
            f"- {_display_method(method)}: {values['discovered_fields']}/{values['fields']} fields discovered; "
            f"median target F1/purity/coverage {_fmt(target['macro_target_f1']['median'])}/"
            f"{_fmt(target['macro_target_purity']['median'])}/{_fmt(target['macro_target_coverage']['median'])}; "
            f"median unmatched candidates {_fmt(burden['unmatched_predicted_cluster_count']['median'])}; "
            f"successful fields {values['successful_fields']}/{values['fields']}."
        )
    lines.extend(["", "All numerical prose and figure data must be generated from the companion JSON.", ""])
    return "\n".join(lines)


def write_outputs(summary: Mapping[str, object], json_path: Path = DEFAULT_JSON, report_path: Path = DEFAULT_REPORT) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-discovery-v1 manuscript statistics")
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    paths = EvidencePaths.canonical(args.evidence_root)
    summary = generate_statistics(paths)
    write_outputs(summary, args.output_json, args.output_report)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_report}")


if __name__ == "__main__":
    main()
