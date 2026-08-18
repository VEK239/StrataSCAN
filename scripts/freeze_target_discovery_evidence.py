from __future__ import annotations

"""Freeze the canonical target-discovery-v1 evidence tables.

All source runs are validated before the output directory is created.  The only
cross-generation transformation is the exact single-target Gaia rescore
documented in each Gaia provenance record.
"""

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results/runs"
DEFAULT_OUTPUT_ROOT = REPO / "results/published/v0.2.4/target-discovery-v1"
DEFAULT_SYNTHETIC = RUNS / "v0.2.4-target-discovery-v1-synthetic/results.csv"
DEFAULT_NOISE = RUNS / "v0.2.4-target-discovery-v1-noise-factorial/results.csv"
DEFAULT_CYTOMETRY = RUNS / "v0.2.4-target-discovery-v1-cytometry/results.csv"
DEFAULT_SCALING = RUNS / "v0.2.4-target-discovery-v1-scaling/results.csv"
DEFAULT_GAIA = RUNS / "v0.2.4-target-discovery-v1-gaia/results.csv"
# External methods are intentionally reused from the audited pre-patch panel.
DEFAULT_LEGACY_GAIA = RUNS / "publication_full_7synthetic_20260722/results.csv"
LEGACY_GAIA_SOURCE_GENERATION = "v0.2.3_external"

PROTOCOL_VERSION = "target-discovery-v1"
NATIVE_PACKAGE_VERSION = "0.2.4"
MATCHING_STRATEGY = "hungarian"
MATCHING_OBJECTIVE = "pairwise_f1"
PURITY_THRESHOLD = 0.90
COVERAGE_THRESHOLD = 0.10
TERMINAL_STATUSES = {"ok", "error", "timeout", "memory_limit"}
METHODS = {
    "DBSCAN", "HDBSCAN", "OPTICS", "SNN-DBSCAN", "VDBSCAN-2007",
    "AMD-DBSCAN", "kNN-DBSCAN", "kNN+Leiden", "StrataSCAN",
}
DEVELOPMENT_SEEDS = {23, 42, 73, 101, 151}
FRESH_SEEDS = {211, 223, 227}

COMMON_SEMANTIC_COLUMNS = {
    "evaluation_protocol_version", "evaluation_mode", "background_semantics",
    "matching_strategy", "matching_objective", "primary_target_aggregation",
    "discovery_purity_threshold", "discovery_coverage_threshold",
    "macro_target_f1", "macro_target_purity", "macro_target_coverage",
    "target_discovery_rate",
}
IDENTITY_COLUMNS = ["suite", "dataset_id", "method", "seed"]
DIAGNOSTIC_TEXT_COLUMNS = ("error", "traceback")


@dataclass(frozen=True)
class ExpectedCounts:
    synthetic_release: int = 504
    noise_factorial: int = 270
    cytometry: int = 117
    synthetic_scaling: int = 28
    gaia_current: int = 359
    gaia_fields: int = 359
    synthetic_cells_per_seed: int = 63
    cytometry_datasets: int = 13


@dataclass(frozen=True)
class SourcePaths:
    synthetic_release: Path = DEFAULT_SYNTHETIC
    noise_factorial: Path = DEFAULT_NOISE
    cytometry: Path = DEFAULT_CYTOMETRY
    synthetic_scaling: Path = DEFAULT_SCALING
    gaia_current: Path = DEFAULT_GAIA
    gaia_legacy: Path = DEFAULT_LEGACY_GAIA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    """Return a privacy-safe repository-relative POSIX path."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO.resolve())
    except ValueError as error:
        raise ValueError(f"provenance path is outside the repository: {resolved.name}") from error
    return relative.as_posix()


def _sanitize_diagnostic(value: Any) -> Any:
    """Remove machine-specific paths without changing diagnostic meaning."""
    if pd.isna(value):
        return value
    text = str(value)
    repository = str(REPO.resolve())
    text = re.sub(re.escape(repository), "<repo>", text, flags=re.IGNORECASE)
    python_environment = re.compile(
        r"[A-Za-z]:[\\/]Users[\\/][^\\/\r\n\"]+[\\/]AppData[\\/]"
        r"(?:Roaming[\\/]Python[\\/]Python\d+|Local[\\/]Programs[\\/]Python[\\/]Python\d+)"
        r"[\\/](?:Lib[\\/])?site-packages",
        flags=re.IGNORECASE,
    )
    text = python_environment.sub("<python-env>/site-packages", text)
    text = re.sub(
        r"[A-Za-z]:[\\/]Users[\\/][^\\/\r\n\"]+",
        "<user-home>",
        text,
        flags=re.IGNORECASE,
    )
    # Only diagnostic fields are normalized, so POSIX separators improve
    # reproducibility without touching scientific values or structured data.
    return text.replace("\\", "/")


def _sanitize_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    result = frame.copy()
    changed: dict[str, int] = {}
    for column in DIAGNOSTIC_TEXT_COLUMNS:
        if column not in result:
            continue
        original = result[column].copy()
        result[column] = result[column].map(_sanitize_diagnostic)
        count = int((original.fillna("").astype(str) != result[column].fillna("").astype(str)).sum())
        changed[column] = count
    return result, changed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _source_records(results_path: Path) -> list[dict[str, Any]]:
    records = []
    for path in (
        results_path,
        results_path.parent / "manifest.json",
        results_path.parent / "validation.json",
    ):
        records.append({
            "path": _repo_relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)
        })
    return records


def _validate_manifest_inputs(manifest: dict[str, Any], label: str) -> None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError(f"{label} manifest has no input checksum list")
    for item in inputs:
        if not isinstance(item, dict) or not {"path", "bytes", "sha256"} <= set(item):
            raise ValueError(f"{label} manifest contains a malformed input checksum")
        path = Path(str(item["path"]))
        path = path if path.is_absolute() else REPO / path
        if not path.is_file():
            raise FileNotFoundError(f"{label} frozen input is missing: {path}")
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise ValueError(f"{label} frozen input checksum drift: {path}")


def _validate_resolved_evaluation(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} has no resolved evaluation contract")
    matching = value.get("matching", {})
    thresholds = value.get("discovery_thresholds", {})
    exact = {
        "protocol_version": value.get("protocol_version"),
        "matching_strategy": matching.get("strategy"),
        "matching_objective": matching.get("objective"),
    }
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "matching_strategy": MATCHING_STRATEGY,
        "matching_objective": MATCHING_OBJECTIVE,
    }
    if exact != expected:
        raise ValueError(f"{label} has the wrong resolved evaluation contract: {exact}")
    if not np.isclose(float(thresholds.get("purity", np.nan)), PURITY_THRESHOLD):
        raise ValueError(f"{label} has the wrong discovery purity threshold")
    if not np.isclose(float(thresholds.get("coverage", np.nan)), COVERAGE_THRESHOLD):
        raise ValueError(f"{label} has the wrong discovery coverage threshold")


def _validate_new_semantics(frame: pd.DataFrame, suite: str, label: str) -> None:
    _require_columns(
        frame,
        {*IDENTITY_COLUMNS, "status", "package_version", "evaluation_artifact_path", "evaluation_artifact_sha256",
         "evaluation_artifact_bytes", *COMMON_SEMANTIC_COLUMNS},
        label,
    )
    if set(frame["suite"].astype(str)) != {suite}:
        raise ValueError(f"{label} contains the wrong suite")
    unknown = set(frame["status"].astype(str)) - TERMINAL_STATUSES
    if unknown:
        raise ValueError(f"{label} contains nonterminal statuses: {sorted(unknown)}")
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(f"{label} contains duplicate identities")
    successful = frame.loc[frame["status"].eq("ok")]
    if successful.empty:
        raise ValueError(f"{label} contains no successful rows")
    if (
        successful["package_version"].isna().any()
        or set(successful["package_version"].astype(str)) != {NATIVE_PACKAGE_VERSION}
    ):
        raise ValueError(f"{label} successful rows are not package version {NATIVE_PACKAGE_VERSION}")
    exact = {
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "matching_strategy": MATCHING_STRATEGY,
        "matching_objective": MATCHING_OBJECTIVE,
        "primary_target_aggregation": "macro",
    }
    for column, expected in exact.items():
        if set(successful[column].dropna().astype(str)) != {expected}:
            raise ValueError(f"{label} has wrong {column}")
    for column, expected in (
        ("discovery_purity_threshold", PURITY_THRESHOLD),
        ("discovery_coverage_threshold", COVERAGE_THRESHOLD),
    ):
        values = pd.to_numeric(successful[column], errors="coerce")
        if values.isna().any() or not np.allclose(values, expected):
            raise ValueError(f"{label} has wrong {column}")
    expected_background = {
        "synthetic": "known_synthetic_noise",
        "cytometry": "heterogeneous_biological_background",
        "gaia": "reference_negative_background",
    }[suite]
    if set(successful["background_semantics"].astype(str)) != {expected_background}:
        raise ValueError(f"{label} has wrong background semantics")
    expected_modes = successful["dataset_id"].astype(str).map(
        lambda dataset: (
            "target_discovery"
            if suite == "gaia" or (suite == "cytometry" and dataset in {"mosmann", "nilsson"})
            else "full_partition"
        )
    )
    if not successful["evaluation_mode"].astype(str).reset_index(drop=True).equals(
        expected_modes.reset_index(drop=True)
    ):
        raise ValueError(f"{label} has wrong evaluation modes")
    if successful[[
        "evaluation_artifact_path", "evaluation_artifact_sha256", "evaluation_artifact_bytes"
    ]].isna().any().any():
        raise ValueError(f"{label} successful rows lack lossless artifact checksums")
    for column in (
        "macro_target_f1", "macro_target_purity", "macro_target_coverage",
        "target_discovery_rate",
    ):
        values = pd.to_numeric(successful[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{label} has invalid successful {column}")
    if suite != "synthetic" and {"noise_f1", "noise_precision", "noise_recall"} & set(frame):
        raise ValueError(f"{label} exposes synthetic-noise names for reference background")


def _load_new_run(path: Path, label: str, expected_rows: int, suite: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} results are not complete: {path}")
    validation = _read_json(path.parent / "validation.json", f"{label} validation")
    manifest = _read_json(path.parent / "manifest.json", f"{label} manifest")
    if (
        validation.get("complete") is not True
        or int(validation.get("expected", -1)) != expected_rows
        or int(validation.get("found", -1)) != expected_rows
    ):
        raise ValueError(f"{label} source run is incomplete or has the wrong expected cell count")
    if (
        int(manifest.get("expected_job_count", -1)) != expected_rows
        or len(manifest.get("jobs", [])) != expected_rows
    ):
        raise ValueError(f"{label} manifest has the wrong expected cell count")
    _validate_resolved_evaluation(manifest.get("evaluation"), f"{label} manifest")
    _validate_manifest_inputs(manifest, label)
    for job in manifest["jobs"]:
        _validate_resolved_evaluation(job.get("evaluation"), f"{label} job {job.get('job_id')}")
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != expected_rows:
        raise ValueError(f"{label} results contain {len(frame)} rows; expected {expected_rows}")
    _validate_new_semantics(frame, suite, label)
    manifest_keys = pd.DataFrame(manifest["jobs"])[IDENTITY_COLUMNS]
    if manifest_keys.duplicated().any():
        raise ValueError(f"{label} manifest contains duplicate identities")
    result_keys = frame[IDENTITY_COLUMNS]
    observed = set(map(tuple, result_keys.itertuples(index=False, name=None)))
    declared = set(map(tuple, manifest_keys.itertuples(index=False, name=None)))
    if observed != declared:
        raise ValueError(f"{label} results identities differ from the frozen manifest")
    return frame, _source_records(path)


def _validate_panel(frame: pd.DataFrame, label: str, methods: set[str] | None = None) -> None:
    observed = set(frame["method"].astype(str))
    if methods is not None and observed != methods:
        raise ValueError(f"{label} method panel mismatch: {sorted(observed)}")


def _prepare_new_sources(paths: SourcePaths, counts: ExpectedCounts) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, Any]]]]:
    specs = {
        "synthetic_release": (paths.synthetic_release, counts.synthetic_release, "synthetic"),
        "noise_factorial": (paths.noise_factorial, counts.noise_factorial, "synthetic"),
        "cytometry": (paths.cytometry, counts.cytometry, "cytometry"),
        "synthetic_scaling": (paths.synthetic_scaling, counts.synthetic_scaling, "synthetic"),
        "gaia_current": (paths.gaia_current, counts.gaia_current, "gaia"),
    }
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for name, (path, expected, suite) in specs.items():
        frames[name], sources[name] = _load_new_run(path, name.replace("_", " "), expected, suite)

    release = frames["synthetic_release"].copy()
    seeds = pd.to_numeric(release["seed"], errors="raise").astype(int)
    unknown = set(seeds) - DEVELOPMENT_SEEDS - FRESH_SEEDS
    if unknown:
        raise ValueError(f"synthetic release has seeds outside the frozen stage policy: {sorted(unknown)}")
    release["evidence_stage"] = np.where(seeds.isin(DEVELOPMENT_SEEDS), "development", "fresh")
    if (
        release.groupby("seed", observed=True).size()
        != counts.synthetic_cells_per_seed
    ).any():
        raise ValueError("synthetic release has the wrong family-method cells per seed")
    _validate_panel(release, "synthetic release", METHODS)
    frames["synthetic_release"] = release

    _validate_panel(frames["noise_factorial"], "noise factorial", METHODS)
    _validate_panel(frames["cytometry"], "cytometry", METHODS)
    if frames["cytometry"]["dataset_id"].nunique() != counts.cytometry_datasets:
        raise ValueError("cytometry has the wrong declared dataset count")
    cytometry_matrix = frames["cytometry"][["dataset_id", "method"]].drop_duplicates()
    if len(cytometry_matrix) != counts.cytometry_datasets * len(METHODS):
        raise ValueError("cytometry dataset-method matrix is incomplete")
    _validate_panel(frames["synthetic_scaling"], "synthetic scaling", {"StrataSCAN"})
    if frames["synthetic_scaling"]["dataset_id"].nunique() != counts.synthetic_scaling:
        raise ValueError("synthetic scaling identities are incomplete")
    _validate_panel(frames["gaia_current"], "current Gaia", {"StrataSCAN"})
    if frames["gaia_current"]["dataset_id"].nunique() != counts.gaia_fields:
        raise ValueError("current Gaia does not cover the declared field count")
    for frame in frames.values():
        frame["evaluation_generation"] = "native_target_discovery_v1"
    return frames, sources


def _legacy_target_match_check(row: pd.Series) -> None:
    try:
        matches = json.loads(row["target_matches_json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("legacy Gaia has malformed target_matches_json") from error
    if not isinstance(matches, list) or len(matches) != 1:
        raise ValueError("legacy Gaia single-target proof failed: target match count is not one")
    match = matches[0]
    for old, new in (
        ("precision", "best_cluster_precision"),
        ("recall", "best_cluster_recall"),
        ("f1", "best_cluster_f1"),
    ):
        if not np.isclose(float(match[old]), float(row[new]), atol=1e-12, rtol=1e-12):
            raise ValueError(f"legacy Gaia single-target proof failed: {old} mismatch")


def _rescore_legacy_gaia(path: Path, fields: set[str]) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"legacy Gaia source is missing: {path}")
    validation = _read_json(path.parent / "validation.json", "legacy Gaia validation")
    manifest = _read_json(path.parent / "manifest.json", "legacy Gaia manifest")
    if validation.get("complete") is not True:
        raise ValueError("legacy Gaia source run is incomplete")
    if int(validation.get("expected", -1)) != int(validation.get("found", -2)):
        raise ValueError("legacy Gaia validation counts disagree")
    if int(manifest.get("expected_job_count", -1)) != int(validation["expected"]):
        raise ValueError("legacy Gaia manifest and validation counts disagree")
    raw = pd.read_csv(path, low_memory=False)
    if len(raw) != int(validation["found"]):
        raise ValueError("legacy source CSV and validation counts disagree")
    legacy = raw.loc[raw["suite"].eq("gaia") & ~raw["method"].eq("StrataSCAN")].copy()
    required = {
        *IDENTITY_COLUMNS, "status", "best_cluster_precision", "best_cluster_recall",
        "best_cluster_f1", "macro_target_f1", "truth_clusters", "target_matches_json",
        "noise_precision", "noise_recall", "noise_f1", "noise_fraction", "n_clusters",
        "spurious_background_majority_clusters",
    }
    _require_columns(legacy, required, "legacy Gaia")
    if legacy.empty or set(legacy["status"].astype(str)) != {"ok"}:
        raise ValueError("legacy external Gaia rows must all be successful")
    if legacy.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("legacy Gaia contains duplicate execution identities")
    if set(legacy["dataset_id"].astype(str)) != fields:
        raise ValueError("legacy and current Gaia field identities differ")
    if not (pd.to_numeric(legacy["truth_clusters"], errors="coerce") == 1).all():
        raise ValueError("legacy Gaia single-target proof failed: truth_clusters != 1")
    if not np.allclose(
        pd.to_numeric(legacy["macro_target_f1"], errors="coerce"),
        pd.to_numeric(legacy["best_cluster_f1"], errors="coerce"),
        atol=1e-12, rtol=1e-12,
    ):
        raise ValueError("legacy Gaia single-target proof failed: macro and best F1 differ")
    for _, row in legacy.iterrows():
        _legacy_target_match_check(row)
    external_methods = set(legacy["method"].astype(str))
    if not external_methods or "StrataSCAN" in external_methods:
        raise ValueError("legacy Gaia external method panel is invalid")
    for method in external_methods:
        if set(legacy.loc[legacy["method"].eq(method), "dataset_id"].astype(str)) != fields:
            raise ValueError(f"legacy Gaia method {method} does not cover every current field")
    for (dataset_id, method), part in legacy.groupby(["dataset_id", "method"], observed=True):
        observed_seeds = set(pd.to_numeric(part["seed"], errors="raise").astype(int))
        expected_seeds = {23, 42, 73} if method == "kNN+Leiden" else {42}
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"legacy Gaia seed policy mismatch for {dataset_id}/{method}: {sorted(observed_seeds)}"
            )

    precision = pd.to_numeric(legacy["best_cluster_precision"], errors="raise")
    coverage = pd.to_numeric(legacy["best_cluster_recall"], errors="raise")
    target_f1 = pd.to_numeric(legacy["best_cluster_f1"], errors="raise")
    computed_f1 = np.divide(
        2 * precision * coverage,
        precision + coverage,
        out=np.zeros(len(legacy), dtype=float),
        where=(precision + coverage).to_numpy() > 0,
    )
    if not np.allclose(target_f1, computed_f1, atol=1e-12, rtol=1e-12):
        raise ValueError("legacy Gaia retained F1 is inconsistent with precision and recall")
    legacy["evaluation_protocol_version"] = PROTOCOL_VERSION
    legacy["evaluation_mode"] = "target_discovery"
    legacy["background_semantics"] = "reference_negative_background"
    legacy["matching_strategy"] = MATCHING_STRATEGY
    legacy["matching_objective"] = MATCHING_OBJECTIVE
    legacy["primary_target_aggregation"] = "macro"
    legacy["discovery_purity_threshold"] = PURITY_THRESHOLD
    legacy["discovery_coverage_threshold"] = COVERAGE_THRESHOLD
    legacy["unmatched_target_score"] = 0.0
    legacy["macro_target_f1"] = target_f1
    legacy["weighted_target_f1"] = target_f1
    legacy["macro_target_purity"] = precision
    legacy["weighted_target_purity"] = precision
    legacy["macro_target_coverage"] = coverage
    legacy["weighted_target_coverage"] = coverage
    discovered = (precision >= PURITY_THRESHOLD) & (coverage >= COVERAGE_THRESHOLD)
    legacy["target_discovery_rate"] = discovered.astype(float)
    legacy["weighted_target_discovery_rate"] = discovered.astype(float)
    legacy["discovered_target_count"] = discovered.astype(int)
    legacy["target_count"] = 1
    legacy["predicted_cluster_count"] = pd.to_numeric(legacy["n_clusters"], errors="raise")
    legacy["matched_predicted_cluster_count"] = (target_f1 > 0).astype(int)
    legacy["unmatched_predicted_cluster_count"] = (
        legacy["predicted_cluster_count"] - legacy["matched_predicted_cluster_count"]
    )
    legacy["candidate_clusters_per_target"] = legacy["predicted_cluster_count"]
    legacy["background_majority_predicted_clusters"] = pd.to_numeric(
        legacy["spurious_background_majority_clusters"], errors="raise"
    )
    legacy["candidate_clusters_with_target_overlap"] = np.nan
    legacy["background_only_predicted_clusters"] = np.nan
    legacy["background_abstention_precision_diagnostic"] = pd.to_numeric(
        legacy["noise_precision"], errors="raise"
    )
    legacy["background_abstention_fraction"] = pd.to_numeric(
        legacy["noise_recall"], errors="raise"
    )
    legacy["background_abstention_f1_diagnostic"] = pd.to_numeric(
        legacy["noise_f1"], errors="raise"
    )
    legacy["predicted_abstention_fraction"] = pd.to_numeric(
        legacy["noise_fraction"], errors="raise"
    )
    legacy["evaluation_generation"] = "legacy_single_target_exact_rescore"

    legacy = legacy.drop(
        columns=[
            column for column in (
                "noise_precision", "noise_recall", "noise_f1", "noise_fraction",
                "binary_macro_f1", "binary_balanced_accuracy",
            ) if column in legacy
        ]
    )

    numeric = legacy.select_dtypes(include=[np.number]).columns.tolist()
    rows: list[dict[str, Any]] = []
    for (dataset_id, method), part in legacy.groupby(["dataset_id", "method"], observed=True):
        first = part.iloc[0].to_dict()
        for column in numeric:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            first[column] = float(values.median()) if len(values) else np.nan
        seeds = sorted(pd.to_numeric(part["seed"], errors="raise").astype(int).tolist())
        first["seed"] = seeds[0] if len(seeds) == 1 else "median[23,42,73]"
        first["seed_aggregation"] = "single_execution" if len(seeds) == 1 else "componentwise_median"
        first["source_seed_count"] = len(seeds)
        first["job_id"] = f"legacy-rescore__{dataset_id}__{method}"
        purity_value = float(first["macro_target_purity"])
        coverage_value = float(first["macro_target_coverage"])
        f1_value = float(first["macro_target_f1"])
        first["target_matches_json"] = json.dumps(
            [
                {
                    "target": str(dataset_id),
                    "target_label": 0,
                    "matched_predicted_cluster": 0 if f1_value > 0 else None,
                    "purity": purity_value,
                    "coverage": coverage_value,
                    "precision": purity_value,
                    "recall": coverage_value,
                    "f1": f1_value,
                    "passes_discovery_thresholds": bool(
                        purity_value >= PURITY_THRESHOLD
                        and coverage_value >= COVERAGE_THRESHOLD
                    ),
                    "evaluation_generation": "legacy_single_target_exact_rescore",
                    "seed_aggregation": first["seed_aggregation"],
                }
            ],
            sort_keys=True,
        )
        rows.append(first)
    result = pd.DataFrame(rows)
    result["evaluation_generation"] = "legacy_single_target_exact_rescore"
    if result.duplicated(["suite", "dataset_id", "method"]).any():
        raise ValueError("collapsed legacy Gaia contains duplicate field-method identities")
    proof = {
        "name": "single_target_hungarian_equivalence",
        "source_generation": LEGACY_GAIA_SOURCE_GENERATION,
        "statement": (
            "For exactly one reference target, Hungarian maximum pairwise-F1 selects the same "
            "best predicted cluster as legacy independent best-cluster matching."
        ),
        "validated_conditions": [
            "truth_clusters == 1 for every source execution",
            "target_matches_json contains exactly one target",
            "stored target precision, recall, and F1 equal best-cluster values",
            "macro_target_f1 equals best_cluster_f1",
            "stored F1 equals the harmonic mean of retained precision and recall",
        ],
        "metric_mapping": {
            "best_cluster_precision": "macro_target_purity",
            "best_cluster_recall": "macro_target_coverage",
            "best_cluster_f1": "macro_target_f1 (unchanged)",
            "discovery": "purity >= 0.9 and coverage >= 0.1",
        },
        "seed_policy": {
            "kNN+Leiden": "component-wise median over seeds 23, 42, and 73",
            "other_external_methods": "single declared seed 42",
        },
        "unavailable_not_fabricated": [
            "candidate_clusters_with_target_overlap", "background_only_predicted_clusters",
        ],
    }
    return result, _source_records(path), proof


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    handle = io.StringIO(newline="")
    frame.to_csv(handle, index=False)
    return handle.getvalue().encode("utf-8")


def _atomic_csv(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w+b", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _provenance(
    name: str,
    output: Path,
    frame: pd.DataFrame,
    sources: list[dict[str, Any]],
    transformation: str,
    diagnostic_changes: dict[str, int],
    legacy_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "purpose": f"canonical {name.replace('_', ' ')} evidence",
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "matching_strategy": MATCHING_STRATEGY,
        "matching_objective": MATCHING_OBJECTIVE,
        "discovery_purity_threshold": PURITY_THRESHOLD,
        "discovery_coverage_threshold": COVERAGE_THRESHOLD,
        "transformation": transformation,
        "diagnostic_path_normalization": {
            "columns": list(DIAGNOSTIC_TEXT_COLUMNS),
            "changed_cells": diagnostic_changes,
            "rules": {
                "repository_absolute_paths": "<repo>/...",
                "python_environment_absolute_paths": "<python-env>/...",
                "other_user_home_absolute_paths": "<user-home>/...",
                "path_separators": "POSIX forward slash",
            },
            "scope": "textual diagnostics only; scientific metrics, identities, and statuses unchanged",
        },
        "sources": sources,
        "output": _repo_relative(output),
        "rows": int(len(frame)),
        "methods": sorted(frame["method"].astype(str).unique()),
        "evaluation_generation_counts": {
            str(key): int(count)
            for key, count in frame["evaluation_generation"].value_counts().items()
        },
        "status_counts": {
            str(key): int(count) for key, count in frame["status"].value_counts().items()
        },
        "sha256": sha256(output),
    }
    if legacy_proof is not None:
        value["legacy_single_target_equivalence_proof"] = legacy_proof
    return value


def freeze(
    paths: SourcePaths = SourcePaths(),
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    counts: ExpectedCounts = ExpectedCounts(),
    *,
    provenance_only: bool = False,
) -> dict[str, Any]:
    # Phase 1 is read-only.  No output path is created until every new and legacy
    # source has passed completion, semantic, identity, and checksum validation.
    _repo_relative(output_root)
    for field in SourcePaths.__dataclass_fields__:
        _repo_relative(getattr(paths, field))
    frames, sources = _prepare_new_sources(paths, counts)
    current_fields = set(frames["gaia_current"]["dataset_id"].astype(str))
    external_gaia, legacy_sources, proof = _rescore_legacy_gaia(
        paths.gaia_legacy, current_fields
    )
    gaia = pd.concat([external_gaia, frames["gaia_current"]], ignore_index=True, sort=False)
    if gaia["dataset_id"].nunique() != counts.gaia_fields:
        raise ValueError("merged Gaia field count is wrong")
    if gaia.duplicated(["dataset_id", "method"]).any():
        raise ValueError("merged Gaia contains duplicate field-method rows")
    frames["gaia"] = gaia
    sources["gaia"] = [*legacy_sources, *sources["gaia_current"]]

    outputs = {
        "synthetic_release": output_root / "synthetic_release.csv",
        "noise_factorial": output_root / "noise_factorial.csv",
        "cytometry": output_root / "cytometry.csv",
        "synthetic_scaling": output_root / "synthetic_scaling.csv",
        "gaia": output_root / "gaia.csv",
    }
    prepared: dict[str, tuple[Path, pd.DataFrame, bytes, str, dict[str, int]]] = {}
    for name, output in outputs.items():
        frame = frames[name].sort_values(
            [column for column in ("dataset_id", "method", "seed") if column in frames[name]],
            kind="stable",
        ).reset_index(drop=True)
        frame, diagnostic_changes = _sanitize_diagnostics(frame)
        content = _csv_bytes(frame)
        transformation = (
            "native rows copied after validation; evidence_stage inferred only from frozen seed sets"
            if name == "synthetic_release"
            else "native rows copied after validation"
        )
        if name == "gaia":
            transformation = (
                "native StrataSCAN rows merged with exact single-target legacy external-method "
                "rescores; kNN+Leiden collapsed by component-wise seed median"
            )
        prepared[name] = (output, frame, content, transformation, diagnostic_changes)

    if provenance_only:
        for name, (output, _, content, _, _) in prepared.items():
            if not output.is_file():
                raise FileNotFoundError(
                    f"cannot regenerate {name} provenance; canonical CSV is missing: {output}"
                )
            expected_hash = hashlib.sha256(content).hexdigest()
            if output.stat().st_size != len(content) or sha256(output) != expected_hash:
                raise ValueError(
                    f"cannot regenerate {name} provenance; canonical CSV differs from validated sources"
                )
    else:
        for output, _, content, _, _ in prepared.values():
            _atomic_csv(content, output)

    provenance_records: dict[str, Any] = {}
    for name, (output, frame, _, transformation, diagnostic_changes) in prepared.items():
        provenance = _provenance(
            name, output, frame, sources[name], transformation, diagnostic_changes,
            legacy_proof=proof if name == "gaia" else None,
        )
        provenance_path = output.with_name(output.stem + "-provenance.json")
        _atomic_json(provenance, provenance_path)
        provenance_records[name] = {
            "output": _repo_relative(output),
            "rows": len(frame),
            "sha256": provenance["sha256"],
            "provenance": _repo_relative(provenance_path),
            "provenance_sha256": sha256(provenance_path),
        }
    manifest = {
        "schema_version": 1,
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "policy": (
            "all sources and existing CSV bytes validated before provenance-only regeneration"
            if provenance_only
            else "all sources validated before any canonical output was written"
        ),
        "outputs": provenance_records,
    }
    _atomic_json(manifest, output_root / "freeze-manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze canonical target-discovery-v1 evidence")
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--noise-factorial", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--cytometry", type=Path, default=DEFAULT_CYTOMETRY)
    parser.add_argument("--scaling", type=Path, default=DEFAULT_SCALING)
    parser.add_argument("--gaia", type=Path, default=DEFAULT_GAIA)
    parser.add_argument("--legacy-gaia", type=Path, default=DEFAULT_LEGACY_GAIA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="validate existing canonical CSV bytes and regenerate only JSON provenance",
    )
    args = parser.parse_args()
    paths = SourcePaths(
        args.synthetic.resolve(), args.noise_factorial.resolve(), args.cytometry.resolve(),
        args.scaling.resolve(), args.gaia.resolve(), args.legacy_gaia.resolve(),
    )
    print(json.dumps(
        freeze(
            paths, args.output_root.resolve(), provenance_only=args.provenance_only
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
