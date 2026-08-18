from __future__ import annotations

"""Build an auditable supplement describing benchmark method parameters.

The output is intentionally derived from the serialized ``parameters_json`` and
``profile_json`` fields in the frozen evidence.  Fixed rules are separated from
quantities that varied across data sets or seeds, and every compact summary is
paired with a hash of the exact distinct JSON values it represents.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results/published/v0.2.3/final_synthetic_robustness.csv"
DEFAULT_OUTPUT = REPO / "manuscript/oedm2026/tables/baseline_parameter_profiles.csv"
DEFAULT_MARKDOWN = REPO / "manuscript/oedm2026/tables/baseline_parameter_profiles.md"
DEFAULT_PROTOCOLS = [
    REPO / "benchmarks/protocol.v0.2.2-seven-family-robustness.json",
    REPO / "benchmarks/protocol.v0.2.3-final-figures-stratascan-robustness.json",
]

METHOD_ORDER = [
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "AMD-DBSCAN",
    "kNN-DBSCAN",
    "kNN+Leiden",
    "StrataSCAN",
]

# These profile fields are compact realized outcomes of adaptive choices.  The
# full profile JSON remains in the evidence CSV and is covered by its file hash.
REALIZED_PROFILE_KEYS = [
    "mdl_gamma_selected_components",
    "mdl_gamma_search_caps",
    "mdl_gamma_search_initial_max",
    "mdl_gamma_search_hard_max",
    "mdl_gamma_background_components",
    "mdl_gamma_background_fraction",
    "optimization_clusters",
    "optimization_noise_fraction",
]

LABEL_USE_POLICY = (
    "Unsupervised fit: run_method receives X and a random seed, not reference labels; "
    "reference labels are used only by the evaluation stage."
)

IMPLEMENTATION_NOTES = {
    "DBSCAN": "scikit-learn DBSCAN; data-derived epsilon rule in the controlled wrapper",
    "HDBSCAN": "scikit-learn HDBSCAN with the controlled low-/high-dimensional profile",
    "OPTICS": "scikit-learn OPTICS; data-derived max-epsilon rule in the controlled wrapper",
    "SNN-DBSCAN": "controlled sparse shared-nearest-neighbor reimplementation",
    "VDBSCAN-2007": "controlled sequential-DBSCAN reimplementation with detected epsilon scales",
    "AMD-DBSCAN": "official-compatible dense compatibility path",
    "kNN-DBSCAN": "controlled corrected-paper-semantics implementation",
    "kNN+Leiden": "controlled kNN graph plus Leiden community-detection comparator",
    "StrataSCAN": "released StrataSCAN estimator invoked by the unified benchmark wrapper",
}

LIMITATIONS = {
    "DBSCAN": "Wrapper profile is benchmark-specific and is not an exhaustive parameter search.",
    "HDBSCAN": "Library implementation and fixed profile; no supervised tuning per data set.",
    "OPTICS": "Wrapper profile is benchmark-specific and is not an exhaustive parameter search.",
    "SNN-DBSCAN": "Controlled reimplementation; not a system-performance reproduction of every published implementation.",
    "VDBSCAN-2007": "Controlled reimplementation; detected scales are data-adaptive and can vary by realization.",
    "AMD-DBSCAN": "Dense O(n^2) compatibility path; memory-limit failures may lack realized parameter JSON.",
    "kNN-DBSCAN": "Controlled corrected-semantics implementation rather than authors' original software.",
    "kNN+Leiden": "Graph-community comparator does not explicitly model a noise class.",
    "StrataSCAN": "Adaptive realized quantities are descriptive outcomes, not user-tuned hyperparameters.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def values_sha256(values: Iterable[Any]) -> str:
    encoded = sorted({canonical_json(value) for value in values})
    return hashlib.sha256(("\n".join(encoded) + "\n").encode("utf-8")).hexdigest()


def parse_json_object(value: object, *, column: str, row_number: object) -> dict[str, Any]:
    if value is None or value is pd.NA:
        return {}
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return {}
    if value == "":
        return {}
    if isinstance(value, dict):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exception:
            raise ValueError(f"invalid {column} JSON at row {row_number}: {exception}") from exception
    if not isinstance(parsed, dict):
        raise ValueError(f"{column} at row {row_number} must encode a JSON object")
    return parsed


def summarize_values(values: Sequence[Any], *, total_records: int) -> dict[str, Any]:
    """Summarize values while hashing the exact distinct canonical JSON set."""
    distinct_json = sorted({canonical_json(value) for value in values})
    result: dict[str, Any] = {
        "observed_records": len(values),
        "missing_records": total_records - len(values),
        "n_distinct": len(distinct_json),
        "distinct_values_sha256": hashlib.sha256(
            ("\n".join(distinct_json) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    if not values:
        return result

    numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
    lists = all(isinstance(value, list) for value in values)
    if numeric:
        array = np.asarray(values, dtype=float)
        result.update(
            {
                "kind": "numeric",
                "minimum": float(np.min(array)),
                "maximum": float(np.max(array)),
            }
        )
    elif lists:
        lengths = [len(value) for value in values]
        result.update(
            {
                "kind": "list",
                "minimum_length": min(lengths),
                "maximum_length": max(lengths),
            }
        )
        elements = [element for value in values for element in value]
        if elements and all(
            isinstance(element, (int, float)) and not isinstance(element, bool)
            for element in elements
        ):
            result["element_minimum"] = float(min(elements))
            result["element_maximum"] = float(max(elements))
    else:
        result["kind"] = "categorical_or_structured"

    if len(distinct_json) <= 8:
        result["distinct_values"] = [json.loads(value) for value in distinct_json]
    return result


def _dimension_regime(row: pd.Series, parameters: dict[str, Any]) -> str | None:
    profile = parameters.get("geometry_profile")
    if profile in {"low_dim", "high_dim"}:
        return str(profile)
    dimension = pd.to_numeric(pd.Series([row.get("dimension")]), errors="coerce").iat[0]
    if pd.isna(dimension):
        metadata = parse_json_object(
            row.get("dataset_metadata_json"), column="dataset_metadata_json", row_number=row.name
        )
        dimension = pd.to_numeric(pd.Series([metadata.get("dimension")]), errors="coerce").iat[0]
    if pd.isna(dimension):
        match = re.search(r"_(\d+)d(?:__|$)", str(row.get("dataset_id", "")))
        if match:
            dimension = float(match.group(1))
    if pd.isna(dimension):
        return None
    return "low_dim" if float(dimension) <= 2 else "high_dim"


def implementation_sources(method: str, repo: Path = REPO) -> list[Path]:
    common = [repo / "benchmarks/methods.py"]
    if method == "StrataSCAN":
        return common + sorted((repo / "src/stratascan").glob("*.py"))
    sources = common + [repo / "baselines/algorithms.py"]
    if method == "SNN-DBSCAN":
        sources.append(repo / "baselines/snn_kernel.py")
    return sources


def source_bundle(method: str, repo: Path = REPO) -> tuple[list[dict[str, str]], str]:
    records: list[dict[str, str]] = []
    for path in implementation_sources(method, repo):
        if not path.exists():
            continue
        records.append({"path": path.relative_to(repo).as_posix(), "sha256": sha256_file(path)})
    payload = "\n".join(f"{item['path']}\0{item['sha256']}" for item in records) + "\n"
    return records, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_resource_profiles(protocol_paths: Sequence[Path], repo: Path = REPO) -> dict[str, Any]:
    protocols: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for path in protocol_paths:
        if not path.exists():
            raise FileNotFoundError(f"resource protocol is missing: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        resource = record.get("resources", {})
        if resource and resource not in profiles:
            profiles.append(resource)
        try:
            display_path = path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            display_path = str(path.resolve())
        protocols.append(
            {
                "path": display_path,
                "sha256": sha256_file(path),
                "protocol_version": record.get("protocol_version"),
                "methods": record.get("methods", []),
                "resources": resource,
            }
        )
    return {"profiles": profiles, "protocols": protocols}


def resource_profiles_for_method(
    resource_profiles: dict[str, Any], method: str, package_versions: set[str]
) -> dict[str, Any]:
    protocols = [
        record
        for record in resource_profiles.get("protocols", [])
        if method in record.get("methods", [])
    ]
    version_matches = [
        record
        for record in protocols
        if any(str(record.get("protocol_version", "")).startswith(version) for version in package_versions)
    ]
    if version_matches:
        protocols = version_matches
    profiles: list[dict[str, Any]] = []
    for record in protocols:
        resource = record.get("resources", {})
        if resource and resource not in profiles:
            profiles.append(resource)
    return {"profiles": profiles, "protocols": protocols}


def build_profiles(
    frame: pd.DataFrame,
    *,
    source_path: Path,
    resource_profiles: dict[str, Any],
    repo: Path = REPO,
) -> pd.DataFrame:
    required = {"method", "status", "parameters_json", "profile_json"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"results CSV lacks required columns: {missing}")

    prepared = frame.copy()
    prepared["_parameters"] = [
        parse_json_object(value, column="parameters_json", row_number=index)
        for index, value in prepared["parameters_json"].items()
    ]
    prepared["_profile"] = [
        parse_json_object(value, column="profile_json", row_number=index)
        for index, value in prepared["profile_json"].items()
    ]
    prepared["dimension_regime"] = [
        _dimension_regime(row, parameters)
        for (_, row), parameters in zip(prepared.iterrows(), prepared["_parameters"], strict=True)
    ]
    prepared = prepared.loc[prepared["dimension_regime"].isin(["low_dim", "high_dim"])]
    source_hash = sha256_file(source_path)
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    rows: list[dict[str, Any]] = []

    for (method, regime), group in prepared.groupby(["method", "dimension_regime"], observed=True):
        parameter_records = [record for record in group["_parameters"] if record]
        fixed: dict[str, Any] = {}
        realized: dict[str, Any] = {}
        keys = sorted(set().union(*(record.keys() for record in parameter_records))) if parameter_records else []
        for key in keys:
            values = [record[key] for record in parameter_records if key in record]
            encoded = {canonical_json(value) for value in values}
            if len(values) == len(parameter_records) and len(encoded) == 1:
                fixed[key] = values[0]
            else:
                realized[key] = summarize_values(values, total_records=len(parameter_records))

        profile_records = [record for record in group["_profile"] if record]
        for key in REALIZED_PROFILE_KEYS:
            values = [record[key] for record in profile_records if key in record]
            if values:
                realized[f"profile.{key}"] = summarize_values(
                    values, total_records=len(profile_records)
                )

        dimensions = sorted(
            {
                int(value)
                for value in pd.to_numeric(group.get("dimension"), errors="coerce").dropna()
            }
        ) if "dimension" in group else []
        status_counts = {
            str(key): int(value) for key, value in group["status"].value_counts(dropna=False).items()
        }
        package_versions = (
            {
                str(value)
                for value in group["package_version"].dropna().unique()
                if str(value)
            }
            if "package_version" in group
            else set()
        )
        method_resources = resource_profiles_for_method(
            resource_profiles, str(method), package_versions
        )
        sources, bundle_hash = source_bundle(str(method), repo)
        exact_parameter_records = sorted({canonical_json(record) for record in parameter_records})
        rows.append(
            {
                "method": str(method),
                "dimension_regime": str(regime),
                "observed_dimensions_json": canonical_json(dimensions),
                "evidence_rows": int(len(group)),
                "successful_rows": int(group["status"].eq("ok").sum()),
                "status_counts_json": canonical_json(status_counts),
                "parameter_records_available": len(parameter_records),
                "distinct_parameter_records": len(exact_parameter_records),
                "distinct_parameter_records_sha256": hashlib.sha256(
                    ("\n".join(exact_parameter_records) + "\n").encode("utf-8")
                ).hexdigest(),
                "fixed_parameter_rules_json": canonical_json(fixed),
                "adaptive_or_realized_parameters_json": canonical_json(realized),
                "implementation": IMPLEMENTATION_NOTES.get(str(method), "unified benchmark method"),
                "implementation_sources_json": canonical_json(sources),
                "implementation_source_bundle_sha256": bundle_hash,
                "label_use_policy": LABEL_USE_POLICY,
                "resource_profile_json": canonical_json(method_resources),
                "limitations": LIMITATIONS.get(
                    str(method), "Interpret within the exact controlled benchmark implementation."
                ),
                "evidence_source": (
                    source_path.resolve().relative_to(repo.resolve()).as_posix()
                    if source_path.resolve().is_relative_to(repo.resolve())
                    else str(source_path.resolve())
                ),
                "evidence_source_sha256": source_hash,
                "_method_rank": method_rank.get(str(method), len(method_rank)),
                "_regime_rank": 0 if regime == "low_dim" else 1,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no low- or high-dimensional parameter profiles were found")
    return (
        result.sort_values(["_method_rank", "_regime_rank", "method"], kind="stable")
        .drop(columns=["_method_rank", "_regime_rank"])
        .reset_index(drop=True)
    )


def _compact_adaptive(value: str) -> str:
    summary = json.loads(value)
    parts: list[str] = []
    for key, item in summary.items():
        if item.get("kind") == "numeric":
            detail = f"{item['minimum']:.4g}-{item['maximum']:.4g}"
        elif item.get("kind") == "list":
            detail = f"length {item['minimum_length']}-{item['maximum_length']}"
            if "element_minimum" in item:
                detail += f", elements {item['element_minimum']:.4g}-{item['element_maximum']:.4g}"
        else:
            detail = f"{item['n_distinct']} distinct"
        parts.append(f"`{key}` ({detail}; n={item['observed_records']})")
    return "; ".join(parts) if parts else "None observed"


def _compact_fixed(value: str) -> str:
    fixed = json.loads(value)
    return "; ".join(f"`{key}={canonical_json(item)}`" for key, item in fixed.items()) or "None recorded"


def render_markdown(table: pd.DataFrame, *, output_csv: Path) -> str:
    source = table.iloc[0]["evidence_source"]
    source_hash = table.iloc[0]["evidence_source_sha256"]
    lines = [
        "# Baseline and StrataSCAN parameter profiles",
        "",
        "This generated supplement records the exact controlled implementations used in the synthetic benchmark. "
        "It is descriptive, not a claim that every authors' production implementation has identical system performance.",
        "",
        f"- Evidence CSV: `{source}`",
        f"- Evidence SHA-256: `{source_hash}`",
        f"- Generated table: `{output_csv.name}`",
        f"- Label-use policy: {LABEL_USE_POLICY}",
        "- Adaptive-value policy: the readable ranges below are accompanied in the CSV by counts and SHA-256 hashes of the exact distinct canonical JSON values.",
        "- Resource policy: `resource_profile_json` in the CSV records the protocol files, their hashes, and their declared thread/time/memory limits.",
        "",
        "| Method | Regime | Fixed rules observed | Adaptive or realized quantities | Coverage | Implementation and limitation |",
        "|---|---|---|---|---:|---|",
    ]
    for _, row in table.iterrows():
        coverage = f"{int(row['parameter_records_available'])}/{int(row['evidence_rows'])} parameter records"
        implementation = f"{row['implementation']} {row['limitations']}"
        cells = [
            str(row["method"]),
            str(row["dimension_regime"]).replace("_", " "),
            _compact_fixed(str(row["fixed_parameter_rules_json"])),
            _compact_adaptive(str(row["adaptive_or_realized_parameters_json"])),
            coverage,
            implementation,
        ]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "A missing realized parameter record generally indicates that execution ended before the method returned its metadata; it is not imputed. "
            "Low dimensional means ambient dimension <= 2 and high dimensional means ambient dimension > 2, matching the benchmark wrapper. "
            "The complete per-run parameter and profile JSON remains in the hashed evidence CSV.",
            "",
            "## Implementation provenance",
            "",
            "Each CSV row includes `implementation_sources_json` with repository-relative file hashes and an `implementation_source_bundle_sha256`. "
            "These hashes make changes in a wrapper or implementation visible even when the human-readable parameter summary is unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def write_profiles(
    results_path: Path,
    output_csv: Path,
    output_markdown: Path,
    *,
    protocol_paths: Sequence[Path],
    repo: Path = REPO,
) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"frozen synthetic evidence is missing: {results_path}")
    frame = pd.read_csv(results_path, low_memory=False)
    resources = load_resource_profiles(protocol_paths, repo) if protocol_paths else {
        "profiles": [],
        "protocols": [],
        "note": "resource policy not supplied; it is not encoded in the results CSV",
    }
    table = build_profiles(
        frame, source_path=results_path, resource_profiles=resources, repo=repo
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    output_markdown.write_text(
        render_markdown(table, output_csv=output_csv), encoding="utf-8"
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the auditable benchmark parameter-profile supplement"
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--protocol",
        type=Path,
        action="append",
        default=None,
        help="resource-policy protocol; repeat when frozen evidence combines campaigns",
    )
    args = parser.parse_args()
    protocols = args.protocol if args.protocol is not None else DEFAULT_PROTOCOLS
    table = write_profiles(
        args.results,
        args.output_csv,
        args.output_markdown,
        protocol_paths=protocols,
    )
    print(
        json.dumps(
            {
                "results": str(args.results.resolve()),
                "rows": len(table),
                "output_csv": str(args.output_csv.resolve()),
                "output_markdown": str(args.output_markdown.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
