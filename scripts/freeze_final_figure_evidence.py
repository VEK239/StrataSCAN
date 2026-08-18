from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = REPO / "results/runs/v0.2.2-seven-family-robustness-v2/results.csv"
DEFAULT_CURRENT = REPO / "results/runs/v0.2.3-final-figures-stratascan-robustness-v1/results.csv"
DEFAULT_OUTPUT = REPO / "results/published/v0.2.3/final_synthetic_robustness.csv"
EXPECTED_CELLS_PER_METHOD = 7 * 5 * 5
BASELINE_METHODS = {
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "AMD-DBSCAN",
    "kNN-DBSCAN",
    "kNN+Leiden",
}
TERMINAL_STATUSES = {"ok", "error", "timeout", "memory_limit"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate(path: Path, label: str, expected_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} evidence is missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"dataset_id", "method", "seed", "status", "suite", "package_version"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} evidence lacks columns: {missing}")
    validation_path = path.parent / "validation.json"
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{label} manifest is missing: {manifest_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"{label} validation is missing: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        not validation.get("complete")
        or int(validation.get("expected", -1)) != expected_rows
        or int(validation.get("found", -1)) != expected_rows
    ):
        raise ValueError(f"{label} validation is incomplete or has the wrong matrix size")
    return frame


def freeze(baseline_path: Path, current_path: Path, output_path: Path) -> dict[str, object]:
    baseline = load_and_validate(
        baseline_path, "baseline", EXPECTED_CELLS_PER_METHOD * (len(BASELINE_METHODS) + 1)
    )
    current = load_and_validate(current_path, "current-release", EXPECTED_CELLS_PER_METHOD)
    if set(baseline["method"].astype(str).unique()) != {*BASELINE_METHODS, "StrataSCAN"}:
        raise ValueError("baseline source does not contain the locked nine-method matrix")
    baseline = baseline.loc[~baseline["method"].eq("StrataSCAN")].copy()

    if set(baseline["method"].astype(str).unique()) != BASELINE_METHODS:
        raise ValueError("baseline evidence does not contain exactly the eight declared comparators")
    if set(current["method"].astype(str).unique()) != {"StrataSCAN"}:
        raise ValueError("current evidence must contain StrataSCAN and no other method")
    if set(baseline["suite"].astype(str).unique()) != {"synthetic"}:
        raise ValueError("baseline evidence contains a non-synthetic suite")
    if set(current["suite"].astype(str).unique()) != {"synthetic"}:
        raise ValueError("current evidence contains a non-synthetic suite")
    if set(current["package_version"].dropna().astype(str).unique()) != {"0.2.3"} or current[
        "package_version"
    ].isna().any():
        raise ValueError("current evidence is not entirely from package version 0.2.3")
    if not set(baseline["status"].astype(str)).issubset(TERMINAL_STATUSES):
        raise ValueError("baseline evidence contains a nonterminal or unknown status")
    if set(current["status"].astype(str)) != {"ok"}:
        raise ValueError("the release robustness claim requires 175 successful current cells")

    if len(current) != EXPECTED_CELLS_PER_METHOD:
        raise ValueError(
            f"expected {EXPECTED_CELLS_PER_METHOD} current StrataSCAN cells, found {len(current)}"
        )
    counts = baseline.groupby("method", observed=True).size()
    incomplete = counts.loc[counts.ne(EXPECTED_CELLS_PER_METHOD)].to_dict()
    if incomplete:
        raise ValueError(f"baseline methods do not cover the locked matrix: {incomplete}")

    keys = ["dataset_id", "seed"]
    baseline_key_sets = {
        method: set(map(tuple, part[keys].itertuples(index=False, name=None)))
        for method, part in baseline.groupby("method", observed=True)
    }
    reference_keys = next(iter(baseline_key_sets.values()))
    mismatched = sorted(method for method, values in baseline_key_sets.items() if values != reference_keys)
    if mismatched:
        raise ValueError(f"baseline methods do not share the same locked cells: {mismatched}")
    current_keys = set(map(tuple, current[keys].itertuples(index=False, name=None)))
    if current_keys != reference_keys:
        missing = sorted(reference_keys - current_keys)[:5]
        extra = sorted(current_keys - reference_keys)[:5]
        raise ValueError(f"current locked cells differ from baselines; missing={missing}, extra={extra}")

    combined = pd.concat([baseline, current], ignore_index=True, sort=False)
    unique_keys = ["dataset_id", "method", "seed"]
    duplicates = combined.duplicated(unique_keys, keep=False)
    if duplicates.any():
        sample = combined.loc[duplicates, unique_keys].head().to_dict("records")
        raise ValueError(f"duplicate final-evidence cells: {sample}")

    combined = combined.sort_values(unique_keys, kind="stable").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    provenance = {
        "purpose": "final synthetic robustness evidence for manuscript figures",
        "policy": (
            "unchanged frozen external baseline rows plus a complete current-release "
            "StrataSCAN rerun; predecessor StrataSCAN rows are excluded"
        ),
        "sources": [
            {"path": str(baseline_path.resolve()), "sha256": sha256(baseline_path)},
            {
                "path": str((baseline_path.parent / "manifest.json").resolve()),
                "sha256": sha256(baseline_path.parent / "manifest.json"),
            },
            {
                "path": str((baseline_path.parent / "validation.json").resolve()),
                "sha256": sha256(baseline_path.parent / "validation.json"),
            },
            {"path": str(current_path.resolve()), "sha256": sha256(current_path)},
            {
                "path": str((current_path.parent / "manifest.json").resolve()),
                "sha256": sha256(current_path.parent / "manifest.json"),
            },
            {
                "path": str((current_path.parent / "validation.json").resolve()),
                "sha256": sha256(current_path.parent / "validation.json"),
            },
        ],
        "output": str(output_path.resolve()),
        "rows": int(len(combined)),
        "methods": sorted(combined["method"].astype(str).unique().tolist()),
        "status_counts": {
            str(key): int(value) for key, value in combined["status"].value_counts().items()
        },
        "sha256": sha256(output_path),
    }
    provenance_path = output_path.with_name(output_path.stem + "-provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze generation-safe evidence for final figures")
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.baselines, args.current, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
