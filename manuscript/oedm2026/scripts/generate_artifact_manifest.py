from __future__ import annotations

"""Generate and verify the canonical OEDM manuscript artifact manifest.

The figure manifest is the authoritative inventory of evidence used to render
the figures.  This script verifies every source hash recorded there, rejects
placeholder figures, and then records the exact PDF, evidence, protocols,
figure outputs, and build inputs that make up the submission artifact.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[3]
MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = MANUSCRIPT_ROOT / "StrataSCAN_OEDM2026_draft.pdf"
DEFAULT_FIGURE_MANIFEST = MANUSCRIPT_ROOT / "final_figures/figure_manifest.json"
DEFAULT_OUTPUT = MANUSCRIPT_ROOT / "artifact_manifest.json"

BUILD_INPUTS = (
    "manuscript/oedm2026/manuscript.tex",
    "manuscript/oedm2026/references.bib",
    "manuscript/oedm2026/scripts/build.ps1",
    "manuscript/oedm2026/scripts/final_manuscript_statistics.py",
    "manuscript/oedm2026/scripts/discovery_threshold_sensitivity.py",
    "manuscript/oedm2026/scripts/synchronize_final_manuscript_statistics.py",
    "manuscript/oedm2026/scripts/final_manuscript_figures.py",
    "manuscript/oedm2026/scripts/generate_artifact_manifest.py",
    "scripts/build_revised_results_presentation.py",
    "manuscript/oedm2026/README.md",
    "manuscript/oedm2026/FINAL_EXPERIMENT_POLICY.md",
    "manuscript/oedm2026/FINAL_FIGURES.md",
    "manuscript/oedm2026/FIGURE_CLAIM_MAP.md",
    "manuscript/oedm2026/SUPPLEMENT.md",
)
SENSITIVITY_OUTPUTS = (
    "manuscript/oedm2026/tables/discovery_threshold_sensitivity.csv",
    "manuscript/oedm2026/tables/discovery_threshold_sensitivity.md",
)
RESULTS_SECTION_OUTPUTS = (
    "manuscript/oedm2026/final_figures/fig2_density_noise.pdf",
    "manuscript/oedm2026/final_figures/fig2_density_noise.svg",
    "manuscript/oedm2026/final_figures/fig3_million_scale.pdf",
    "manuscript/oedm2026/final_figures/fig3_million_scale.svg",
    "manuscript/oedm2026/final_figures/fig4_cytometry.pdf",
    "manuscript/oedm2026/final_figures/fig4_cytometry.svg",
    "manuscript/oedm2026/tables/table_rq1_synthetic_fresh.tex",
    "manuscript/oedm2026/tables/table_rq2_noise99.tex",
    "manuscript/oedm2026/tables/table_rq3_scaling.tex",
)
RESULTS_SECTION_AUDITS = (
    "outputs/rq1_synthetic/rq1_audit.json",
    "outputs/rq2_density_noise/rq2_provenance.json",
    "outputs/rq3_scaling/validation.json",
    "outputs/rq4_cytometry/rq4_audit.json",
    "outputs/rq4_cytometry/merge_aware_sensitivity.csv",
    "outputs/rq4_cytometry/merge_aware_sensitivity.json",
)


class ManifestError(ValueError):
    """Raised when the artifact inventory is incomplete or stale."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ManifestError(f"artifact is outside the repository: {path}") from exc


def _record(path: Path, repo: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return {
        "path": _relative_path(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _repo_path(repo: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"invalid repository-relative path in figure manifest: {value!r}")
    normalized = value.replace("\\", "/")
    path = repo.joinpath(*Path(normalized).parts)
    _relative_path(path, repo)
    return path


def _load_figure_manifest(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"figure manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ManifestError("figure manifest must contain a JSON object")
    placeholders = value.get("placeholders")
    if placeholders != []:
        raise ManifestError(f"figure manifest contains placeholders: {placeholders!r}")
    return value


def _verified_figure_sources(
    figure_manifest: Mapping[str, object], repo: Path
) -> list[Path]:
    values = figure_manifest.get("sources")
    if not isinstance(values, list) or not values:
        raise ManifestError("figure manifest has no source inventory")
    paths: list[Path] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ManifestError(f"figure source {index} is not an object")
        path = _repo_path(repo, value.get("path"))
        expected = value.get("sha256")
        if value.get("exists") is not True or not isinstance(expected, str):
            raise ManifestError(f"figure source is not recorded as complete: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"figure source is missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise ManifestError(
                f"figure source hash is stale for {_relative_path(path, repo)}: "
                f"expected {expected}, observed {observed}"
            )
        paths.append(path)
    return list(dict.fromkeys(paths))


def _listed_files(
    figure_manifest: Mapping[str, object], key: str, directory: Path, repo: Path
) -> list[Path]:
    values = figure_manifest.get(key)
    if not isinstance(values, list):
        raise ManifestError(f"figure manifest field {key!r} must be a list")
    paths: list[Path] = []
    for value in values:
        if not isinstance(value, str) or Path(value).name != value:
            raise ManifestError(f"invalid {key} entry: {value!r}")
        path = directory / value
        _relative_path(path, repo)
        if not path.is_file():
            raise FileNotFoundError(f"listed figure artifact is missing: {path}")
        paths.append(path)
    return list(dict.fromkeys(paths))


def _records(paths: Iterable[Path], repo: Path) -> list[dict[str, object]]:
    return [_record(path, repo) for path in sorted(set(paths), key=lambda item: item.as_posix())]


def build_manifest(
    *,
    repo: Path = REPO,
    pdf: Path = DEFAULT_PDF,
    figure_manifest_path: Path = DEFAULT_FIGURE_MANIFEST,
    build_inputs: Sequence[str] = BUILD_INPUTS,
    results_section_outputs: Sequence[str] = (),
    results_section_audits: Sequence[str] = (),
) -> dict[str, object]:
    repo = repo.resolve()
    figure_manifest = _load_figure_manifest(figure_manifest_path)
    sources = _verified_figure_sources(figure_manifest, repo)
    evidence = [path for path in sources if _relative_path(path, repo).startswith("results/")]
    protocols = [
        path for path in sources if _relative_path(path, repo).startswith("benchmarks/")
    ]
    source_support = [path for path in sources if path not in evidence and path not in protocols]
    figures = _listed_files(figure_manifest, "figures", figure_manifest_path.parent, repo)
    ancillary = _listed_files(
        figure_manifest, "ancillary_files", figure_manifest_path.parent, repo
    )
    inputs = [_repo_path(repo, path) for path in build_inputs]
    sensitivity = [_repo_path(repo, path) for path in SENSITIVITY_OUTPUTS]
    results_outputs = [_repo_path(repo, path) for path in results_section_outputs]
    results_audits = [_repo_path(repo, path) for path in results_section_audits]

    return {
        "schema_version": 2,
        "canonical": True,
        "manuscript": "IEEE ICDM OEDM 2026",
        "evaluated_method": {
            "display_name": "MDL-StrataSCAN",
            "snapshot": "0.2.4",
            "component_search": {"initial_max": 8, "hard_max": 24, "adaptive": True},
            "shell_ranks": [4, 8, 16, 32],
            "core_rank": 4,
            "connectivity_rank": 4,
        },
        "artifacts": {
            "manuscript_pdf": _record(pdf, repo),
            "figure_manifest": _record(figure_manifest_path, repo),
            "protocols": _records(protocols, repo),
            "evidence_sources_and_audits": _records(evidence, repo),
            "figure_source_support": _records(source_support, repo),
            "figure_outputs": _records([*figures, *ancillary], repo),
            "build_inputs": _records(inputs, repo),
            "discovery_threshold_sensitivity": _records(sensitivity, repo),
            "results_section_outputs": _records(results_outputs, repo),
            "results_section_audits": _records(results_audits, repo),
        },
        "validation": {
            "figure_placeholders": 0,
            "figure_sources_verified": len(sources),
            "protocols_hashed": len(protocols),
            "evidence_files_hashed": len(evidence),
            "figure_outputs_hashed": len(figures) + len(ancillary),
            "sensitivity_outputs_hashed": len(sensitivity),
            "results_section_outputs_hashed": len(results_outputs),
            "results_section_audits_hashed": len(results_audits),
        },
        "historical_artifacts": {
            "canonical": False,
            "policy": (
                "Historical pilot tables, superseded figures, and earlier protocol generations "
                "may remain in the repository for lineage. They are noncanonical unless listed "
                "above as a provenance source for the exact Gaia single-target rescore."
            ),
        },
    }


def _serialized(manifest: Mapping[str, object]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_or_check(output: Path, manifest: Mapping[str, object], *, check: bool) -> None:
    desired = _serialized(manifest)
    if check:
        if not output.is_file() or output.read_text(encoding="utf-8") != desired:
            raise ManifestError(f"artifact manifest is missing or stale: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(desired, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or verify the canonical OEDM artifact manifest"
    )
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--figure-manifest", type=Path, default=DEFAULT_FIGURE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(
        repo=args.repo,
        pdf=args.pdf,
        figure_manifest_path=args.figure_manifest,
        results_section_outputs=RESULTS_SECTION_OUTPUTS,
        results_section_audits=RESULTS_SECTION_AUDITS,
    )
    write_or_check(args.output, manifest, check=args.check)
    print(f"{'verified' if args.check else 'wrote'} {args.output}")


if __name__ == "__main__":
    main()
