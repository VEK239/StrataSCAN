from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "manuscript/oedm2026/scripts/generate_artifact_manifest.py"
SPEC = importlib.util.spec_from_file_location("generate_artifact_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
artifact = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = artifact
SPEC.loader.exec_module(artifact)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    evidence = _write(tmp_path / "results/final/results.csv", "metric\n1\n")
    protocol = _write(tmp_path / "benchmarks/protocol.final.json", "{}\n")
    support = _write(tmp_path / "scripts/figure_builder.py", "# builder\n")
    build_input = _write(tmp_path / "manuscript/build_input.tex", "source\n")
    figure = _write(tmp_path / "manuscript/final_figures/figure.pdf", "figure\n")
    ancillary = _write(tmp_path / "manuscript/final_figures/figure-data.csv", "x\n1\n")
    pdf = _write(tmp_path / "manuscript/paper.pdf", "pdf\n")
    _write(tmp_path / "manuscript/oedm2026/tables/discovery_threshold_sensitivity.csv", "evidence_block,method\nsynthetic,StrataSCAN\n")
    _write(tmp_path / "manuscript/oedm2026/tables/discovery_threshold_sensitivity.md", "# sensitivity\n")
    sources = [evidence, protocol, support]
    figure_manifest = tmp_path / "manuscript/final_figures/figure_manifest.json"
    figure_manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "path": path.relative_to(tmp_path).as_posix(),
                        "exists": True,
                        "sha256": artifact.sha256(path),
                    }
                    for path in sources
                ],
                "figures": [figure.name],
                "ancillary_files": [ancillary.name],
                "placeholders": [],
            }
        ),
        encoding="utf-8",
    )
    return pdf, figure_manifest, (build_input.relative_to(tmp_path).as_posix(),)


def test_manifest_hashes_current_pdf_evidence_protocols_figures_and_inputs(
    tmp_path: Path,
) -> None:
    pdf, figure_manifest, build_inputs = _fixture(tmp_path)
    result = artifact.build_manifest(
        repo=tmp_path,
        pdf=pdf,
        figure_manifest_path=figure_manifest,
        build_inputs=build_inputs,
    )

    artifacts = result["artifacts"]
    assert artifacts["manuscript_pdf"]["sha256"] == artifact.sha256(pdf)
    assert [row["path"] for row in artifacts["protocols"]] == [
        "benchmarks/protocol.final.json"
    ]
    assert [row["path"] for row in artifacts["evidence_sources_and_audits"]] == [
        "results/final/results.csv"
    ]
    assert {row["path"] for row in artifacts["figure_outputs"]} == {
        "manuscript/final_figures/figure.pdf",
        "manuscript/final_figures/figure-data.csv",
    }
    assert result["historical_artifacts"]["canonical"] is False


def test_manifest_rejects_stale_source_hash_and_placeholders(tmp_path: Path) -> None:
    pdf, figure_manifest, build_inputs = _fixture(tmp_path)
    evidence = tmp_path / "results/final/results.csv"
    evidence.write_text("changed\n", encoding="utf-8")
    with pytest.raises(artifact.ManifestError, match="source hash is stale"):
        artifact.build_manifest(
            repo=tmp_path,
            pdf=pdf,
            figure_manifest_path=figure_manifest,
            build_inputs=build_inputs,
        )

    payload = json.loads(figure_manifest.read_text(encoding="utf-8"))
    payload["placeholders"] = ["missing panel"]
    figure_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(artifact.ManifestError, match="contains placeholders"):
        artifact.build_manifest(
            repo=tmp_path,
            pdf=pdf,
            figure_manifest_path=figure_manifest,
            build_inputs=build_inputs,
        )


def test_check_mode_detects_stale_manifest(tmp_path: Path) -> None:
    pdf, figure_manifest, build_inputs = _fixture(tmp_path)
    manifest = artifact.build_manifest(
        repo=tmp_path,
        pdf=pdf,
        figure_manifest_path=figure_manifest,
        build_inputs=build_inputs,
    )
    output = tmp_path / "artifact_manifest.json"
    artifact.write_or_check(output, manifest, check=False)
    artifact.write_or_check(output, manifest, check=True)
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(artifact.ManifestError, match="missing or stale"):
        artifact.write_or_check(output, manifest, check=True)


def test_build_wires_final_pipeline_in_fail_closed_order() -> None:
    build = (REPO / "manuscript/oedm2026/scripts/build.ps1").read_text(encoding="utf-8")
    statistics = build.index("final_manuscript_statistics.py")
    sensitivity = build.index("discovery_threshold_sensitivity.py")
    synchronize = build.index("synchronize_final_manuscript_statistics.py")
    figures = build.index("final_manuscript_figures.py")
    compile_pdf = build.index("tectonic -X compile")
    manifest = build.index("generate_artifact_manifest.py")

    assert statistics < sensitivity < synchronize < figures < compile_pdf < manifest
    assert "Invoke-CheckedPython" in build
    assert "--check" in build
    assert "--allow-placeholders" not in build
    assert "analyze_results.py" not in build
    assert "revision_results.py" not in build
