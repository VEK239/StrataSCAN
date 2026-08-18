from __future__ import annotations

import json
from pathlib import Path

import re


ROOT = Path(__file__).resolve().parents[1]


def test_revised_result_tables_are_amd_free_and_use_requested_columns() -> None:
    table_i = (ROOT / "manuscript/oedm2026/tables/table_rq1_synthetic_fresh.tex").read_text(encoding="utf-8")
    table_ii = (ROOT / "manuscript/oedm2026/tables/table_rq2_noise99.tex").read_text(encoding="utf-8")
    table_iii = (ROOT / "manuscript/oedm2026/tables/table_rq3_scaling.tex").read_text(encoding="utf-8")
    assert "AMD" not in table_i + table_ii + table_iii
    assert "Target F1 & Discovery" in table_i
    assert "\\pm" in table_i
    assert "Done" not in table_i
    assert "Median" not in table_i
    assert "Method & Target F1 & Background F1" in table_ii
    assert "Done" not in table_ii


def test_revised_figures_fit_ieee_widths() -> None:
    for name in ("fig2_density_noise.pdf", "fig4_cytometry.pdf"):
        payload = (ROOT / "manuscript/oedm2026/final_figures" / name).read_bytes()
        match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", payload)
        assert match is not None
        width = float(match.group(1)) / 72
        height = float(match.group(2)) / 72
        assert width <= 7.2
        assert height <= 3.0


def test_merge_aware_metric_does_not_replace_primary_endpoint() -> None:
    audit = json.loads((ROOT / "outputs/rq4_cytometry/merge_aware_sensitivity.json").read_text(encoding="utf-8"))
    assert audit["stratascan_merge_aware_median"] > audit["stratascan_one_to_one_median"]
    assert audit["stratascan_merge_aware_median"] < 0.3
    assert "retain" in audit["decision"]
