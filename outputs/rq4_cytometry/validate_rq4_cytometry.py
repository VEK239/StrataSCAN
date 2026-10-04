from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent


def main() -> None:
    audit = json.loads((HERE / "rq4_audit.json").read_text(encoding="utf-8"))
    assert all(audit["checks"].values())
    assert audit["rows"] == 117
    assert not any("gaia" in item.lower() for item in audit["datasets"])

    with (HERE / "rq4_method_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = {row["method"]: row for row in csv.DictReader(handle)}
    assert len(summary) == 9
    strata = summary["StrataSCAN"]
    assert int(strata["completed"]) == 13
    assert math.isclose(float(strata["median_target_f1"]), 0.2033431136404888, abs_tol=1e-15)
    assert int(summary["AMD-DBSCAN"]["completed"]) == 0
    assert math.isnan(float(summary["AMD-DBSCAN"]["median_target_f1"]))

    with (HERE / "rq4_by_dataset.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 117
    for row in rows:
        if row["status"] != "ok":
            assert row["macro_target_f1"] == ""
            assert row["macro_target_purity"] == ""
            assert row["macro_target_coverage"] == ""
            assert row["target_discovery_rate"] == ""

    page = PdfReader(HERE / "fig_rq4_cytometry.pdf").pages[0]
    width_inches = float(page.mediabox.width) / 72.0
    height_inches = float(page.mediabox.height) / 72.0
    assert width_inches <= 3.5
    assert height_inches <= 2.5
    assert (HERE / "fig_rq4_cytometry.svg").stat().st_size > 10_000
    assert (HERE / "fig_rq4_cytometry.png").stat().st_size > 10_000

    print(f"RQ4 validation passed: {width_inches:.3f} x {height_inches:.3f} in; 117 cells; failures are NA.")


if __name__ == "__main__":
    main()
