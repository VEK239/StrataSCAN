from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.run_benchmark import input_drift, select_jobs
from benchmarks.run_campaign import phase_command, read_campaign, selected_phases


def test_select_jobs_requires_exact_known_ids() -> None:
    jobs = [{"job_id": "a"}, {"job_id": "b"}]
    assert select_jobs(jobs, {"b"}) == [{"job_id": "b"}]
    with pytest.raises(ValueError, match="unknown benchmark job IDs"):
        select_jobs(jobs, {"missing"})


def test_input_drift_detects_changed_added_and_removed_paths() -> None:
    recorded = [
        {"path": "a.py", "bytes": 1, "sha256": "one"},
        {"path": "removed.py", "bytes": 2, "sha256": "two"},
    ]
    current = [
        {"path": "a.py", "bytes": 3, "sha256": "three"},
        {"path": "added.py", "bytes": 2, "sha256": "two"},
    ]
    assert input_drift(recorded, current) == ["a.py", "added.py", "removed.py"]


def test_campaign_selection_and_command_are_deterministic(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        '{"schema_version":1,"campaign_version":"test","phases":'
        '[{"name":"quality","protocol":"quality.json","max_workers":1}]}',
        encoding="utf-8",
    )
    campaign = read_campaign(campaign_path)
    phase = selected_phases(campaign, {"quality"})[0]
    command = phase_command(
        phase,
        campaign_path=campaign_path,
        output_root=tmp_path / "runs",
        resume=True,
        validate_only=False,
        worker_override=4,
    )
    assert command[-3:] == ["--max-workers", "4", "--resume"]
    assert "benchmarks.run_benchmark" in command
    with pytest.raises(ValueError, match="unknown campaign phases"):
        selected_phases(campaign, {"missing"})
