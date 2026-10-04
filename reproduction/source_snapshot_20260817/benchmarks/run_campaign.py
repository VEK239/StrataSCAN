from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from benchmarks.run_benchmark import REPO, expand_jobs, read_protocol


DEFAULT_CAMPAIGN = Path(__file__).with_name("campaign.v0.2.3-release.json")


def read_campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "campaign_version", "phases"}
    missing = required - campaign.keys()
    if missing:
        raise ValueError(f"campaign is missing keys: {sorted(missing)}")
    names = [phase["name"] for phase in campaign["phases"]]
    if len(names) != len(set(names)):
        raise ValueError("campaign contains duplicate phase names")
    for phase in campaign["phases"]:
        missing_phase = {"name", "protocol"} - phase.keys()
        if missing_phase:
            raise ValueError(
                f"campaign phase is missing keys: {sorted(missing_phase)}"
            )
        if int(phase.get("max_workers", 1)) < 1:
            raise ValueError("campaign phase max_workers must be at least 1")
    return campaign


def selected_phases(
    campaign: dict[str, Any], requested: set[str]
) -> list[dict[str, Any]]:
    phases = campaign["phases"]
    if not requested:
        return phases
    available = {phase["name"] for phase in phases}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"unknown campaign phases: {unknown}")
    return [phase for phase in phases if phase["name"] in requested]


def phase_command(
    phase: dict[str, Any],
    *,
    campaign_path: Path,
    output_root: Path,
    resume: bool,
    validate_only: bool,
    worker_override: int | None,
) -> list[str]:
    protocol = (campaign_path.parent / phase["protocol"]).resolve()
    output = (output_root / phase.get("output", phase["name"])).resolve()
    workers = int(worker_override or phase.get("max_workers", 1))
    command = [
        sys.executable,
        "-m",
        "benchmarks.run_benchmark",
        "--protocol",
        str(protocol),
        "--output-dir",
        str(output),
        "--suite",
        phase.get("suite", "all"),
        "--max-workers",
        str(workers),
    ]
    if resume:
        command.append("--resume")
    if validate_only:
        command.append("--validate-only")
    if phase.get("compare_to"):
        command.extend(
            ["--compare-to", str((REPO / phase["compare_to"]).resolve())]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen multi-protocol StrataSCAN benchmark campaign"
    )
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    campaign_path = args.campaign.resolve()
    campaign = read_campaign(campaign_path)
    phases = selected_phases(campaign, set(args.phase))
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (REPO / "results" / "runs" / campaign["campaign_version"]).resolve()
    )
    if args.list:
        listing = []
        for phase in phases:
            protocol_path = (campaign_path.parent / phase["protocol"]).resolve()
            protocol = read_protocol(protocol_path)
            listing.append({
                "name": phase["name"],
                "protocol": str(protocol_path.relative_to(REPO)),
                "suite": phase.get("suite", "all"),
                "jobs": len(expand_jobs(protocol, phase.get("suite", "all"))),
                "max_workers": int(args.max_workers or phase.get("max_workers", 1)),
                "output": str(output_root / phase.get("output", phase["name"])),
            })
        print(json.dumps({"campaign": campaign["campaign_version"], "phases": listing}, indent=2))
        return

    for phase in phases:
        command = phase_command(
            phase,
            campaign_path=campaign_path,
            output_root=output_root,
            resume=args.resume,
            validate_only=args.validate_only,
            worker_override=args.max_workers,
        )
        print(f"[campaign] {phase['name']}", flush=True)
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
