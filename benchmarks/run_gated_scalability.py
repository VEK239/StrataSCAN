"""Run a synthetic scalability matrix in ascending-size, failure-gated order.

For each (dataset family, method, seed) ladder, a non-``ok`` result at one
size records explicit ``skipped_prior_failure`` outcomes at larger sizes.  The
other family/method ladders continue, so one unscalable baseline cannot block
the whole benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks import run_benchmark as benchmark


def _identity(job: dict) -> tuple[str, str, int]:
    return job["dataset"]["id"], job["method"], int(job["seed"])


def _skip_result(job: dict, blocker: dict) -> dict:
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "status": "skipped_prior_failure",
        "suite": job["suite"],
        "dataset_id": job["dataset_id"],
        "method": job["method"],
        "seed": job["seed"],
        "error": (
            "not launched because the same family/method ladder failed at "
            f"n={blocker.get('n')}: status={blocker.get('status')}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run size-ascending gated synthetic scaling")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = benchmark.read_protocol(protocol_path)
    benchmark.require_runtime_dependencies(protocol, "synthetic")
    jobs = benchmark.expand_jobs(protocol, "synthetic")
    if any(job["dataset"]["tier"] != "scaling" for job in jobs):
        raise ValueError("gated runner accepts a scaling-only protocol")

    output = args.output_dir.resolve()
    jobs_dir = output / "jobs"
    specs_dir = output / "specs"
    output.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(exist_ok=True)
    specs_dir.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if [job["job_id"] for job in manifest["jobs"]] != [job["job_id"] for job in jobs]:
            raise ValueError("existing manifest does not match the protocol matrix")
    else:
        manifest = benchmark.create_manifest(
            protocol, protocol_path, "synthetic", jobs,
            expanded_job_count=len(jobs), max_workers=1,
        )
        manifest["execution"]["mode"] = "serial_size_ascending_failure_gated"
        benchmark.atomic_json(manifest_path, manifest)

    if not args.resume and any(jobs_dir.glob("*.json")):
        raise FileExistsError("results already exist; use --resume")
    results: dict[str, dict] = {}
    for result_path in jobs_dir.glob("*.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        results[result["job_id"]] = result

    ordered = sorted(
        jobs,
        key=lambda job: (int(job["dataset"]["n"]), job["dataset"]["id"], job["method"], int(job["seed"])),
    )
    prior: dict[tuple[str, str, int], dict] = {}
    for index, job in enumerate(ordered, start=1):
        key = _identity(job)
        result_path = jobs_dir / f"{job['job_id']}.json"
        if job["job_id"] in results:
            result = results[job["job_id"]]
            print(f"[{index}/{len(ordered)}] resume {job['job_id']}", flush=True)
        elif key in prior and prior[key].get("status") != "ok":
            result = _skip_result(job, prior[key])
            benchmark.atomic_json(result_path, result)
            print(f"[{index}/{len(ordered)}] {result['status']} {job['job_id']}", flush=True)
        else:
            spec_path = specs_dir / f"{job['job_id']}.json"
            benchmark.atomic_json(spec_path, job)
            print(f"[{index}/{len(ordered)}] run {job['job_id']}", flush=True)
            result = benchmark.run_job(job, spec_path, result_path, manifest["resources"])
            print(f"[{index}/{len(ordered)}] {result['status']} {job['job_id']}", flush=True)
        results[job["job_id"]] = result
        prior[key] = result

    validation, rows = benchmark.validate(manifest, jobs_dir)
    benchmark.atomic_json(output / "validation.json", validation)
    if validation["complete"]:
        benchmark.write_csv(output / "results.csv", rows)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
