from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.pilot.json")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True); temporary = Path(handle.name)
    os.replace(temporary, path)


def build_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    design = config["design"]
    parameters = {"mass_per_target": int(design["mass_per_target"]), "minimum_local_contrast_at_99pct": float(design["minimum_local_contrast_at_99pct"])}
    signal = int(design["target_count"]) * int(design["mass_per_target"])
    jobs = []
    for cell in config["cells"]:
        for fraction in cell["noise_fractions"]:
            n = signal + int(round(signal * float(fraction) / (1.0 - float(fraction))))
            for seed in config["sampling_seeds"]:
                for method in config["methods"]:
                    slug = method.lower().replace("-", "_").replace("+", "plus")
                    jobs.append({"job_id": f"{cell['cell_id']}__noise{int(round(1000 * fraction)):04d}__n{n}__seed{seed}__{slug}", "cell_id": cell["cell_id"], "topology": cell["topology"], "density_ratio": float(cell["density_ratio"]), "noise_fraction": float(fraction), "expected_n": n, "geometry_seed": int(config["geometry_seed"]), "sampling_seed": int(seed), "method": method, "design_parameters": parameters})
    return jobs


def run_one(job: dict[str, Any], output: Path, timeout: int) -> dict[str, Any]:
    job_path = output / "jobs" / f"{job['job_id']}.json"; result_path = output / "job_results" / f"{job['job_id']}.json"
    atomic_json(job_path, job)
    env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"): env[variable] = "1"
    try:
        subprocess.run([sys.executable, "-m", "topology_noise_benchmark.worker", "--job", str(job_path), "--output", str(result_path)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        payload = {"schema_version": 1, "pilot_label": "PILOT / exploratory; not final evidence", "job_id": job["job_id"], "status": "timeout", **{key: job[key] for key in ("method", "cell_id", "topology", "density_ratio", "sampling_seed", "noise_fraction")}, "n": job["expected_n"], "timeout_seconds": timeout}
        atomic_json(result_path, payload); return payload
    if result_path.exists(): return json.loads(result_path.read_text(encoding="utf-8"))
    payload = {"schema_version": 1, "job_id": job["job_id"], "status": "worker_failure", **{key: job[key] for key in ("method", "cell_id", "topology", "density_ratio", "sampling_seed", "noise_fraction")}, "n": job["expected_n"]}
    atomic_json(result_path, payload); return payload


def flatten(result: dict[str, Any]) -> dict[str, Any]:
    fields = ("job_id", "status", "method", "cell_id", "topology", "density_ratio", "sampling_seed", "noise_fraction", "n", "runtime_seconds", "peak_rss_mb", "incremental_rss_mb", "skip_reason", "error_type", "error", "timeout_seconds")
    row = {key: result.get(key) for key in fields}
    for key in ("hungarian_macro_target_f1", "pairwise_f1", "noise_f1", "predicted_clusters", "truth_clusters", "predicted_true_cluster_ratio", "extra_signal_fragments", "merged_predicted_clusters", "false_discovery_signal_fraction"):
        row[key] = result.get("metrics", {}).get(key)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--output-root", type=Path, default=ROOT / "topology_noise_results"); parser.add_argument("--run-id", default=None); args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); output = args.output_root / (args.run_id or f"topology_noise_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True); jobs = build_jobs(config)
    atomic_json(output / "resolved_config.json", config); atomic_json(output / "manifest.json", {"pilot_label": "PILOT / exploratory; not final evidence", "jobs": jobs})
    results, runnable = [], []
    for job in jobs:
        if job["method"] == "AMD-DBSCAN":
            skipped = {"schema_version": 1, "pilot_label": "PILOT / exploratory; not final evidence", "job_id": job["job_id"], "status": "skipped_resource_limit", **{key: job[key] for key in ("method", "cell_id", "topology", "density_ratio", "sampling_seed", "noise_fraction")}, "n": job["expected_n"], "skip_reason": config["resource_policy"]["reason"]}
            atomic_json(output / "job_results" / f"{job['job_id']}.json", skipped); results.append(skipped)
        else: runnable.append(job)
    with ThreadPoolExecutor(max_workers=int(config["max_workers"])) as executor:
        futures = {executor.submit(run_one, job, output, int(config["timeout_seconds"])): job for job in runnable}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result(); results.append(result); print(f"[{index}/{len(runnable)}] {result['job_id']}: {result['status']}", flush=True)
    order = {job["job_id"]: index for index, job in enumerate(jobs)}; results.sort(key=lambda result: order[result["job_id"]])
    atomic_json(output / "results.json", results)
    rows = [flatten(result) for result in results]
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    status: dict[str, int] = {}
    for result in results: status[result["status"]] = status.get(result["status"], 0) + 1
    atomic_json(output / "completion.json", {"pilot_label": "PILOT / exploratory; not final evidence", "total_jobs": len(results), "status_counts": status})
    print(f"Results: {output}")


if __name__ == "__main__": main()
