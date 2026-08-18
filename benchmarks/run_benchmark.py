from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import statistics
import sys
import tempfile
import time
from typing import Any

import psutil
import numpy as np

from benchmarks.datasets import gaia_field_ids


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.gamma-strict-core.json")
DEFAULT_EVALUATION_PROTOCOL = Path(__file__).with_name("evaluation_protocol.v1.json")


def read_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "protocol_version", "parameter_mode", "resources", "methods"}
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"protocol is missing keys: {sorted(missing)}")
    if len(protocol["methods"]) != len(set(protocol["methods"])):
        raise ValueError("protocol contains duplicate methods")
    return protocol


def read_evaluation_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "protocol_version", "matching", "aggregation",
        "discovery_thresholds", "fragment_min_target_fraction", "failure_policy", "suites",
    }
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"evaluation protocol is missing keys: {sorted(missing)}")
    if protocol["schema_version"] != 1:
        raise ValueError("unsupported evaluation protocol schema_version")
    matching = protocol["matching"]
    if matching.get("strategy") != "hungarian" or matching.get("objective") != "pairwise_f1":
        raise ValueError("evaluation matching must be Hungarian maximum pairwise F1")
    if float(matching.get("unmatched_target_score", float("nan"))) != 0.0:
        raise ValueError("unmatched targets must receive zero")
    thresholds = protocol["discovery_thresholds"]
    for key in ("purity", "coverage"):
        value = thresholds.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"evaluation discovery threshold {key} must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"evaluation discovery threshold {key} must lie in [0, 1]")
    fraction = protocol["fragment_min_target_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError("fragment_min_target_fraction must be numeric")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("fragment_min_target_fraction must lie in [0, 1]")
    if set(protocol["suites"]) != {"synthetic", "cytometry", "gaia"}:
        raise ValueError("evaluation protocol must define synthetic, cytometry, and gaia suites")
    return protocol


def evaluation_for_job(
    protocol: dict[str, Any], suite: str, dataset_id: str
) -> dict[str, Any]:
    suite_config = dict(protocol["suites"][suite])
    overrides = suite_config.pop("dataset_overrides", {})
    override = overrides.get(dataset_id, {})
    resolved = {
        "protocol_version": protocol["protocol_version"],
        "matching": {
            "strategy": protocol["matching"]["strategy"],
            "objective": protocol["matching"]["objective"],
        },
        "discovery_thresholds": dict(protocol["discovery_thresholds"]),
        "fragment_min_target_fraction": protocol["fragment_min_target_fraction"],
        **suite_config,
        **override,
    }
    return resolved


def require_runtime_dependencies(protocol: dict[str, Any], suite: str) -> None:
    modules = {"psutil", "threadpoolctl"}
    if suite in {"all", "cytometry", "gaia"}:
        modules.update({"pandas", "faiss"})
    if suite in {"all", "synthetic"} and any(
        int(case["dimension"]) > 2 for case in protocol["synthetic"]["cases"]
    ):
        modules.add("faiss")
    if "SNN-DBSCAN" in protocol["methods"]:
        modules.add("numba")
    if "kNN+Leiden" in protocol["methods"]:
        modules.update({"igraph", "leidenalg"})
    missing = sorted(module for module in modules if importlib.util.find_spec(module) is None)
    if missing:
        raise RuntimeError(
            "missing benchmark dependencies: " + ", ".join(missing)
            + '; install them with: python -m pip install -e ".[benchmark,perf]"'
        )


def _job_id(job: dict[str, Any]) -> str:
    encoded = json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    readable = "__".join(
        str(job[key]).replace("/", "-").replace(" ", "_")
        for key in ("suite", "dataset_id", "method", "seed")
    )
    return f"{readable}__{digest}"


def _seeds(protocol: dict[str, Any], method: str, suite_seeds: list[int]) -> list[int]:
    if method in set(protocol["stochastic_methods"]):
        return [int(seed) for seed in suite_seeds]
    return [42 if 42 in suite_seeds else int(suite_seeds[0])]


def select_gaia_fields(fields: list[str], config: dict[str, Any]) -> list[str]:
    selection = config.get("field_selection")
    if selection is None:
        return fields
    if selection.get("strategy") != "stable_hash":
        raise ValueError("Gaia field_selection.strategy must be 'stable_hash'")
    count = int(selection["count"])
    if count < 1 or count > len(fields):
        raise ValueError(f"Gaia field selection count must be between 1 and {len(fields)}")
    salt = str(selection["salt"])
    ranked = sorted(
        fields,
        key=lambda field: hashlib.sha256(f"{salt}:{field}".encode()).hexdigest(),
    )
    return sorted(ranked[:count])


def expand_jobs(
    protocol: dict[str, Any],
    suite: str = "all",
    evaluation_protocol: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    methods = protocol["methods"]
    if suite in {"all", "synthetic"}:
        synthetic = protocol["synthetic"]
        size_mode = synthetic.get("size_mode", "total_rows")
        if size_mode not in {"total_rows", "fixed_signal_count"}:
            raise ValueError(
                "synthetic.size_mode must be 'total_rows' or 'fixed_signal_count'"
            )
        tiers = (
            ("quality", synthetic["quality_sizes"], synthetic["quality_seeds"]),
            ("scaling", synthetic["scaling_sizes"], synthetic["scaling_seeds"]),
        )
        for case in synthetic["cases"]:
            for tier, sizes, seeds in tiers:
                for n in sizes:
                    base_size = int(n)
                    if size_mode == "fixed_signal_count":
                        fraction = float(case["noise_fraction"])
                        if not 0.0 < fraction < 1.0:
                            raise ValueError(
                                "fixed_signal_count requires noise_fraction in (0, 1)"
                            )
                        background_size = int(round(base_size * fraction / (1.0 - fraction)))
                        total_size = base_size + background_size
                        dataset = {
                            **case,
                            "n": total_size,
                            "signal_size": base_size,
                            "tier": tier,
                        }
                        dataset_id = (
                            f"{case['id']}__{tier}__signal{base_size}__n{total_size}"
                        )
                    else:
                        dataset = {**case, "n": base_size, "tier": tier}
                        dataset_id = f"{case['id']}__{tier}__n{base_size}"
                    for method in methods:
                        for seed in seeds:
                            job = {
                                "schema_version": 1,
                                "protocol_version": protocol["protocol_version"],
                                "suite": "synthetic",
                                "dataset_id": dataset_id,
                                "dataset": dataset,
                                "method": method,
                                "seed": int(seed),
                            }
                            if evaluation_protocol is not None:
                                job["evaluation"] = evaluation_for_job(
                                    evaluation_protocol, "synthetic", dataset_id
                                )
                            job["job_id"] = _job_id(job)
                            jobs.append(job)
    if suite in {"all", "cytometry"}:
        cytometry = protocol["cytometry"]
        for dataset in cytometry["datasets"]:
            for method in methods:
                for seed in _seeds(protocol, method, cytometry["seeds"]):
                    job = {
                        "schema_version": 1,
                        "protocol_version": protocol["protocol_version"],
                        "suite": "cytometry",
                        "dataset_id": dataset["id"],
                        "dataset": dataset,
                        "method": method,
                        "seed": int(seed),
                    }
                    if evaluation_protocol is not None:
                        job["evaluation"] = evaluation_for_job(
                            evaluation_protocol, "cytometry", dataset["id"]
                        )
                    job["job_id"] = _job_id(job)
                    jobs.append(job)
    if suite in {"all", "gaia"}:
        gaia = protocol["gaia"]
        data_root = REPO / gaia["data_root"]
        fields = select_gaia_fields(gaia_field_ids(data_root), gaia)
        if not fields:
            raise ValueError("no Gaia fields match the reference tables")
        for profile in gaia["profiles"]:
            for field_id in fields:
                dataset_id = f"{field_id}__{profile}"
                dataset = {
                    "id": field_id,
                    "field_id": field_id,
                    "preprocessing": profile,
                    "data_root": gaia["data_root"],
                    "ruwe_max": gaia["ruwe_max"],
                }
                for method in methods:
                    for seed in _seeds(protocol, method, gaia["seeds"]):
                        job = {
                            "schema_version": 1,
                            "protocol_version": protocol["protocol_version"],
                            "suite": "gaia",
                            "dataset_id": dataset_id,
                            "dataset": dataset,
                            "method": method,
                            "seed": int(seed),
                        }
                        if evaluation_protocol is not None:
                            job["evaluation"] = evaluation_for_job(
                                evaluation_protocol, "gaia", dataset_id
                            )
                        job["job_id"] = _job_id(job)
                        jobs.append(job)
    ids = [job["job_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded benchmark matrix contains duplicate job IDs")
    return jobs


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            # Windows virus scanners and indexers can briefly hold a newly
            # closed file. Keep the atomic replace, but tolerate that race.
            time.sleep(0.05 * (2**attempt))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def input_paths(
    protocol: dict[str, Any],
    protocol_path: Path,
    suite: str,
    evaluation_protocol_path: Path | None = None,
) -> list[Path]:
    paths = [
        protocol_path.resolve(),
        *sorted((REPO / "benchmarks").glob("*.py")),
        *sorted((REPO / "baselines").glob("*.py")),
        *sorted((REPO / "src" / "stratascan").glob("*.py")),
    ]
    if evaluation_protocol_path is not None:
        paths.append(evaluation_protocol_path.resolve())
    if suite in {"all", "synthetic"}:
        paths.extend([
            REPO / "benchmarks" / "datasets.py",
            REPO / "src" / "stratascan" / "synthetic.py",
        ])
    if suite in {"all", "cytometry"}:
        for dataset in protocol["cytometry"]["datasets"]:
            paths.extend([
                REPO / dataset["features"],
                REPO / dataset["feature_metadata"],
                REPO / dataset["metadata"],
            ])
    if suite in {"all", "gaia"}:
        root = REPO / protocol["gaia"]["data_root"]
        paths.extend([
            root / "benchmark_reference_tables" / "ocfinder_table1.csv",
            root / "benchmark_reference_tables" / "ocfinder_table2.csv",
        ])
        fields = root / "raw_open_cluster_fields" / "gaia_dr3_cone_fields"
        selected = select_gaia_fields(gaia_field_ids(root), protocol["gaia"])
        paths.extend(fields / f"gaia_cone_{field}.csv" for field in selected)
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"benchmark inputs are missing: {missing}")
    return paths


def input_checksums(
    protocol: dict[str, Any],
    protocol_path: Path,
    suite: str,
    evaluation_protocol_path: Path | None = None,
) -> list[dict[str, Any]]:
    records = []
    for path in input_paths(protocol, protocol_path, suite, evaluation_protocol_path):
        records.append({
            "path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return records


def input_drift(
    recorded: list[dict[str, Any]], current: list[dict[str, Any]]
) -> list[str]:
    """Return paths added, removed, or changed since a manifest was frozen."""

    expected = {item["path"]: (item["bytes"], item["sha256"]) for item in recorded}
    observed = {item["path"]: (item["bytes"], item["sha256"]) for item in current}
    return sorted(
        path
        for path in expected.keys() | observed.keys()
        if expected.get(path) != observed.get(path)
    )


def select_jobs(
    jobs: list[dict[str, Any]], requested_ids: set[str]
) -> list[dict[str, Any]]:
    if not requested_ids:
        return jobs
    available = {job["job_id"] for job in jobs}
    unknown = sorted(requested_ids - available)
    if unknown:
        raise ValueError(f"unknown benchmark job IDs: {unknown}")
    return [job for job in jobs if job["job_id"] in requested_ids]


def environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy", "scipy", "scikit-learn", "pandas", "psutil", "threadpoolctl",
        "faiss-cpu", "numba", "igraph", "leidenalg",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = None
    dirty = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True))
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_mb": psutil.virtual_memory().total / 2**20,
        "git_commit": commit,
        "git_dirty": dirty,
        "packages": packages,
    }


def create_manifest(
    protocol: dict[str, Any],
    protocol_path: Path,
    suite: str,
    jobs: list[dict[str, Any]],
    *,
    expanded_job_count: int,
    max_workers: int,
    evaluation_protocol: dict[str, Any] | None = None,
    evaluation_protocol_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        "purpose": protocol.get("purpose", "unspecified"),
        "parameter_mode": protocol["parameter_mode"],
        "suite": suite,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resources": protocol["resources"],
        "execution": {
            "max_workers": int(max_workers),
            "parallel_resource_measurements": bool(max_workers > 1),
        },
        "methods": protocol["methods"],
        "comparison": protocol.get("comparison"),
        "environment": environment(),
        "inputs": input_checksums(
            protocol, protocol_path, suite, evaluation_protocol_path
        ),
        "evaluation": evaluation_protocol,
        "evaluation_artifacts": {
            "required_for_status": "ok",
            "path_template": "evaluation_state/{job_id}.npz",
            "format": "numpy_npz_compressed",
            "lossless_arrays": ["y_true", "y_pred"],
            "audit_arrays": [
                "target_labels", "predicted_labels", "target_sizes", "predicted_sizes",
                "target_predicted_contingency",
            ],
        } if evaluation_protocol is not None else None,
        "expanded_job_count": int(expanded_job_count),
        "expected_job_count": len(jobs),
        "jobs": jobs,
    }


def kill_tree(process: subprocess.Popen[str]) -> None:
    try:
        root = psutil.Process(process.pid)
        for child in root.children(recursive=True):
            child.kill()
        root.kill()
    except psutil.Error:
        return


def run_job(
    job: dict[str, Any],
    spec_path: Path,
    result_path: Path,
    resources: dict[str, Any],
    evaluation_state_path: Path | None = None,
    evaluation_state_reference: str | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT",
    ):
        env[variable] = str(resources["threads"])
    command = [
        sys.executable,
        "-m",
        "benchmarks.worker",
        "--job", str(spec_path),
        "--output", str(result_path),
        "--repo", str(REPO),
    ]
    if evaluation_state_path is not None:
        if not evaluation_state_reference:
            raise ValueError("evaluation_state_reference is required with evaluation_state_path")
        command.extend([
            "--evaluation-state-output", str(evaluation_state_path),
            "--evaluation-state-reference", evaluation_state_reference,
        ])
    result_path.unlink(missing_ok=True)
    if evaluation_state_path is not None:
        evaluation_state_path.unlink(missing_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    peak_mb = 0.0
    forced_status: str | None = None
    forced_error = ""
    while process.poll() is None:
        elapsed = time.perf_counter() - started
        try:
            root = psutil.Process(process.pid)
            rss = root.memory_info().rss
            for child in root.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    continue
            peak_mb = max(peak_mb, rss / 2**20)
        except psutil.Error:
            peak_mb = max(peak_mb, 0.0)
        if elapsed > float(resources["timeout_seconds"]):
            forced_status = "timeout"
            forced_error = f"timeout_seconds={resources['timeout_seconds']}"
            kill_tree(process)
            break
        if peak_mb > float(resources["memory_limit_mb"]):
            forced_status = "memory_limit"
            forced_error = f"peak_rss_mb={peak_mb:.3f}; limit_mb={resources['memory_limit_mb']}"
            kill_tree(process)
            break
        time.sleep(0.02)
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        kill_tree(process)
        stdout, stderr = process.communicate()
    process_seconds = time.perf_counter() - started

    if forced_status is not None:
        result = {
            "schema_version": 2,
            "job_id": job["job_id"],
            "status": forced_status,
            "suite": job["suite"],
            "dataset_id": job["dataset_id"],
            "method": job["method"],
            "seed": job["seed"],
            "error": forced_error,
            "traceback": stderr[-8000:],
        }
    elif result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {
            "schema_version": 2,
            "job_id": job["job_id"],
            "status": "error",
            "suite": job["suite"],
            "dataset_id": job["dataset_id"],
            "method": job["method"],
            "seed": job["seed"],
            "error": f"worker exited with code {process.returncode} without a result",
            "traceback": (stderr or stdout)[-8000:],
        }
    result["process_runtime_seconds"] = float(process_seconds)
    result["process_peak_rss_mb"] = float(peak_mb)
    atomic_json(result_path, result)
    return result


def validate(manifest: dict[str, Any], jobs_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = {job["job_id"]: job for job in manifest["jobs"]}
    results: list[dict[str, Any]] = []
    malformed: list[str] = []
    duplicate: list[str] = []
    seen: set[str] = set()
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            job_id = result["job_id"]
            if job_id in seen:
                duplicate.append(job_id)
            seen.add(job_id)
            if job_id in expected:
                results.append(result)
        except (OSError, ValueError, KeyError, TypeError):
            malformed.append(path.name)
    missing = sorted(set(expected) - seen)
    unexpected = sorted(seen - set(expected))
    statuses: dict[str, int] = {}
    invalid_identity: list[str] = []
    invalid_evaluation_artifacts: list[str] = []
    required_artifact_arrays = {
        "y_true", "y_pred", "target_labels", "predicted_labels", "target_sizes",
        "predicted_sizes", "target_predicted_contingency",
    }
    for result in results:
        statuses[result.get("status", "missing_status")] = statuses.get(result.get("status", "missing_status"), 0) + 1
        job = expected[result["job_id"]]
        identity = ("suite", "dataset_id", "method", "seed")
        if any(result.get(key) != job.get(key) for key in identity):
            invalid_identity.append(result["job_id"])
        if result.get("status") == "ok" and manifest.get("evaluation_artifacts"):
            artifact = result.get("evaluation_artifact")
            expected_reference = f"evaluation_state/{result['job_id']}.npz"
            try:
                if not isinstance(artifact, dict) or artifact.get("path") != expected_reference:
                    raise ValueError("invalid artifact reference")
                artifact_path = (jobs_dir.parent / expected_reference).resolve()
                state_root = (jobs_dir.parent / "evaluation_state").resolve()
                if artifact_path.parent != state_root or not artifact_path.is_file():
                    raise ValueError("artifact missing or outside evaluation_state")
                if artifact.get("bytes") != artifact_path.stat().st_size:
                    raise ValueError("artifact byte count mismatch")
                if artifact.get("sha256") != sha256(artifact_path):
                    raise ValueError("artifact checksum mismatch")
                with np.load(artifact_path, allow_pickle=False) as state:
                    if set(state.files) != required_artifact_arrays:
                        raise ValueError("artifact array schema mismatch")
                    y_true = state["y_true"]
                    y_pred = state["y_pred"]
                    table = state["target_predicted_contingency"]
                    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
                        raise ValueError("artifact label arrays do not align")
                    if y_true.size != int(result["n"]):
                        raise ValueError("artifact label count does not match result")
                    if table.shape != (
                        state["target_labels"].size, state["predicted_labels"].size
                    ):
                        raise ValueError("artifact contingency dimensions do not align")
                    if int(np.sum(table)) != int(np.sum((y_true >= 0) & (y_pred >= 0))):
                        raise ValueError("artifact contingency is inconsistent with labels")
            except (KeyError, OSError, TypeError, ValueError):
                invalid_evaluation_artifacts.append(result["job_id"])
    complete = not (
        missing or unexpected or duplicate or malformed or invalid_identity
        or invalid_evaluation_artifacts
    )
    report = {
        "schema_version": 1,
        "complete": complete,
        "expected": len(expected),
        "found": len(results),
        "statuses": statuses,
        "missing_job_ids": missing,
        "unexpected_job_ids": unexpected,
        "duplicate_job_ids": duplicate,
        "malformed_files": malformed,
        "invalid_identity_job_ids": invalid_identity,
        "invalid_evaluation_artifact_job_ids": invalid_evaluation_artifacts,
    }
    return report, results


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    scalar_keys = [
        "job_id", "status", "suite", "dataset_id", "method", "seed", "package_version",
        "n", "dimension", "load_seconds", "runtime_seconds", "process_runtime_seconds",
        "peak_rss_mb", "incremental_rss_mb", "process_peak_rss_mb", "error_type", "error",
        "evaluation_artifact_path", "evaluation_artifact_sha256", "evaluation_artifact_bytes",
    ]
    metric_keys = sorted({
        key
        for result in results
        for key, value in result.get("metrics", {}).items()
        if not isinstance(value, (dict, list))
    })
    fields = scalar_keys + metric_keys + ["dataset_metadata_json", "parameters_json", "profile_json", "target_matches_json", "traceback"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda row: row["job_id"]):
            metrics = result.get("metrics", {})
            row = {key: result.get(key) for key in scalar_keys}
            artifact = result.get("evaluation_artifact", {})
            row.update({
                "evaluation_artifact_path": artifact.get("path"),
                "evaluation_artifact_sha256": artifact.get("sha256"),
                "evaluation_artifact_bytes": artifact.get("bytes"),
            })
            row.update({key: metrics.get(key) for key in metric_keys})
            row.update({
                "dataset_metadata_json": json.dumps(result.get("dataset_metadata", {}), sort_keys=True),
                "parameters_json": json.dumps(result.get("parameters", {}), sort_keys=True),
                "profile_json": json.dumps(result.get("profile", {}), sort_keys=True),
                "target_matches_json": json.dumps(metrics.get("target_matches", []), sort_keys=True),
                "traceback": result.get("traceback", ""),
            })
            writer.writerow(row)


def _primary_metric(result: dict[str, Any]) -> float | None:
    metrics = result.get("metrics", {})
    key = {
        "synthetic": "macro_target_f1",
        "cytometry": "macro_target_f1",
        "gaia": "best_cluster_f1",
    }.get(result.get("suite"))
    value = metrics.get(key) if key else None
    return None if value is None else float(value)


def compare_with_reference(
    reference_path: Path,
    current_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if reference_path.is_dir():
        reference_path = reference_path / "results.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference results do not exist: {reference_path}")
    # Rich estimator diagnostics can legitimately exceed the csv module's
    # conservative 128 KiB default field limit (for example, long objective
    # traces from large benchmark jobs).  The comparison reads trusted local
    # artifacts and must not fail after the expensive matrix has completed.
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with reference_path.open(encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    identity_keys = ("suite", "dataset_id", "method", "seed")
    reference = {
        tuple(str(row[key]) for key in identity_keys): row
        for row in reference_rows
    }
    current = {
        tuple(str(row[key]) for key in identity_keys): row
        for row in current_results
    }
    paired: list[dict[str, Any]] = []
    new_failures: list[str] = []
    recovered: list[str] = []
    for identity in sorted(set(reference) & set(current)):
        old = reference[identity]
        new = current[identity]
        old_ok = old.get("status") == "ok"
        new_ok = new.get("status") == "ok"
        label = " / ".join(identity)
        if old_ok and not new_ok:
            new_failures.append(label)
            continue
        if new_ok and not old_ok:
            recovered.append(label)
            continue
        if not old_ok:
            continue
        metric_name = {
            "synthetic": "macro_target_f1",
            "cytometry": "macro_target_f1",
            "gaia": "best_cluster_f1",
        }[new["suite"]]
        old_metric_text = old.get(metric_name)
        new_metric = new.get("metrics", {}).get(metric_name)
        if old_metric_text in {None, ""} or new_metric is None:
            continue
        old_metric = float(old_metric_text)
        new_metric = float(new_metric)
        old_runtime_text = old.get("runtime_seconds")
        new_runtime = new.get("runtime_seconds")
        runtime_ratio = (
            float(new_runtime) / float(old_runtime_text)
            if old_runtime_text not in {None, "", "0", "0.0"} and new_runtime is not None
            else None
        )
        paired.append({
            "suite": new["suite"],
            "dataset_id": new["dataset_id"],
            "method": new["method"],
            "seed": new["seed"],
            "metric": metric_name,
            "reference_quality": old_metric,
            "current_quality": new_metric,
            "quality_delta": new_metric - old_metric,
            "runtime_ratio": runtime_ratio,
        })
    deltas = [row["quality_delta"] for row in paired]
    runtime_ratios = [row["runtime_ratio"] for row in paired if row["runtime_ratio"] is not None]
    tolerance = float(config.get("quality_tolerance", 0.002))
    severe_limit = float(config.get("severe_regression", 0.02))
    runtime_limit = float(config.get("runtime_ratio_limit", 1.25))
    median_delta = statistics.median(deltas) if deltas else None
    worst_delta = min(deltas) if deltas else None
    median_runtime_ratio = statistics.median(runtime_ratios) if runtime_ratios else None
    suite_delta_groups = {
        suite: [row["quality_delta"] for row in paired if row["suite"] == suite]
        for suite in {row["suite"] for row in paired}
    }
    suite_regression = any(
        statistics.median(suite_deltas) < -tolerance
        for suite_deltas in suite_delta_groups.values()
    )
    missing_reference = sorted(" / ".join(key) for key in set(current) - set(reference))
    missing_current = sorted(" / ".join(key) for key in set(reference) - set(current))
    if missing_reference or missing_current:
        verdict = "incomparable"
    elif (
        new_failures
        or suite_regression
        or (worst_delta is not None and worst_delta < -severe_limit)
        or (median_delta is not None and median_delta < -tolerance)
    ):
        verdict = "regressed"
    elif not paired:
        verdict = "incomparable"
    elif median_delta is not None and median_delta > tolerance:
        verdict = "improved"
    else:
        verdict = "unchanged"
    if verdict in {"improved", "unchanged"} and median_runtime_ratio is not None and median_runtime_ratio > runtime_limit:
        verdict += "_slower"
    by_suite: dict[str, dict[str, Any]] = {}
    for suite in sorted({row["suite"] for row in paired}):
        suite_rows = [row for row in paired if row["suite"] == suite]
        suite_deltas = [row["quality_delta"] for row in suite_rows]
        suite_ratios = [row["runtime_ratio"] for row in suite_rows if row["runtime_ratio"] is not None]
        by_suite[suite] = {
            "pairs": len(suite_rows),
            "median_quality_delta": statistics.median(suite_deltas),
            "worst_quality_delta": min(suite_deltas),
            "improved_pairs": sum(delta > tolerance for delta in suite_deltas),
            "regressed_pairs": sum(delta < -tolerance for delta in suite_deltas),
            "median_runtime_ratio": statistics.median(suite_ratios) if suite_ratios else None,
        }
    return {
        "schema_version": 1,
        "verdict": verdict,
        "reference": str(reference_path.resolve()),
        "paired_jobs": len(paired),
        "median_quality_delta": median_delta,
        "worst_quality_delta": worst_delta,
        "median_runtime_ratio": median_runtime_ratio,
        "new_failures": new_failures,
        "recovered_jobs": recovered,
        "missing_reference_jobs": missing_reference,
        "missing_current_jobs": missing_current,
        "by_suite": by_suite,
        "pairs": paired,
    }


def create_report(
    path: Path,
    manifest: dict[str, Any],
    validation: dict[str, Any],
    results: list[dict[str, Any]],
    comparison: dict[str, Any] | None = None,
) -> None:
    lines = [
        f"# StrataSCAN benchmark protocol {manifest['protocol_version']}",
        "",
        f"Suite: `{manifest['suite']}`. Expected jobs: {validation['expected']}. "
        f"Structural validation: **{'complete' if validation['complete'] else 'incomplete'}**.",
        "",
        f"Resources: {manifest['resources']['threads']} thread(s), "
        f"{manifest['resources']['timeout_seconds']} s timeout, "
        f"{manifest['resources']['memory_limit_mb']} MB RSS limit.",
        "",
        "| Suite | Method | Jobs | OK | Failures | Median primary metric | Median runtime, s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault((result["suite"], result["method"]), []).append(result)
    for (suite, method), rows in sorted(groups.items()):
        ok = [row for row in rows if row.get("status") == "ok"]
        scores = sorted(score for row in ok if (score := _primary_metric(row)) is not None)
        runtimes = sorted(float(row["runtime_seconds"]) for row in ok if row.get("runtime_seconds") is not None)
        median_score = statistics.median(scores) if scores else None
        median_runtime = statistics.median(runtimes) if runtimes else None
        score_text = "—" if median_score is None else f"{median_score:.4f}"
        runtime_text = "—" if median_runtime is None else f"{median_runtime:.3f}"
        lines.append(
            f"| {suite} | {method} | {len(rows)} | {len(ok)} | {len(rows) - len(ok)} | "
            f"{score_text} | {runtime_text} |"
        )
    lines.extend([
        "",
        "Controlled failures remain in the denominator. This report is generated only from job records listed in the frozen manifest.",
        "",
    ])
    if comparison is not None:
        lines.extend([
            "## Paired comparison",
            "",
            f"Verdict: **{comparison['verdict']}**. Paired jobs: {comparison['paired_jobs']}.",
            "",
            f"Median quality delta: {comparison['median_quality_delta']}. "
            f"Worst quality delta: {comparison['worst_quality_delta']}. "
            f"Median runtime ratio: {comparison['median_runtime_ratio']}.",
            "",
            "| Suite | Pairs | Median quality delta | Worst delta | Improved | Regressed | Median runtime ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for suite, summary in comparison["by_suite"].items():
            lines.append(
                f"| {suite} | {summary['pairs']} | {summary['median_quality_delta']:.6f} | "
                f"{summary['worst_quality_delta']:.6f} | {summary['improved_pairs']} | "
                f"{summary['regressed_pairs']} | {summary['median_runtime_ratio']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified StrataSCAN benchmark matrix")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--evaluation-protocol", type=Path, default=DEFAULT_EVALUATION_PROTOCOL
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/full_v200"))
    parser.add_argument("--suite", choices=("all", "synthetic", "cytometry", "gaia"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--job-id",
        action="append",
        default=[],
        help="run only this exact expanded job ID; repeat for multiple jobs",
    )
    parser.add_argument(
        "--job-ids-from",
        type=Path,
        help="UTF-8 text file containing one exact expanded job ID per line",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="number of isolated benchmark jobs to execute concurrently",
    )
    parser.add_argument("--list", action="store_true", help="print the frozen matrix size without executing jobs")
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="print every expanded job as one JSON object without executing",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--compare-to", type=Path, help="reference results.csv or its run directory")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    protocol_path = args.protocol.resolve()
    protocol = read_protocol(protocol_path)
    evaluation_protocol_path = args.evaluation_protocol.resolve()
    evaluation_protocol = read_evaluation_protocol(evaluation_protocol_path)
    expanded_jobs = expand_jobs(protocol, args.suite, evaluation_protocol)
    requested_ids = set(args.job_id)
    if args.job_ids_from is not None:
        requested_ids.update(
            line.strip()
            for line in args.job_ids_from.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    jobs = select_jobs(expanded_jobs, requested_ids)
    if args.list_jobs:
        for job in jobs:
            print(json.dumps(job, sort_keys=True))
        return
    if args.list:
        counts: dict[str, int] = {}
        for job in jobs:
            counts[job["suite"]] = counts.get(job["suite"], 0) + 1
        print(json.dumps({"total": len(jobs), "by_suite": counts}, indent=2, sort_keys=True))
        return
    if not args.validate_only:
        require_runtime_dependencies(protocol, args.suite)

    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    jobs_dir = output / "jobs"
    specs_dir = output / "specs"
    evaluation_state_dir = output / "evaluation_state"
    output.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(exist_ok=True)
    specs_dir.mkdir(exist_ok=True)
    evaluation_state_dir.mkdir(exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_ids = [job["job_id"] for job in manifest["jobs"]]
        current_ids = [job["job_id"] for job in jobs]
        if existing_ids != current_ids:
            raise ValueError("existing manifest does not match the requested protocol matrix")
        if not args.validate_only:
            drift = input_drift(
                manifest.get("inputs", []),
                input_checksums(
                    protocol, protocol_path, args.suite, evaluation_protocol_path
                ),
            )
            if drift:
                raise ValueError(
                    "benchmark inputs changed since the manifest was frozen; "
                    f"use a new output directory. Changed paths: {drift}"
                )
            recorded_workers = int(
                manifest.get("execution", {}).get("max_workers", 1)
            )
            if recorded_workers != args.max_workers:
                raise ValueError(
                    "--max-workers differs from the frozen execution mode "
                    f"({recorded_workers}); use a new output directory"
                )
    elif args.validate_only:
        raise FileNotFoundError("validate-only requires an existing manifest")
    else:
        manifest = create_manifest(
            protocol,
            protocol_path,
            args.suite,
            jobs,
            expanded_job_count=len(expanded_jobs),
            max_workers=args.max_workers,
            evaluation_protocol=evaluation_protocol,
            evaluation_protocol_path=evaluation_protocol_path,
        )
        atomic_json(manifest_path, manifest)

    if not args.validate_only:
        existing_results = list(jobs_dir.glob("*.json"))
        if existing_results and not args.resume:
            raise FileExistsError("results already exist; use --resume or a new output directory")
        total = len(jobs)
        pending: list[tuple[int, dict[str, Any], Path, Path, Path, str]] = []
        for index, job in enumerate(jobs, start=1):
            spec_path = specs_dir / f"{job['job_id']}.json"
            result_path = jobs_dir / f"{job['job_id']}.json"
            evaluation_state_path = evaluation_state_dir / f"{job['job_id']}.npz"
            evaluation_state_reference = f"evaluation_state/{job['job_id']}.npz"
            atomic_json(spec_path, job)
            if args.resume and result_path.exists():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                artifact = existing.get("evaluation_artifact", {})
                if (
                    existing.get("job_id") == job["job_id"]
                    and evaluation_state_path.is_file()
                    and artifact.get("path") == evaluation_state_reference
                    and artifact.get("sha256") == sha256(evaluation_state_path)
                ):
                    print(f"[{index}/{total}] resume {job['job_id']}", flush=True)
                    continue
            pending.append((
                index, job, spec_path, result_path,
                evaluation_state_path, evaluation_state_reference,
            ))

        def execute_pending(
            item: tuple[int, dict[str, Any], Path, Path, Path, str]
        ) -> tuple[int, dict[str, Any], dict[str, Any]]:
            index, job, spec_path, result_path, state_path, state_reference = item
            print(f"[{index}/{total}] run {job['job_id']}", flush=True)
            result = run_job(
                job, spec_path, result_path, manifest["resources"],
                state_path, state_reference,
            )
            return index, job, result

        if args.max_workers == 1:
            completed = map(execute_pending, pending)
            for index, job, result in completed:
                print(f"[{index}/{total}] {result['status']} {job['job_id']}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(execute_pending, item) for item in pending]
                for future in as_completed(futures):
                    index, job, result = future.result()
                    print(f"[{index}/{total}] {result['status']} {job['job_id']}", flush=True)

    validation, results = validate(manifest, jobs_dir)
    atomic_json(output / "validation.json", validation)
    comparison = None
    if validation["complete"]:
        write_csv(output / "results.csv", results)
        if args.compare_to is not None:
            comparison = compare_with_reference(
                args.compare_to.resolve(), results, protocol.get("comparison", {})
            )
            atomic_json(output / "comparison.json", comparison)
        create_report(output / "report.md", manifest, validation, results, comparison)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not validation["complete"]:
        raise SystemExit(2)
    if comparison is not None and comparison["verdict"] == "regressed":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
