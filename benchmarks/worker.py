from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import threading
from time import perf_counter
import traceback
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[variable] = "1"

import psutil
from threadpoolctl import threadpool_limits

from benchmarks.datasets import load_dataset
from benchmarks.evaluation import evaluate
from benchmarks.methods import run_method
from stratascan import __version__


class PeakRSS:
    def __init__(self, interval: float = 0.005) -> None:
        self.process = psutil.Process()
        self.interval = interval
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self.stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRSS":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / 2**20

    @property
    def delta_mb(self) -> float:
        return max(0.0, (self.peak - self.baseline) / 2**20)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def execute(job: dict[str, Any], repo: Path) -> dict[str, Any]:
    loaded_at = perf_counter()
    dataset = load_dataset(job, repo)
    load_seconds = perf_counter() - loaded_at
    if job["method"] == "SNN-DBSCAN":
        from baselines import warm_numba_kernels

        warm_numba_kernels()
    elif job["method"] == "kNN+Leiden":
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    with threadpool_limits(limits=1), PeakRSS() as memory:
        started = perf_counter()
        method = run_method(job["method"], dataset.X, int(job["seed"]))
        runtime_seconds = perf_counter() - started
    metrics = evaluate(dataset, method.labels, job["suite"])
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "status": "ok",
        "suite": job["suite"],
        "dataset_id": job["dataset_id"],
        "method": job["method"],
        "seed": int(job["seed"]),
        "package_version": __version__,
        "n": int(dataset.X.shape[0]),
        "dimension": int(dataset.X.shape[1]),
        "load_seconds": float(load_seconds),
        "runtime_seconds": float(runtime_seconds),
        "peak_rss_mb": float(memory.peak_mb),
        "incremental_rss_mb": float(memory.delta_mb),
        "dataset_metadata": dataset.metadata,
        "parameters": method.parameters,
        "profile": method.profile,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    try:
        result = execute(job, args.repo.resolve())
    except Exception as error:
        result = {
            "schema_version": 1,
            "job_id": job.get("job_id"),
            "status": "error",
            "suite": job.get("suite"),
            "dataset_id": job.get("dataset_id"),
            "method": job.get("method"),
            "seed": job.get("seed"),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_json(args.output, result)
        raise SystemExit(1)
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
