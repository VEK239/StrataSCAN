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

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[variable] = "1"

import psutil
from threadpoolctl import threadpool_limits

from benchmarks.methods import run_method
from intended_regime_benchmark.metrics import evaluate
from topology_noise_benchmark.generators import generate


class PeakRSS:
    def __init__(self) -> None:
        self.process = psutil.Process(); self.baseline = self.process.memory_info().rss; self.peak = self.baseline
        self.stop = threading.Event(); self.thread = threading.Thread(target=self._watch, daemon=True)
    def _watch(self) -> None:
        while not self.stop.wait(.01): self.peak = max(self.peak, self.process.memory_info().rss)
    def __enter__(self) -> "PeakRSS": self.thread.start(); return self
    def __exit__(self, *_: object) -> None:
        self.stop.set(); self.thread.join(); self.peak = max(self.peak, self.process.memory_info().rss)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True); temporary = Path(handle.name)
    os.replace(temporary, path)


def execute(job: dict[str, Any]) -> dict[str, Any]:
    dataset = generate(topology=job["topology"], density_ratio=float(job["density_ratio"]), noise_fraction=float(job["noise_fraction"]), geometry_seed=int(job["geometry_seed"]), sampling_seed=int(job["sampling_seed"]), **job["design_parameters"])
    if job["method"] == "SNN-DBSCAN":
        from baselines import warm_numba_kernels
        warm_numba_kernels()
    elif job["method"] == "kNN+Leiden":
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    with threadpool_limits(limits=1), PeakRSS() as memory:
        started = perf_counter(); fitted = run_method(job["method"], dataset.X, int(job["sampling_seed"])); runtime = perf_counter() - started
    return {"schema_version": 1, "pilot_label": "PILOT / exploratory; not final evidence", "job_id": job["job_id"], "status": "ok", "method": job["method"], "cell_id": job["cell_id"], "topology": job["topology"], "density_ratio": job["density_ratio"], "sampling_seed": int(job["sampling_seed"]), "noise_fraction": float(job["noise_fraction"]), "n": int(dataset.X.shape[0]), "runtime_seconds": runtime, "peak_rss_mb": memory.peak / 2**20, "incremental_rss_mb": max(0.0, (memory.peak - memory.baseline) / 2**20), "dataset_metadata": dataset.metadata, "method_parameters": fitted.parameters, "method_profile": fitted.profile, "metrics": evaluate(dataset.y, fitted.labels)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--job", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); job = json.loads(args.job.read_text(encoding="utf-8"))
    try: payload = execute(job)
    except Exception as error:
        payload = {"schema_version": 1, "pilot_label": "PILOT / exploratory; not final evidence", "job_id": job.get("job_id"), "status": "error", **{key: job.get(key) for key in ("method", "cell_id", "topology", "density_ratio", "sampling_seed", "noise_fraction")}, "n": job.get("expected_n"), "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
    atomic_json(args.output, payload)


if __name__ == "__main__": main()
