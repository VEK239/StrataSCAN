from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.datasets import load_synthetic
from benchmarks.evaluation import evaluate
from stratascan import build_knn_graph
from stratascan.optimization import (
    GammaMDLConfig,
    OptimizationStrictCoreConfig,
    estimate_mdl_multiscale_stratification,
    optimize_strict_core_from_graph,
)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _configurations(
    *,
    initial_max: int,
    hard_max: int,
    n_init: int,
) -> dict[str, GammaMDLConfig]:
    common = {
        "max_components": initial_max,
        "hard_max_components": hard_max,
        "n_init": n_init,
    }
    return {
        "dev10_fixed_gap": GammaMDLConfig(
            max_components=initial_max,
            n_init=n_init,
            adaptive_components=False,
            hard_max_components=initial_max,
            component_roles="largest_rate_gap",
            entropy_scope="basis",
            split_warm_start=False,
        ),
        "adaptive_gap": GammaMDLConfig(
            **common,
            component_roles="largest_rate_gap",
        ),
        "adaptive_rate_changepoint": GammaMDLConfig(
            **common,
            component_roles="rate_changepoint_mdl",
        ),
        "adaptive_coded_background": GammaMDLConfig(
            **common,
            component_roles="background_scan_mdl",
        ),
        "validated_rate_prefix": GammaMDLConfig(
            **common,
            component_roles="validated_rate_prefix",
        ),
        "continuous_rate_background": GammaMDLConfig(
            **common,
            background_distribution="gamma_rate_mixture",
        ),
    }


def _profile_row(
    *,
    case_id: str,
    seed: int,
    model: str,
    profile: dict[str, object],
    labels: np.ndarray,
    fit_seconds: float,
    quality: dict[str, float] | None,
) -> dict[str, object]:
    candidates = list(map(int, profile["mdl_gamma_candidate_components"]))
    selected = int(profile["mdl_gamma_selected_components"])
    selected_index = candidates.index(selected)
    converged = list(profile["mdl_gamma_candidate_converged"])
    row: dict[str, object] = {
        "case_id": case_id,
        "seed": seed,
        "model": model,
        "fit_seconds": fit_seconds,
        "selected_gamma_bases": selected,
        "selected_signal_strata": int(profile["mdl_gamma_signal_components"]),
        "selected_background_bases": int(profile["mdl_gamma_background_components"]),
        "background_fraction": float(profile["mdl_gamma_background_fraction"]),
        "mean_background_probability": float(
            profile.get("mdl_gamma_mean_background_probability", np.nan)
        ),
        "search_caps": json.dumps(profile["mdl_gamma_search_caps"]),
        "hit_hard_max": bool(profile["mdl_gamma_search_hit_hard_max"]),
        "selected_fit_converged": bool(converged[selected_index]),
        "n_clusters": int(np.unique(labels[labels >= 0]).size),
        "noise_fraction": float(np.mean(labels < 0)),
    }
    if quality is not None:
        row.update(
            {
                "macro_target_f1": quality["macro_target_f1"],
                "pairwise_f1": quality["pairwise_f1"],
                "noise_f1": quality["noise_f1"],
            }
        )
    return row


def _run_one(
    graph,
    *,
    dimension: int,
    config: GammaMDLConfig,
) -> tuple[np.ndarray, dict[str, object], float]:
    started = perf_counter()
    stratification = estimate_mdl_multiscale_stratification(
        graph,
        ambient_dimension=float(dimension),
        config=config,
    )
    result = optimize_strict_core_from_graph(
        graph,
        stratification,
        ambient_dimension=float(dimension),
        config=OptimizationStrictCoreConfig(),
    )
    profile = {**stratification.diagnostics, **result.profile}
    return result.labels, profile, perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate adaptive Gamma bases and semantic background models."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "benchmarks" / "protocol.optimization-v0.2.0-dev10.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2_000)
    parser.add_argument("--seeds", default="23")
    parser.add_argument("--cases", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--initial-max", type=int, default=8)
    parser.add_argument("--hard-max", type=int, default=24)
    parser.add_argument("--null-dimensions", default="2,8,16")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    seeds = [int(value) for value in _parse_csv(args.seeds)]
    requested_cases = set(_parse_csv(args.cases))
    requested_models = set(_parse_csv(args.models))
    null_dimensions = [int(value) for value in _parse_csv(args.null_dimensions)]
    configurations = _configurations(
        initial_max=args.initial_max,
        hard_max=args.hard_max,
        n_init=args.n_init,
    )
    if requested_models:
        unknown = requested_models - set(configurations)
        if unknown:
            raise ValueError(f"unknown validation models: {sorted(unknown)}")
        configurations = {
            name: config
            for name, config in configurations.items()
            if name in requested_models
        }

    rows: list[dict[str, object]] = []
    for seed in seeds:
        for case in protocol["synthetic"]["cases"]:
            if requested_cases and case["id"] not in requested_cases:
                continue
            dataset = load_synthetic({**case, "n": args.n}, seed)
            dimension = int(dataset.X.shape[1])
            backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
            graph, _ = build_knn_graph(
                dataset.X, k=32, backend=backend, n_jobs=1
            )
            for name, config in configurations.items():
                labels, profile, seconds = _run_one(
                    graph,
                    dimension=dimension,
                    config=config,
                )
                rows.append(
                    _profile_row(
                        case_id=case["id"],
                        seed=seed,
                        model=name,
                        profile=profile,
                        labels=labels,
                        fit_seconds=seconds,
                        quality=evaluate(dataset, labels, "synthetic"),
                    )
                )
                print(f"completed {case['id']} seed={seed} model={name}", flush=True)

        for dimension in null_dimensions:
            rng = np.random.default_rng(seed + 100_000 + dimension)
            X = rng.uniform(-1.0, 1.0, size=(args.n, dimension)).astype(np.float32)
            backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
            graph, _ = build_knn_graph(X, k=32, backend=backend, n_jobs=1)
            for name, config in configurations.items():
                labels, profile, seconds = _run_one(
                    graph,
                    dimension=dimension,
                    config=config,
                )
                rows.append(
                    _profile_row(
                        case_id=f"uniform_null_{dimension}d",
                        seed=seed,
                        model=name,
                        profile=profile,
                        labels=labels,
                        fit_seconds=seconds,
                        quality=None,
                    )
                )
                print(
                    f"completed uniform_null_{dimension}d seed={seed} model={name}",
                    flush=True,
                )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "model-validation.csv", index=False)
    summary = (
        frame.groupby("model", as_index=False)
        .agg(
            mean_signal_strata=("selected_signal_strata", "mean"),
            hard_max_rate=("hit_hard_max", "mean"),
            convergence_rate=("selected_fit_converged", "mean"),
            mean_runtime_seconds=("fit_seconds", "mean"),
            mean_macro_target_f1=("macro_target_f1", "mean"),
            mean_pairwise_f1=("pairwise_f1", "mean"),
            mean_noise_f1=("noise_f1", "mean"),
        )
        .sort_values("model")
    )
    null = frame[frame["case_id"].str.startswith("uniform_null_")]
    null_summary = (
        null.groupby("model", as_index=False)
        .agg(
            null_false_signal_rate=(
                "selected_signal_strata",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
            null_false_cluster_rate=(
                "n_clusters",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
        )
        .sort_values("model")
    )
    summary = summary.merge(null_summary, on="model", how="left")
    summary.to_csv(output / "model-summary.csv", index=False)
    (output / "configuration.json").write_text(
        json.dumps(
            {
                "n": args.n,
                "seeds": seeds,
                "cases": sorted(requested_cases),
                "models": list(configurations),
                "n_init": args.n_init,
                "initial_max": args.initial_max,
                "hard_max": args.hard_max,
                "null_dimensions": null_dimensions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
