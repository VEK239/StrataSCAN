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


BACKGROUND_FAMILIES = (
    "gamma_rate_mixture",
    "lognormal_rate_mixture",
    "inverse_gamma_rate_mixture",
)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _configurations(
    *,
    initial_max: int,
    hard_max: int,
    n_init: int,
    quadrature_order: int,
) -> dict[str, GammaMDLConfig]:
    common = {
        "max_components": initial_max,
        "hard_max_components": hard_max,
        "n_init": n_init,
        "background_quadrature_order": quadrature_order,
    }
    return {
        family: GammaMDLConfig(
            **common,
            background_distribution=family,
        )
        for family in BACKGROUND_FAMILIES
    }


def _profile_row(
    *,
    case_id: str,
    seed: int,
    family: str,
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
        "background_distribution": family,
        "fit_seconds": fit_seconds,
        "selected_semantic_states": selected,
        "selected_signal_strata": int(profile["mdl_gamma_signal_components"]),
        "selected_background_states": int(
            profile["mdl_gamma_background_components"]
        ),
        "background_fraction": float(profile["mdl_gamma_background_fraction"]),
        "mean_background_probability": float(
            profile["mdl_gamma_mean_background_probability"]
        ),
        "background_parameters": json.dumps(
            profile["mdl_gamma_background_parameters"], sort_keys=True
        ),
        "background_parameters_at_bounds": json.dumps(
            profile["mdl_gamma_background_parameters_at_bounds"]
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


def _summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        frame.groupby("background_distribution", as_index=False)
        .agg(
            mean_signal_strata=("selected_signal_strata", "mean"),
            hard_max_rate=("hit_hard_max", "mean"),
            convergence_rate=("selected_fit_converged", "mean"),
            mean_runtime_seconds=("fit_seconds", "mean"),
            mean_macro_target_f1=("macro_target_f1", "mean"),
            mean_pairwise_f1=("pairwise_f1", "mean"),
            mean_noise_f1=("noise_f1", "mean"),
        )
        .sort_values("background_distribution")
    )
    null = frame[frame["case_id"].str.startswith("uniform_null_")]
    null_summary = (
        null.groupby("background_distribution", as_index=False)
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
        .sort_values("background_distribution")
    )
    return summary.merge(null_summary, on="background_distribution"), null_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare latent-rate distributions for the background state."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            REPO
            / "benchmarks"
            / "protocol.optimization-v0.2.0-dev12-background.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2_000)
    parser.add_argument("--seeds", default="23")
    parser.add_argument("--cases", default="")
    parser.add_argument("--families", default="")
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--initial-max", type=int, default=8)
    parser.add_argument("--hard-max", type=int, default=16)
    parser.add_argument("--quadrature-order", type=int, default=24)
    parser.add_argument("--null-dimensions", default="2,8,16")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    seeds = [int(value) for value in _parse_csv(args.seeds)]
    requested_cases = set(_parse_csv(args.cases))
    requested_families = set(_parse_csv(args.families))
    null_dimensions = [int(value) for value in _parse_csv(args.null_dimensions)]
    configurations = _configurations(
        initial_max=args.initial_max,
        hard_max=args.hard_max,
        n_init=args.n_init,
        quadrature_order=args.quadrature_order,
    )
    if requested_families:
        unknown = requested_families - set(configurations)
        if unknown:
            raise ValueError(f"unknown background families: {sorted(unknown)}")
        configurations = {
            family: config
            for family, config in configurations.items()
            if family in requested_families
        }

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
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
            for family, config in configurations.items():
                labels, profile, seconds = _run_one(
                    graph,
                    dimension=dimension,
                    config=config,
                )
                rows.append(
                    _profile_row(
                        case_id=case["id"],
                        seed=seed,
                        family=family,
                        profile=profile,
                        labels=labels,
                        fit_seconds=seconds,
                        quality=evaluate(dataset, labels, "synthetic"),
                    )
                )
                pd.DataFrame(rows).to_csv(
                    output / "background-validation.csv", index=False
                )
                print(
                    f"completed {case['id']} seed={seed} background={family}",
                    flush=True,
                )

        for dimension in null_dimensions:
            rng = np.random.default_rng(seed + 100_000 + dimension)
            X = rng.uniform(-1.0, 1.0, size=(args.n, dimension)).astype(np.float32)
            backend = "kd_tree" if dimension <= 2 else "faiss_hnsw"
            graph, _ = build_knn_graph(X, k=32, backend=backend, n_jobs=1)
            for family, config in configurations.items():
                labels, profile, seconds = _run_one(
                    graph,
                    dimension=dimension,
                    config=config,
                )
                rows.append(
                    _profile_row(
                        case_id=f"uniform_null_{dimension}d",
                        seed=seed,
                        family=family,
                        profile=profile,
                        labels=labels,
                        fit_seconds=seconds,
                        quality=None,
                    )
                )
                pd.DataFrame(rows).to_csv(
                    output / "background-validation.csv", index=False
                )
                print(
                    f"completed uniform_null_{dimension}d seed={seed} "
                    f"background={family}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "background-validation.csv", index=False)
    summary, null_summary = _summaries(frame)
    summary.to_csv(output / "background-summary.csv", index=False)
    null_summary.to_csv(output / "background-null-summary.csv", index=False)
    (output / "configuration.json").write_text(
        json.dumps(
            {
                "n": args.n,
                "seeds": seeds,
                "cases": sorted(requested_cases),
                "background_distributions": list(configurations),
                "n_init": args.n_init,
                "initial_max": args.initial_max,
                "hard_max": args.hard_max,
                "quadrature_order": args.quadrature_order,
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
