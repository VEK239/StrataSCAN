from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from benchmarks.datasets import Dataset
from stratascan.metrics import clustering_metrics


EVALUATION_PROTOCOL_VERSION = "target-discovery-v1"
MATCHING_STRATEGY = "hungarian"
MATCHING_OBJECTIVE = "pairwise_f1"


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Evaluation-only settings; none of these values affect clustering."""

    protocol_version: str = EVALUATION_PROTOCOL_VERSION
    mode: str = "full_partition"
    background_semantics: str = "reference_negative_background"
    matching_strategy: str = MATCHING_STRATEGY
    matching_objective: str = MATCHING_OBJECTIVE
    discovery_purity: float = 0.90
    discovery_coverage: float = 0.10
    fragment_min_target_fraction: float = 0.05

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None, *, suite: str) -> "EvaluationConfig":
        defaults = {
            "mode": "full_partition" if suite == "synthetic" else "target_discovery",
            "background_semantics": (
                "known_synthetic_noise" if suite == "synthetic" else "reference_negative_background"
            ),
        }
        supplied = dict(value or {})
        thresholds = supplied.pop("discovery_thresholds", {})
        if not isinstance(thresholds, Mapping):
            raise TypeError("evaluation.discovery_thresholds must be an object")
        if "purity" in thresholds:
            supplied["discovery_purity"] = thresholds["purity"]
        if "coverage" in thresholds:
            supplied["discovery_coverage"] = thresholds["coverage"]
        matching = supplied.pop("matching", {})
        if not isinstance(matching, Mapping):
            raise TypeError("evaluation.matching must be an object")
        if "strategy" in matching:
            supplied["matching_strategy"] = matching["strategy"]
        if "objective" in matching:
            supplied["matching_objective"] = matching["objective"]
        allowed = set(cls.__dataclass_fields__)
        unknown = set(supplied) - allowed
        if unknown:
            raise ValueError(f"unknown evaluation settings: {sorted(unknown)}")
        config = cls(**{**defaults, **supplied})
        config.validate()
        return config

    def validate(self) -> None:
        if not self.protocol_version:
            raise ValueError("evaluation protocol_version must be non-empty")
        if self.mode not in {"full_partition", "target_discovery"}:
            raise ValueError("evaluation mode must be 'full_partition' or 'target_discovery'")
        if self.background_semantics not in {
            "known_synthetic_noise",
            "heterogeneous_biological_background",
            "reference_negative_background",
        }:
            raise ValueError(f"unsupported background semantics: {self.background_semantics}")
        if self.matching_strategy != MATCHING_STRATEGY:
            raise ValueError(f"matching strategy must be {MATCHING_STRATEGY!r}")
        if self.matching_objective != MATCHING_OBJECTIVE:
            raise ValueError(f"matching objective must be {MATCHING_OBJECTIVE!r}")
        for name in (
            "discovery_purity",
            "discovery_coverage",
            "fragment_min_target_fraction",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


def target_background_hmean_f1(macro_target_f1: float, background_f1: float) -> float:
    """Balance target recovery with a separately named background diagnostic."""
    target = float(macro_target_f1)
    background = float(background_f1)
    if not math.isfinite(target) or not math.isfinite(background):
        raise ValueError("F1 inputs must be finite")
    if not 0.0 <= target <= 1.0 or not 0.0 <= background <= 1.0:
        raise ValueError("F1 inputs must lie in [0, 1]")
    total = target + background
    return 2.0 * target * background / total if total else 0.0


def noise_aware_macro_f1(macro_target_f1: float, noise_f1: float) -> float:
    """Backward-compatible alias for :func:`target_background_hmean_f1`."""
    return target_background_hmean_f1(macro_target_f1, noise_f1)


def target_background_structure_hmean_f1(
    macro_target_f1: float, background_f1: float, pairwise_f1: float
) -> float:
    """Three-way sensitivity diagnostic for target, background, and structure."""
    values = np.asarray([macro_target_f1, background_f1, pairwise_f1], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("F1 inputs must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("F1 inputs must lie in [0, 1]")
    if np.any(values == 0.0):
        return 0.0
    return float(3.0 / np.sum(1.0 / values))


def _validated_labels(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(result.dtype, np.integer) or np.issubdtype(result.dtype, np.bool_):
        raise TypeError(f"{name} must contain integer labels")
    result = result.astype(np.int64, copy=False)
    if np.any(result < -1):
        raise ValueError(f"{name} may use -1 for background/abstention but no lower label")
    return result


def _contingency(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_values, target_sizes = np.unique(y_true[y_true >= 0], return_counts=True)
    pred_values, predicted_sizes = np.unique(y_pred[y_pred >= 0], return_counts=True)
    table = np.zeros((true_values.size, pred_values.size), dtype=np.int64)
    if table.size:
        true_index = {int(value): index for index, value in enumerate(true_values)}
        pred_index = {int(value): index for index, value in enumerate(pred_values)}
        for true_label, pred_label in zip(y_true, y_pred, strict=True):
            if true_label >= 0 and pred_label >= 0:
                table[true_index[int(true_label)], pred_index[int(pred_label)]] += 1
    return table, true_values, pred_values, target_sizes, predicted_sizes


def _target_name(dataset: Dataset, true_values: np.ndarray, row: int) -> str:
    label = int(true_values[row])
    if len(dataset.target_names) == len(true_values):
        return str(dataset.target_names[row])
    if 0 <= label < len(dataset.target_names):
        return str(dataset.target_names[label])
    raise ValueError(
        "target_names must either align with sorted observed target labels or be indexable by label"
    )


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    weights = np.asarray([row["target_size"] for row in rows], dtype=float)
    values = np.asarray([row[key] for row in rows], dtype=float)
    return float(np.average(values, weights=weights))


def _target_matches(
    dataset: Dataset, labels: np.ndarray, config: EvaluationConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table, true_values, pred_values, target_sizes, predicted_sizes = _contingency(
        dataset.y, labels
    )
    if true_values.size == 0:
        raise ValueError("evaluation requires at least one reference target")
    purity = np.zeros_like(table, dtype=float)
    coverage = np.zeros_like(table, dtype=float)
    f1 = np.zeros_like(table, dtype=float)
    if table.size:
        purity = table / predicted_sizes[None, :]
        coverage = table / target_sizes[:, None]
        denominator = purity + coverage
        f1 = np.divide(
            2.0 * purity * coverage,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        assigned_rows, assigned_columns = linear_sum_assignment(-f1)
        assignment = {
            int(row): int(column)
            for row, column in zip(assigned_rows, assigned_columns, strict=True)
            if f1[row, column] > 0.0
        }
    else:
        assignment = {}
    rows: list[dict[str, Any]] = []
    for row, true_label in enumerate(true_values):
        column = assignment.get(row)
        matched_purity = float(purity[row, column]) if column is not None else 0.0
        matched_coverage = float(coverage[row, column]) if column is not None else 0.0
        matched_f1 = float(f1[row, column]) if column is not None else 0.0
        target_size = int(target_sizes[row])
        fragment_threshold = max(
            1, int(np.ceil(target_size * config.fragment_min_target_fraction))
        )
        target_mask = dataset.y == true_label
        rows.append(
            {
                "target": _target_name(dataset, true_values, row),
                "target_label": int(true_label),
                "target_size": target_size,
                "matched_predicted_cluster": int(pred_values[column]) if column is not None else None,
                "overlap": int(table[row, column]) if column is not None else 0,
                "purity": matched_purity,
                "coverage": matched_coverage,
                "precision": matched_purity,
                "recall": matched_coverage,
                "f1": matched_f1,
                "target_noise_loss": float(np.mean(labels[target_mask] < 0)),
                "fragments": int(np.sum(table[row] >= fragment_threshold)),
                "passes_discovery_thresholds": bool(
                    matched_purity >= config.discovery_purity
                    and matched_coverage >= config.discovery_coverage
                ),
            }
        )
    matched_columns = set(assignment.values())
    target_overlap_by_cluster = np.sum(table, axis=0) if table.size else np.zeros(0, dtype=int)
    background_majority = 0
    for column in range(pred_values.size):
        target_overlap = int(target_overlap_by_cluster[column])
        if int(predicted_sizes[column]) - target_overlap > target_overlap:
            background_majority += 1
    burden = {
        "predicted_cluster_count": int(pred_values.size),
        "matched_predicted_cluster_count": int(len(matched_columns)),
        "unmatched_predicted_cluster_count": int(pred_values.size - len(matched_columns)),
        "candidate_clusters_with_target_overlap": int(np.sum(target_overlap_by_cluster > 0)),
        "background_only_predicted_clusters": int(np.sum(target_overlap_by_cluster == 0)),
        "background_majority_predicted_clusters": int(background_majority),
        "candidate_clusters_per_target": float(pred_values.size / true_values.size),
    }
    return rows, burden


def evaluation_state(dataset: Dataset, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Return the lossless arrays needed to reproduce every evaluation metric."""
    y_true = _validated_labels(dataset.y, "y_true")
    y_pred = _validated_labels(labels, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("predicted labels do not match dataset length")
    table, true_values, pred_values, target_sizes, predicted_sizes = _contingency(y_true, y_pred)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "target_labels": true_values,
        "predicted_labels": pred_values,
        "target_sizes": target_sizes,
        "predicted_sizes": predicted_sizes,
        "target_predicted_contingency": table,
    }


def evaluate(
    dataset: Dataset,
    labels: np.ndarray,
    suite: str,
    config: Mapping[str, Any] | EvaluationConfig | None = None,
) -> dict[str, Any]:
    if suite not in {"synthetic", "cytometry", "gaia"}:
        raise ValueError(f"unsupported evaluation suite: {suite}")
    resolved = (
        config if isinstance(config, EvaluationConfig) else EvaluationConfig.from_mapping(config, suite=suite)
    )
    resolved.validate()
    if suite == "synthetic" and resolved.background_semantics != "known_synthetic_noise":
        raise ValueError("synthetic evaluation must use known_synthetic_noise semantics")
    if suite != "synthetic" and resolved.background_semantics == "known_synthetic_noise":
        raise ValueError("biological and Gaia reference negatives are not known synthetic noise")
    y_true = _validated_labels(dataset.y, "y_true")
    y_pred = _validated_labels(labels, "y_pred")
    if y_pred.shape != y_true.shape:
        raise ValueError("predicted labels do not match dataset length")
    dataset = Dataset(dataset.X, y_true, dataset.target_names, dataset.metadata)
    base = clustering_metrics(y_true, y_pred)
    matches, burden = _target_matches(dataset, y_pred, resolved)
    signal = y_true >= 0
    ari_signal = float(adjusted_rand_score(y_true[signal], y_pred[signal]))
    ami_signal = float(adjusted_mutual_info_score(y_true[signal], y_pred[signal]))
    macro_f1 = float(np.mean([row["f1"] for row in matches]))
    macro_purity = float(np.mean([row["purity"] for row in matches]))
    macro_coverage = float(np.mean([row["coverage"] for row in matches]))
    discovery_rate = float(np.mean([row["passes_discovery_thresholds"] for row in matches]))
    background_f1 = float(base["noise_f1"])
    table, _, _, target_sizes, _ = _contingency(y_true, y_pred)
    substantial = np.zeros_like(table, dtype=bool)
    for row, target_size in enumerate(target_sizes):
        substantial[row] = table[row] >= max(
            1, int(np.ceil(int(target_size) * resolved.fragment_min_target_fraction))
        )
    merged_predicted_clusters = int(np.sum(np.sum(substantial, axis=0) >= 2)) if table.size else 0
    metrics: dict[str, Any] = {
        **base,
        "evaluation_protocol_version": resolved.protocol_version,
        "evaluation_mode": resolved.mode,
        "background_semantics": resolved.background_semantics,
        "matching_strategy": resolved.matching_strategy,
        "matching_objective": resolved.matching_objective,
        "unmatched_target_score": 0.0,
        "discovery_purity_threshold": float(resolved.discovery_purity),
        "discovery_coverage_threshold": float(resolved.discovery_coverage),
        "fragment_min_target_fraction": float(resolved.fragment_min_target_fraction),
        "primary_target_aggregation": "macro",
        "ari_signal": ari_signal,
        "ami_signal": ami_signal,
        "macro_target_f1": macro_f1,
        "weighted_target_f1": _weighted_mean(matches, "f1"),
        "macro_target_purity": macro_purity,
        "weighted_target_purity": _weighted_mean(matches, "purity"),
        "macro_target_coverage": macro_coverage,
        "weighted_target_coverage": _weighted_mean(matches, "coverage"),
        "target_discovery_rate": discovery_rate,
        "weighted_target_discovery_rate": _weighted_mean(matches, "passes_discovery_thresholds"),
        "discovered_target_count": int(np.sum([row["passes_discovery_thresholds"] for row in matches])),
        "target_count": int(len(matches)),
        "truth_clusters": int(len(matches)),
        **burden,
        "spurious_background_majority_clusters": burden["background_majority_predicted_clusters"],
        "merged_predicted_clusters": merged_predicted_clusters,
        "extra_signal_fragments": int(np.sum([max(0, row["fragments"] - 1) for row in matches])),
        "recovered_truth_clusters_f1_080": int(np.sum([row["f1"] >= 0.8 for row in matches])),
        "recovered_truth_clusters_f1_090": int(np.sum([row["f1"] >= 0.9 for row in matches])),
        "target_success_rate_f1_080": float(np.mean([row["f1"] >= 0.8 for row in matches])),
        "mean_target_fragments": float(np.mean([row["fragments"] for row in matches])),
        "macro_target_noise_loss": float(np.mean([row["target_noise_loss"] for row in matches])),
        "predicted_abstention_fraction": float(np.mean(y_pred < 0)),
        "target_matches": matches,
    }
    if suite == "synthetic":
        metrics["noise_evidence_f1"] = background_f1
        metrics["noise_aware_macro_f1"] = target_background_hmean_f1(macro_f1, background_f1)
    else:
        metrics["background_abstention_precision_diagnostic"] = metrics.pop("noise_precision")
        metrics["background_abstention_fraction"] = metrics.pop("noise_recall")
        metrics["background_abstention_f1_diagnostic"] = metrics.pop("noise_f1")
        metrics["predicted_abstention_fraction"] = metrics.pop("noise_fraction")
        metrics["binary_reference_macro_f1_diagnostic"] = metrics.pop("binary_macro_f1")
        metrics["binary_reference_balanced_accuracy_diagnostic"] = metrics.pop(
            "binary_balanced_accuracy"
        )
        metrics["target_background_hmean_f1"] = target_background_hmean_f1(
            macro_f1, background_f1
        )
        metrics["target_background_structure_hmean_f1"] = target_background_structure_hmean_f1(
            macro_f1, background_f1, float(base["pairwise_f1"])
        )
    if suite == "gaia":
        first = matches[0]
        metrics["best_cluster_f1"] = first["f1"]
        metrics["best_cluster_precision"] = first["purity"]
        metrics["best_cluster_recall"] = first["coverage"]
        metrics["matched_predicted_cluster"] = first["matched_predicted_cluster"]
    return metrics
