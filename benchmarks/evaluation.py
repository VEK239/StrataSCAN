from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from benchmarks.datasets import Dataset
from stratascan.metrics import clustering_metrics


def target_background_hmean_f1(
    macro_target_f1: float, background_f1: float
) -> float:
    """Balance population recovery with reference-background identification.

    The harmonic mean is zero when either task is completely missed, so a
    partitioner that assigns every observation cannot lead solely by recovering
    annotated populations. The reference background can contain heterogeneous
    or uncharacterized observations, so this is an evaluation diagnostic rather
    than a claim that every reference-background observation is physical noise.
    """
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
    """Post-hoc balance of target, reference-background, and partition structure.

    This three-way harmonic mean is deliberately a sensitivity diagnostic, not
    a replacement primary endpoint. It is zero if any constituent task is
    completely missed, so many-to-one target matching cannot hide an
    all-assigned, fully merged, or severely fragmented partition.
    """
    values = np.asarray(
        [macro_target_f1, background_f1, pairwise_f1], dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("F1 inputs must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("F1 inputs must lie in [0, 1]")
    if np.any(values == 0.0):
        return 0.0
    return float(3.0 / np.sum(1.0 / values))


def _contingency(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_values = np.unique(y_true[y_true >= 0])
    pred_values = np.unique(y_pred[y_pred >= 0])
    table = np.zeros((true_values.size, pred_values.size), dtype=np.int64)
    for row, true_label in enumerate(true_values):
        mask = y_true == true_label
        for column, pred_label in enumerate(pred_values):
            table[row, column] = int(np.sum(mask & (y_pred == pred_label)))
    return table, true_values, pred_values


def _target_matches(dataset: Dataset, labels: np.ndarray) -> tuple[list[dict[str, Any]], float]:
    table, true_values, pred_values = _contingency(dataset.y, labels)
    rows: list[dict[str, Any]] = []
    for row, true_label in enumerate(true_values):
        true_size = int(np.sum(dataset.y == true_label))
        best = {
            "target": dataset.target_names[int(true_label)],
            "target_size": true_size,
            "matched_predicted_cluster": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fragments": 0,
        }
        best_f1 = 0.0
        for column, pred_label in enumerate(pred_values):
            tp = int(table[row, column])
            if not tp:
                continue
            predicted_size = int(np.sum(labels == pred_label))
            precision = tp / predicted_size
            recall = tp / true_size
            f1 = 2.0 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best.update(
                    matched_predicted_cluster=int(pred_label),
                    precision=float(precision),
                    recall=float(recall),
                    f1=float(f1),
                )
        best["fragments"] = int(np.sum(table[row] >= max(1, int(np.ceil(true_size * 0.05)))))
        rows.append(best)
    macro = float(np.mean([row["f1"] for row in rows])) if rows else 0.0
    return rows, macro


def evaluate(dataset: Dataset, labels: np.ndarray, suite: str) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != dataset.y.shape:
        raise ValueError("predicted labels do not match dataset length")
    base = clustering_metrics(dataset.y, labels)
    matches, macro_f1 = _target_matches(dataset, labels)
    signal = dataset.y >= 0
    if np.any(signal):
        ari_signal = float(adjusted_rand_score(dataset.y[signal], labels[signal]))
        ami_signal = float(adjusted_mutual_info_score(dataset.y[signal], labels[signal]))
    else:
        ari_signal = ami_signal = 0.0
    table, true_values, pred_values = _contingency(dataset.y, labels)
    substantial = np.zeros_like(table, dtype=bool)
    for row, true_label in enumerate(true_values):
        true_size = int(np.sum(dataset.y == true_label))
        substantial[row] = table[row] >= max(1, int(np.ceil(true_size * 0.05)))
    merged_predicted_clusters = int(np.sum(np.sum(substantial, axis=0) >= 2)) if table.size else 0
    spurious_background_majority_clusters = 0
    for pred_label in pred_values:
        predicted = labels == pred_label
        if int(np.sum(predicted & (dataset.y < 0))) > int(np.sum(predicted & (dataset.y >= 0))):
            spurious_background_majority_clusters += 1
    metrics: dict[str, Any] = {
        **base,
        "ari_signal": ari_signal,
        "ami_signal": ami_signal,
        "macro_target_f1": macro_f1,
        "target_background_hmean_f1": target_background_hmean_f1(
            macro_f1, float(base["noise_f1"])
        ),
        # Retained so frozen CSV readers continue to work.
        "noise_aware_macro_f1": target_background_hmean_f1(
            macro_f1, float(base["noise_f1"])
        ),
        "target_background_structure_hmean_f1": target_background_structure_hmean_f1(
            macro_f1, float(base["noise_f1"]), float(base["pairwise_f1"])
        ),
        "recovered_truth_clusters_f1_080": int(np.sum([row["f1"] >= 0.8 for row in matches])),
        "recovered_truth_clusters_f1_090": int(np.sum([row["f1"] >= 0.9 for row in matches])),
        "truth_clusters": int(len(matches)),
        "spurious_background_majority_clusters": int(spurious_background_majority_clusters),
        "merged_predicted_clusters": merged_predicted_clusters,
        "extra_signal_fragments": int(np.sum([max(0, row["fragments"] - 1) for row in matches])),
        "target_success_rate_f1_080": float(np.mean([row["f1"] >= 0.8 for row in matches]))
        if matches
        else 0.0,
        "mean_target_fragments": float(np.mean([row["fragments"] for row in matches]))
        if matches
        else 0.0,
        "target_matches": matches,
    }
    if suite == "gaia":
        metrics["best_cluster_f1"] = matches[0]["f1"]
        metrics["best_cluster_precision"] = matches[0]["precision"]
        metrics["best_cluster_recall"] = matches[0]["recall"]
        metrics["matched_predicted_cluster"] = matches[0]["matched_predicted_cluster"]
    return metrics
