from __future__ import annotations

from typing import Any

import numpy as np

from redesigned_benchmark_pilot.metrics import evaluate as pilot_evaluate


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = pilot_evaluate(y_true, y_pred)
    true_labels = np.unique(y_true[y_true >= 0])
    pred_labels = np.unique(y_pred[y_pred >= 0])
    substantial = np.zeros((true_labels.size, pred_labels.size), dtype=bool)
    for row, true_label in enumerate(true_labels):
        truth = y_true == true_label
        threshold = max(1, int(np.ceil(np.sum(truth) * 0.05)))
        for column, pred_label in enumerate(pred_labels):
            substantial[row, column] = np.sum(truth & (y_pred == pred_label)) >= threshold
    fragments = np.sum(substantial, axis=1) if substantial.size else np.zeros(true_labels.size)
    metrics["extra_signal_fragments"] = int(np.sum(np.maximum(fragments - 1, 0)))
    metrics["mean_target_fragments"] = float(np.mean(fragments)) if fragments.size else 0.0
    metrics["merged_predicted_clusters"] = (
        int(np.sum(np.sum(substantial, axis=0) >= 2)) if substantial.size else 0
    )
    metrics["spurious_background_majority_clusters"] = int(
        sum(
            np.sum((y_pred == label) & (y_true < 0))
            > np.sum((y_pred == label) & (y_true >= 0))
            for label in pred_labels
        )
    )
    return metrics

