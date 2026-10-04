from __future__ import annotations

import numpy as np


def _comb2(x: np.ndarray | int) -> np.ndarray | int:
    return x * (x - 1) // 2


def _validated_labels(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{name} must contain integer labels")
    return result.astype(np.int64, copy=False)


def _pair_count(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    _, counts = np.unique(values, return_counts=True)
    return int(np.sum(_comb2(counts)))


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = _validated_labels(y_true, "y_true")
    y_pred = _validated_labels(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    true_signal = y_true >= 0
    pred_signal = y_pred >= 0

    mask = true_signal & pred_signal
    true_pairs = _pair_count(y_true[true_signal])
    pred_pairs = _pair_count(y_pred[pred_signal])
    if np.any(mask):
        intersections = np.column_stack((y_true[mask], y_pred[mask]))
        _, counts = np.unique(intersections, axis=0, return_counts=True)
        tp = int(np.sum(_comb2(counts)))
    else:
        tp = 0
    precision = tp / pred_pairs if pred_pairs else 0.0
    recall = tp / true_pairs if true_pairs else 0.0
    pairwise_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    true_noise = ~true_signal
    pred_noise = ~pred_signal
    signal_tp = int(np.sum(true_signal & pred_signal))
    signal_precision = signal_tp / int(np.sum(pred_signal)) if np.any(pred_signal) else 0.0
    signal_recall = signal_tp / int(np.sum(true_signal)) if np.any(true_signal) else 0.0
    signal_f1 = (
        2 * signal_precision * signal_recall / (signal_precision + signal_recall)
        if signal_precision + signal_recall
        else 0.0
    )
    noise_tp = int(np.sum(true_noise & pred_noise))
    noise_precision = noise_tp / int(np.sum(pred_noise)) if np.any(pred_noise) else 0.0
    noise_recall = noise_tp / int(np.sum(true_noise)) if np.any(true_noise) else 0.0
    noise_f1 = (
        2 * noise_precision * noise_recall / (noise_precision + noise_recall)
        if noise_precision + noise_recall
        else 0.0
    )
    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": pairwise_f1,
        "noise_precision": noise_precision,
        "noise_recall": noise_recall,
        "noise_f1": noise_f1,
        "signal_precision": signal_precision,
        "signal_recall": signal_recall,
        "signal_f1": signal_f1,
        "binary_macro_f1": 0.5 * (signal_f1 + noise_f1),
        "binary_balanced_accuracy": 0.5 * (signal_recall + noise_recall),
        "signal_coverage": signal_recall,
        "background_rejection": noise_recall,
        "n_clusters": int(np.unique(y_pred[y_pred >= 0]).size),
        "noise_fraction": float(np.mean(y_pred < 0)),
    }
