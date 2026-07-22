from __future__ import annotations

import numpy as np


def _comb2(x: np.ndarray | int) -> np.ndarray | int:
    return x * (x - 1) // 2


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    true_signal = y_true >= 0
    pred_signal = y_pred >= 0

    mask = true_signal & pred_signal
    true_pairs = int(np.sum(_comb2(np.bincount(y_true[true_signal])))) if np.any(true_signal) else 0
    pred_vals = y_pred[pred_signal]
    pred_pairs = int(np.sum(_comb2(np.bincount(pred_vals)))) if pred_vals.size else 0
    if np.any(mask):
        tvals, ti = np.unique(y_true[mask], return_inverse=True)
        pvals, pi = np.unique(y_pred[mask], return_inverse=True)
        counts = np.bincount(ti * len(pvals) + pi)
        tp = int(np.sum(_comb2(counts)))
    else:
        tp = 0
    precision = tp / pred_pairs if pred_pairs else 0.0
    recall = tp / true_pairs if true_pairs else 0.0
    pairwise_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    true_noise = ~true_signal
    pred_noise = ~pred_signal
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
        "signal_coverage": float(np.mean(pred_signal[true_signal])) if np.any(true_signal) else 0.0,
        "background_rejection": float(np.mean(pred_noise[true_noise])) if np.any(true_noise) else 0.0,
        "n_clusters": int(np.unique(y_pred[y_pred >= 0]).size),
        "noise_fraction": float(np.mean(y_pred < 0)),
    }
