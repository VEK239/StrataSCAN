from __future__ import annotations

import numpy as np

from stratascan.metrics import clustering_metrics


def test_binary_signal_background_metrics_ignore_population_identity() -> None:
    truth = np.array([0, 1, -1, -1], dtype=np.int64)
    predicted = np.array([7, -1, 7, -1], dtype=np.int64)

    metrics = clustering_metrics(truth, predicted)

    assert metrics["signal_precision"] == 0.5
    assert metrics["signal_recall"] == 0.5
    assert metrics["signal_f1"] == 0.5
    assert metrics["noise_precision"] == 0.5
    assert metrics["noise_recall"] == 0.5
    assert metrics["noise_f1"] == 0.5
    assert metrics["binary_macro_f1"] == 0.5
    assert metrics["binary_balanced_accuracy"] == 0.5


def test_binary_metrics_penalize_calling_every_cell_noise() -> None:
    truth = np.array([0, 1, -1, -1], dtype=np.int64)
    predicted = np.full(4, -1, dtype=np.int64)

    metrics = clustering_metrics(truth, predicted)

    assert metrics["signal_f1"] == 0.0
    assert np.isclose(metrics["noise_f1"], 2.0 / 3.0)
    assert np.isclose(metrics["binary_macro_f1"], 1.0 / 3.0)
    assert metrics["binary_balanced_accuracy"] == 0.5
