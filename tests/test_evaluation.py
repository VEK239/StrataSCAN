from __future__ import annotations

import numpy as np

from benchmarks.datasets import Dataset
from benchmarks.evaluation import (
    evaluate,
    noise_aware_macro_f1,
    target_background_hmean_f1,
    target_background_structure_hmean_f1,
)


def _dataset() -> Dataset:
    return Dataset(
        X=np.zeros((6, 2), dtype=np.float32),
        y=np.array([0, 0, 1, 1, -1, -1], dtype=np.int64),
        target_names=["a", "b"],
        metadata={},
    )


def test_noise_aware_macro_f1_requires_noise_identification() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    metrics = evaluate(_dataset(), labels, suite="cytometry")

    assert metrics["macro_target_f1"] == 1.0
    assert metrics["noise_f1"] == 0.0
    assert metrics["noise_aware_macro_f1"] == 0.0
    assert metrics["target_background_hmean_f1"] == 0.0


def test_noise_aware_macro_f1_is_one_for_jointly_perfect_output() -> None:
    labels = np.array([0, 0, 1, 1, -1, -1], dtype=np.int64)

    metrics = evaluate(_dataset(), labels, suite="cytometry")

    assert metrics["noise_aware_macro_f1"] == 1.0


def test_noise_aware_macro_f1_is_harmonic_mean() -> None:
    assert np.isclose(noise_aware_macro_f1(0.8, 0.5), 2.0 * 0.8 * 0.5 / 1.3)


def test_target_background_hmean_rejects_invalid_f1() -> None:
    for values in ((-0.1, 0.5), (0.5, 1.1), (float("nan"), 0.5)):
        try:
            target_background_hmean_f1(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid inputs were accepted: {values}")


def test_structure_hmean_requires_all_three_tasks() -> None:
    assert target_background_structure_hmean_f1(1.0, 1.0, 1.0) == 1.0
    assert target_background_structure_hmean_f1(1.0, 1.0, 0.0) == 0.0
    assert np.isclose(
        target_background_structure_hmean_f1(0.6, 0.3, 0.2),
        3.0 / (1.0 / 0.6 + 1.0 / 0.3 + 1.0 / 0.2),
    )


def test_structure_hmean_rejects_invalid_f1() -> None:
    for values in ((-0.1, 0.5, 0.5), (0.5, 1.1, 0.5), (0.5, 0.5, np.nan)):
        with np.testing.assert_raises(ValueError):
            target_background_structure_hmean_f1(*values)
