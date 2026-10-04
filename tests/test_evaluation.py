from __future__ import annotations

import numpy as np
import pytest

from benchmarks.datasets import Dataset
from benchmarks.evaluation import (
    EvaluationConfig,
    evaluate,
    evaluation_state,
    noise_aware_macro_f1,
    target_background_hmean_f1,
    target_background_structure_hmean_f1,
)


def _dataset(y: list[int], names: list[str] | None = None) -> Dataset:
    labels = np.asarray(y, dtype=np.int64)
    targets = np.unique(labels[labels >= 0])
    return Dataset(
        X=np.zeros((labels.size, 2), dtype=np.float32),
        y=labels,
        target_names=names or [f"target-{value}" for value in targets],
        metadata={},
    )


def _synthetic(
    y: list[int], predicted: list[int], names: list[str] | None = None, **config: object
) -> dict[str, object]:
    return evaluate(
        _dataset(y, names),
        np.asarray(predicted, dtype=np.int64),
        suite="synthetic",
        config={"background_semantics": "known_synthetic_noise", **config},
    )


def test_perfect_clustering_has_perfect_one_to_one_evidence() -> None:
    metrics = _synthetic([0, 0, 1, 1, -1, -1], [7, 7, 3, 3, -1, -1])
    assert metrics["matching_strategy"] == "hungarian"
    assert metrics["matching_objective"] == "pairwise_f1"
    assert metrics["macro_target_f1"] == metrics["weighted_target_f1"] == 1.0
    assert metrics["macro_target_purity"] == metrics["macro_target_coverage"] == 1.0
    assert metrics["target_discovery_rate"] == 1.0
    assert metrics["unmatched_predicted_cluster_count"] == 0


def test_everything_declared_noise_gives_every_target_zero() -> None:
    metrics = _synthetic([0, 0, 1, 1, -1], [-1, -1, -1, -1, -1])
    assert metrics["macro_target_f1"] == 0.0
    assert metrics["predicted_cluster_count"] == 0
    assert metrics["target_discovery_rate"] == 0.0
    assert all(row["matched_predicted_cluster"] is None for row in metrics["target_matches"])


def test_single_merged_cluster_cannot_explain_multiple_targets() -> None:
    metrics = _synthetic([0, 0, 1, 1], [8, 8, 8, 8])
    matches = metrics["target_matches"]
    assert np.isclose(metrics["macro_target_f1"], 1.0 / 3.0)
    assert sum(row["matched_predicted_cluster"] == 8 for row in matches) == 1
    assert sum(row["matched_predicted_cluster"] is None for row in matches) == 1
    assert metrics["merged_predicted_clusters"] == 1


def test_fragmented_target_uses_best_fragment_and_reports_burden() -> None:
    metrics = _synthetic([0] * 10, [4] * 6 + [9] * 4)
    match = metrics["target_matches"][0]
    assert np.isclose(match["f1"], 0.75)
    assert match["fragments"] == 2
    assert metrics["extra_signal_fragments"] == 1
    assert metrics["unmatched_predicted_cluster_count"] == 1


def test_extra_clusters_only_on_background_do_not_inflate_target_scores() -> None:
    metrics = _synthetic([0, 0, 1, 1, -1, -1, -1], [1, 1, 2, 2, 8, 9, 9])
    assert metrics["macro_target_f1"] == 1.0
    assert metrics["background_only_predicted_clusters"] == 2
    assert metrics["unmatched_predicted_cluster_count"] == 2
    assert metrics["candidate_clusters_per_target"] == 2.0


def test_label_permutations_leave_aggregates_unchanged() -> None:
    first = _synthetic([0, 0, 1, 1, -1], [4, 4, 9, 9, -1])
    second = _synthetic([1, 1, 0, 0, -1], [900, 900, 12, 12, -1])
    for key in ("macro_target_f1", "macro_target_purity", "macro_target_coverage"):
        assert first[key] == second[key]


def test_absent_predicted_target_is_an_unmatched_zero() -> None:
    metrics = _synthetic([0, 0, 1, 1], [5, 5, -1, -1])
    assert metrics["macro_target_f1"] == 0.5
    assert metrics["target_matches"][1]["matched_predicted_cluster"] is None
    assert metrics["target_matches"][1]["f1"] == 0.0


def test_zero_overlap_assignment_is_not_reported_as_a_match() -> None:
    metrics = _synthetic([0, 0, -1, -1], [-1, -1, 42, 42])
    match = metrics["target_matches"][0]
    assert match["matched_predicted_cluster"] is None
    assert match["overlap"] == 0
    assert metrics["background_only_predicted_clusters"] == 1


def test_sparse_noncontiguous_labels_use_positional_target_names() -> None:
    metrics = _synthetic([10, 10, 30, 30, -1], [700, 700, 5, 5, -1], ["ten", "thirty"])
    assert [row["target"] for row in metrics["target_matches"]] == ["ten", "thirty"]
    assert [row["target_label"] for row in metrics["target_matches"]] == [10, 30]
    assert metrics["macro_target_f1"] == 1.0


def test_discovery_thresholds_are_configurable_and_names_are_threshold_independent() -> None:
    y = [0] * 10 + [-1] * 9
    predicted = [3] + [-1] * 18
    default = _synthetic(y, predicted)
    strict = _synthetic(
        y, predicted, discovery_thresholds={"purity": 0.9, "coverage": 0.2}
    )
    assert default["target_discovery_rate"] == 1.0
    assert strict["target_discovery_rate"] == 0.0
    assert strict["discovery_coverage_threshold"] == 0.2
    assert not any(key.startswith("target_discovery_rate_p") for key in strict)


def test_macro_is_primary_and_weighted_is_a_separate_imbalance_diagnostic() -> None:
    metrics = _synthetic([0] * 9 + [1], [2] * 9 + [-1])
    assert metrics["primary_target_aggregation"] == "macro"
    assert metrics["macro_target_f1"] == 0.5
    assert metrics["weighted_target_f1"] == 0.9


def test_background_semantics_separate_true_noise_from_reference_background() -> None:
    dataset = _dataset([0, 0, -1, -1])
    predicted = np.array([1, 1, -1, -1], dtype=np.int64)
    synthetic = evaluate(dataset, predicted, "synthetic")
    biological = evaluate(
        dataset,
        predicted,
        "cytometry",
        {"background_semantics": "heterogeneous_biological_background"},
    )
    assert synthetic["background_semantics"] == "known_synthetic_noise"
    assert synthetic["noise_f1"] == synthetic["noise_evidence_f1"] == 1.0
    assert "noise_f1" not in biological
    assert biological["background_abstention_f1_diagnostic"] == 1.0


def test_evaluation_state_is_lossless_and_contains_audit_contingency() -> None:
    dataset = _dataset([10, 10, 30, -1], ["ten", "thirty"])
    predicted = np.array([7, -1, 9, 12], dtype=np.int64)
    state = evaluation_state(dataset, predicted)
    np.testing.assert_array_equal(state["y_true"], dataset.y)
    np.testing.assert_array_equal(state["y_pred"], predicted)
    np.testing.assert_array_equal(
        state["target_predicted_contingency"], [[1, 0, 0], [0, 1, 0]]
    )


@pytest.mark.parametrize(
    ("labels", "error"),
    [
        (np.array([0.0, 0.0]), TypeError),
        (np.array([[0, 0]]), ValueError),
        (np.array([0, -2]), ValueError),
    ],
)
def test_invalid_predicted_labels_are_rejected(labels: np.ndarray, error: type[Exception]) -> None:
    with pytest.raises(error):
        evaluate(_dataset([0, 0]), labels, "synthetic")


def test_no_reference_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one reference target"):
        evaluate(_dataset([-1, -1]), np.array([-1, -1]), "synthetic")


def test_invalid_evaluation_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="discovery_purity"):
        EvaluationConfig(discovery_purity=1.1).validate()
    with pytest.raises(ValueError, match="matching objective"):
        EvaluationConfig(matching_objective="overlap").validate()


def test_noise_aware_macro_f1_is_harmonic_mean() -> None:
    assert np.isclose(noise_aware_macro_f1(0.8, 0.5), 2.0 * 0.8 * 0.5 / 1.3)


def test_target_background_hmean_rejects_invalid_f1() -> None:
    for values in ((-0.1, 0.5), (0.5, 1.1), (float("nan"), 0.5)):
        with pytest.raises(ValueError):
            target_background_hmean_f1(*values)


def test_structure_hmean_requires_all_three_tasks() -> None:
    assert target_background_structure_hmean_f1(1.0, 1.0, 1.0) == 1.0
    assert target_background_structure_hmean_f1(1.0, 1.0, 0.0) == 0.0
    assert np.isclose(
        target_background_structure_hmean_f1(0.6, 0.3, 0.2),
        3.0 / (1.0 / 0.6 + 1.0 / 0.3 + 1.0 / 0.2),
    )


def test_structure_hmean_rejects_invalid_f1() -> None:
    for values in ((-0.1, 0.5, 0.5), (0.5, 1.1, 0.5), (0.5, 0.5, np.nan)):
        with pytest.raises(ValueError):
            target_background_structure_hmean_f1(*values)
