import numpy as np

from stratascan import (
    PredictiveMultiscaleConfig,
    PredictiveMultiscaleStrataSCAN,
    build_knn_graph,
)
from stratascan.multiscale import (
    MultiscaleConfig,
    estimate_multiscale_stratification,
    multiscale_density_signatures,
)


def test_multiscale_signatures_are_scale_and_anchored_ratios() -> None:
    rng = np.random.default_rng(4)
    graph, _ = build_knn_graph(rng.normal(size=(100, 2)), k=32, backend="brute")
    signatures = multiscale_density_signatures(graph)
    logged = np.log(graph.distances[:, [3, 7, 15, 31]])
    assert signatures.shape == (100, 4)
    assert np.allclose(signatures[:, 0], logged[:, 0])
    assert np.allclose(signatures[:, 1:], logged[:, [0]] - logged[:, 1:])


def test_multiscale_stratification_and_estimator_align_with_graph() -> None:
    rng = np.random.default_rng(7)
    X = np.vstack([
        rng.normal(0.0, 0.15, size=(120, 2)),
        rng.normal(3.0, 0.55, size=(180, 2)),
    ]).astype(np.float32)
    graph, _ = build_knn_graph(X, k=32, backend="brute")
    config = MultiscaleConfig(max_components=4, max_fit_samples=250)
    stratification = estimate_multiscale_stratification(
        graph, ambient_dimension=2.0, config=config
    )
    assert stratification.groups.shape == (X.shape[0],)
    assert stratification.signatures.shape == (X.shape[0], 4)
    assert stratification.mode == "multiscale_gamma_uniform_tail"
    assert stratification.selected_components >= 2
    assert stratification.diagnostics["multiscale_shell_shapes"] == [4, 4, 8, 16]
    assert stratification.supported_groups.size < stratification.selected_components
    model = PredictiveMultiscaleStrataSCAN(
        backend="brute",
        ambient_dimension=2.0,
        predictive_config=PredictiveMultiscaleConfig(
            min_components=2,
            max_components=4,
            validation_repeats=2,
            final_n_init=2,
            max_fit_samples=250,
            max_iter=100,
        ),
    ).fit(X)
    assert model.labels_.shape == (X.shape[0],)
    assert model.profile_["strict_core_stratification_mode"] == (
        "predictive_constrained_multiscale_gamma_uniform_tail"
    )
