from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from stratascan.synthetic import make_overlapping_density_16d


def test_overlapping_density_has_variable_signal_density_and_background_overlap() -> None:
    x, y = make_overlapping_density_16d(5_000, seed=42)
    assert x.shape == (5_000, 16)
    assert set(np.unique(y[y >= 0]).tolist()) == set(range(8))
    assert np.mean(y >= 0) == 0.20

    distances, _ = NearestNeighbors(n_neighbors=5, algorithm="brute", n_jobs=1).fit(x).kneighbors(x)
    fourth = distances[:, 4]
    background = fourth[y < 0]
    signal = fourth[y >= 0]

    # Neither class occupies a disjoint interval of the k-distance curve.
    assert np.quantile(signal, 0.90) > np.quantile(background, 0.10)
    assert np.quantile(signal, 0.10) < np.quantile(background, 0.90)

    medians = np.array([np.median(fourth[y == cluster]) for cluster in range(8)])
    assert np.ptp(medians) > 0.35
