from __future__ import annotations

import numpy as np


def make_density_contrast(
    n: int,
    *,
    dimension: int,
    density_ratio: float,
    shape: str = "gaussian",
    noise_fraction: float = 0.25,
    n_clusters: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-mass clusters with a controlled peak-density contrast.

    Cluster covariance determinants, rather than radii alone, define the
    contrast.  If ``r`` is ``density_ratio`` and ``d`` is ``dimension``, the
    sparsest cluster has marginal standard deviation 0.75 and, when target
    counts divide evenly, the densest has standard deviation
    ``0.75 / r**(1/d)``.  Intermediate clusters are spaced geometrically in
    peak density.  A count correction keeps the Gaussian density at the
    densest mean exactly ``r`` times that at the sparsest mean even when integer
    target counts differ by one.

    ``shape="anisotropic"`` applies a rotated, determinant-one axis transform
    with a 4:1 longest-to-shortest standard-deviation ratio.  It changes shape
    without changing the declared peak-density contrast.  Target counts,
    center separation, and background support are otherwise held fixed.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer")
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise TypeError("dimension must be an integer")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
        raise TypeError("n_clusters must be an integer")
    if int(dimension) < 2:
        raise ValueError("dimension must be at least two")
    if int(n_clusters) < 3:
        raise ValueError("n_clusters must be at least three")
    if not np.isfinite(density_ratio) or float(density_ratio) < 1.0:
        raise ValueError("density_ratio must be finite and at least one")
    if not np.isfinite(noise_fraction) or not 0.0 <= float(noise_fraction) < 1.0:
        raise ValueError("noise_fraction must lie in [0, 1)")
    if shape not in {"gaussian", "anisotropic"}:
        raise ValueError("shape must be 'gaussian' or 'anisotropic'")

    n = int(n)
    dimension = int(dimension)
    n_clusters = int(n_clusters)
    n_signal = int(round((1.0 - float(noise_fraction)) * n))
    if n_signal < 20 * n_clusters:
        raise ValueError("n and noise_fraction must leave at least 20 points per cluster")

    rng = np.random.default_rng(seed)
    counts = np.full(n_clusters, n_signal // n_clusters, dtype=np.int64)
    counts[: n_signal % n_clusters] += 1

    # Relative peak densities run from r (cluster 0) to 1 (last cluster).
    relative_peak_density = np.geomspace(float(density_ratio), 1.0, n_clusters)
    count_ratio = counts.astype(np.float64) / float(counts[-1])
    scales = 0.75 * (count_ratio / relative_peak_density) ** (1.0 / dimension)

    center_radius = 8.0
    angles = 2.0 * np.pi * np.arange(n_clusters, dtype=np.float64) / n_clusters
    centers = np.zeros((n_clusters, dimension), dtype=np.float64)
    centers[:, 0] = center_radius * np.cos(angles)
    centers[:, 1] = center_radius * np.sin(angles)

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for cluster, (count, scale, angle) in enumerate(
        zip(counts, scales, angles, strict=True)
    ):
        root = np.eye(dimension, dtype=np.float64)
        if shape == "anisotropic":
            # Product of the axis multipliers is one, preserving det(covariance).
            axis = np.ones(dimension, dtype=np.float64)
            axis[:2] = (2.0, 0.5)
            rotation = np.eye(dimension, dtype=np.float64)
            cosine, sine = np.cos(angle + 0.31), np.sin(angle + 0.31)
            rotation[:2, :2] = ((cosine, -sine), (sine, cosine))
            root = rotation @ np.diag(axis)
        points = rng.normal(size=(int(count), dimension)) @ root.T
        points = points * float(scale) + centers[cluster]
        blocks.append(points.astype(np.float32))
        labels.append(np.full(int(count), cluster, dtype=np.int64))

    n_noise = n - n_signal
    # A common uniform background makes the comparison about target-density
    # contrast; it does not use target labels or change with density_ratio.
    lower = np.full(dimension, -3.0, dtype=np.float64)
    upper = np.full(dimension, 3.0, dtype=np.float64)
    lower[:2] = -12.0
    upper[:2] = 12.0
    background = rng.uniform(lower, upper, size=(n_noise, dimension)).astype(np.float32)

    X = np.vstack([*blocks, background])
    y = np.concatenate([*labels, np.full(n_noise, -1, dtype=np.int64)])
    order = rng.permutation(n)
    return np.asarray(X[order], dtype=np.float32, order="C"), y[order]


def make_multidensity_2d(n: int, *, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Six anisotropic clusters embedded in 75% uniform background."""
    rng = np.random.default_rng(seed)
    n_signal = int(round(0.25 * n))
    n_noise = n - n_signal
    centers = np.array([
        [-9.0, -7.0], [-1.5, -7.5], [7.5, -6.0],
        [-7.0, 5.5], [0.5, 6.5], [8.0, 5.0],
    ], dtype=np.float32)
    weights = np.array([0.12, 0.16, 0.18, 0.14, 0.19, 0.21])
    counts = rng.multinomial(n_signal, weights / weights.sum())
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for c, (center, count) in enumerate(zip(centers, counts, strict=True)):
        scale = 0.25 * (1.0 + 0.35 * (c % 3))
        angle = (c + 1) * 0.37
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        cov_root = rot @ np.diag([scale, scale * (1.8 + 0.2 * (c % 2))])
        pts = rng.normal(size=(count, 2)) @ cov_root.T + center
        blocks.append(pts.astype(np.float32))
        labels.append(np.full(count, c, dtype=np.int64))
    # Deliberately broad background: the target regime is a small dense signal
    # embedded in a much larger low-density support, matching the v0.3.3
    # scalability protocol.
    noise = rng.uniform([-55.0, -48.0], [55.0, 48.0], size=(n_noise, 2)).astype(np.float32)
    X = np.vstack([*blocks, noise])
    y = np.concatenate([*labels, np.full(n_noise, -1, dtype=np.int64)])
    order = rng.permutation(n)
    return X[order], y[order]


def make_ultrasparse_16d(n: int, *, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Twelve separated anisotropic clusters embedded in 95% uniform noise."""
    rng = np.random.default_rng(seed)
    d = 16
    n_signal = int(round(0.05 * n))
    n_noise = n - n_signal
    counts = np.full(12, n_signal // 12, dtype=int)
    counts[: n_signal % 12] += 1
    centers = np.zeros((12, d), dtype=np.float32)
    grid = [(a, b) for a in (-10.0, -3.3, 3.3, 10.0) for b in (-8.0, 0.0, 8.0)]
    for c, (a, b) in enumerate(grid):
        centers[c, 0] = a
        centers[c, 1] = b
        centers[c, 2:] = rng.normal(0, 3.5, size=d - 2)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    scales = [0.34, 0.50, 0.72]
    for c, count in enumerate(counts):
        diag = np.full(d, scales[c % 3], dtype=np.float32)
        diag[0] *= 1.6
        diag[1] *= 0.8
        pts = rng.normal(size=(count, d)).astype(np.float32) * diag + centers[c]
        blocks.append(pts)
        labels.append(np.full(count, c, dtype=np.int64))
    lo = np.min(centers, axis=0) - 8.0
    hi = np.max(centers, axis=0) + 8.0
    noise = rng.uniform(lo, hi, size=(n_noise, d)).astype(np.float32)
    X = np.vstack([*blocks, noise])
    y = np.concatenate([*labels, np.full(n_noise, -1, dtype=np.int64)])
    order = rng.permutation(n)
    return X[order], y[order]


def make_overlapping_density_16d(
    n: int,
    *,
    seed: int = 42,
    signal_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Eight variable-density overdensities embedded in one continuous background.

    The first four coordinates carry the clustering geometry.  Background points
    have a constant density on that four-dimensional support, while every point
    shares the same twelve-dimensional Gaussian nuisance distribution.  Signal
    populations are local Gaussian overdensities with increasing radii and
    unequal masses.  The diffuse populations deliberately overlap the background
    in k-nearest-neighbour distance, so a global quantile cannot act as an oracle
    signal/background separator.
    """
    if n < 8 * 20:
        raise ValueError("n must leave at least 20 expected points per signal cluster")
    if not 0.0 < signal_fraction < 1.0:
        raise ValueError("signal_fraction must lie strictly between zero and one")

    rng = np.random.default_rng(seed)
    d = 16
    n_signal = int(round(signal_fraction * n))
    n_noise = n - n_signal
    if n_signal < 8 * 20:
        raise ValueError("signal_fraction leaves fewer than 20 points per cluster")

    # Dense rare populations through diffuse populous populations.  Increasing
    # mass only partly offsets the increasing radius, leaving a broad continuum
    # of local density levels rather than eight equally easy Gaussian islands.
    weights = np.array([0.075, 0.0875, 0.10, 0.1125, 0.125, 0.1375, 0.1625, 0.20])
    counts = rng.multinomial(n_signal, weights / weights.sum())
    scales = np.array([0.45, 0.60, 0.80, 1.05, 1.35, 1.70, 2.10, 2.60])
    centers = np.array([
        [-6.0, -6.0, -4.0, 0.0],
        [-6.0,  6.0,  4.0, 0.0],
        [ 6.0, -6.0,  4.0, 0.0],
        [ 6.0,  6.0, -4.0, 0.0],
        [-2.0, -2.0,  5.0, 0.0],
        [-2.0,  2.0, -5.0, 0.0],
        [ 2.0, -2.0, -5.0, 0.0],
        [ 2.0,  2.0,  5.0, 0.0],
    ], dtype=np.float32)

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    informative_shape = np.array([1.0, 0.80, 1.20, 0.90], dtype=np.float32)
    for cluster, (count, scale) in enumerate(zip(counts, scales, strict=True)):
        informative = (
            rng.normal(size=(count, 4)).astype(np.float32)
            * (np.float32(scale) * informative_shape)
            + centers[cluster]
        )
        nuisance = rng.normal(0.0, 1.0, size=(count, d - 4)).astype(np.float32)
        blocks.append(np.column_stack([informative, nuisance]).astype(np.float32))
        labels.append(np.full(count, cluster, dtype=np.int64))

    background = np.column_stack([
        rng.uniform(-10.0, 10.0, size=(n_noise, 4)),
        rng.normal(0.0, 1.0, size=(n_noise, d - 4)),
    ]).astype(np.float32)
    X = np.vstack([*blocks, background])
    y = np.concatenate([*labels, np.full(n_noise, -1, dtype=np.int64)])
    order = rng.permutation(n)
    return X[order], y[order]
