from __future__ import annotations

import numpy as np


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
