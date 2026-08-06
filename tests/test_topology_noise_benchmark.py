from __future__ import annotations

import numpy as np

from topology_noise_benchmark.generators import generate
from topology_noise_benchmark.run_benchmark import build_jobs


def test_all_topologies_are_reproducible_and_auditable() -> None:
    for topology, ratio in (("compact_gaussian", 64.0), ("anisotropic_ellipses", 1.0), ("moon_arcs", 1.0), ("rings", 1.0)):
        first = generate(topology=topology, density_ratio=ratio, noise_fraction=.99, geometry_seed=3571, sampling_seed=42)
        second = generate(topology=topology, density_ratio=ratio, noise_fraction=.99, geometry_seed=3571, sampling_seed=42)
        assert np.array_equal(first.X, second.X)
        assert first.X.shape == (180_000, 2)
        assert np.sum(first.y >= 0) == 1_800
        assert first.metadata["minimum_local_target_background_contrast"] >= 50.0
        assert first.metadata["minimum_edge_gap"] > 0.0


def test_sampling_seed_changes_draws_not_geometry() -> None:
    first = generate(topology="rings", density_ratio=1.0, noise_fraction=.95, geometry_seed=3571, sampling_seed=23)
    second = generate(topology="rings", density_ratio=1.0, noise_fraction=.95, geometry_seed=3571, sampling_seed=42)
    assert first.metadata["geometry_instance_id"] == second.metadata["geometry_instance_id"]
    assert not np.array_equal(first.X, second.X)


def test_manifest_has_four_standard_seeds_and_amd_skips() -> None:
    config = {"geometry_seed": 3571, "sampling_seeds": [23, 42, 73, 151], "methods": ["StrataSCAN", "AMD-DBSCAN"], "design": {"target_count": 6, "mass_per_target": 300, "minimum_local_contrast_at_99pct": 50.0}, "cells": [{"cell_id": "a", "topology": "compact_gaussian", "density_ratio": 4.0, "noise_fractions": [.5, .99]}]}
    jobs = build_jobs(config)
    assert len(jobs) == 16
    assert {job["sampling_seed"] for job in jobs} == {23, 42, 73, 151}
    assert sum(job["method"] == "AMD-DBSCAN" for job in jobs) == 8
