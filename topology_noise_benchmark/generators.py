from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


NOISE_FRACTIONS = {0.50, 0.75, 0.90, 0.95, 0.975, 0.99}


@dataclass(frozen=True, slots=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any]


def _background_count(signal_count: int, fraction: float) -> int:
    return int(round(signal_count * fraction / (1.0 - fraction)))


def _centers(seed: int, radius: float = 12.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    angles = angle + np.arange(6) * 2.0 * np.pi / 6.0
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def fixed_geometry(*, topology: str, density_ratio: float, geometry_seed: int, mass_per_target: int = 300, minimum_local_contrast_at_99pct: float = 50.0) -> dict[str, Any]:
    if topology not in {"compact_gaussian", "anisotropic_ellipses", "moon_arcs", "rings"}:
        raise ValueError(f"unknown topology: {topology}")
    if density_ratio < 1.0:
        raise ValueError("density_ratio must be at least one")
    signal_count = 6 * mass_per_target
    sparse_sigma = 0.36
    # All topologies share the support calibrated from this least-dense compact
    # reference.  Their local density must therefore be >= this lower bound.
    reference_minimum_density = mass_per_target / (2.0 * np.pi * sparse_sigma**2)
    max_background = _background_count(signal_count, 0.99)
    window_area = max_background / (reference_minimum_density / minimum_local_contrast_at_99pct)
    rng = np.random.default_rng(geometry_seed)
    centers = _centers(geometry_seed)
    payload = {"topology": topology, "density_ratio": density_ratio, "geometry_seed": geometry_seed, "mass_per_target": mass_per_target, "window_area": window_area}
    geometry: dict[str, Any] = {"centers": centers, "window_area": float(window_area), "window_side": float(np.sqrt(window_area)), "reference_minimum_density": float(reference_minimum_density)}
    if topology == "compact_gaussian":
        sigmas = np.geomspace(sparse_sigma / np.sqrt(density_ratio), sparse_sigma, 6)
        geometry.update({"sigmas": sigmas, "local_peak_densities": (mass_per_target / (2.0 * np.pi * sigmas**2)).tolist(), "support_radii": (5.0 * sigmas).tolist()})
    elif topology == "anisotropic_ellipses":
        aspect = 8.0
        major, minor = sparse_sigma * np.sqrt(aspect), sparse_sigma / np.sqrt(aspect)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=6)
        geometry.update({"major_sigma": major, "minor_sigma": minor, "orientations": angles, "local_peak_densities": [reference_minimum_density] * 6, "support_radii": [5.0 * major] * 6, "aspect_ratio": aspect})
    elif topology == "moon_arcs":
        curve_radius, tube_sigma = 0.50, 0.08
        length = np.pi * curve_radius
        local_density = mass_per_target / (length * np.sqrt(2.0 * np.pi) * tube_sigma)
        geometry.update({"curve_radius": curve_radius, "tube_sigma": tube_sigma, "orientations": rng.uniform(0.0, 2.0 * np.pi, size=6), "curve_length": length, "local_peak_densities": [local_density] * 6, "support_radii": [curve_radius + 5.0 * tube_sigma] * 6})
    else:
        ring_radius, tube_sigma = 0.50, 0.08
        length = 2.0 * np.pi * ring_radius
        local_density = mass_per_target / (length * np.sqrt(2.0 * np.pi) * tube_sigma)
        geometry.update({"ring_radius": ring_radius, "tube_sigma": tube_sigma, "curve_length": length, "local_peak_densities": [local_density] * 6, "support_radii": [ring_radius + 5.0 * tube_sigma] * 6})
    center_distances = [np.linalg.norm(centers[a] - centers[b]) for a in range(6) for b in range(a + 1, 6)]
    geometry["minimum_center_distance"] = float(min(center_distances))
    geometry["minimum_edge_gap"] = float(min(center_distances) - 2.0 * max(geometry["support_radii"]))
    geometry["geometry_instance_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    if geometry["minimum_edge_gap"] <= 0.0:
        raise ValueError("topology supports overlap")
    return geometry


def _sample_targets(topology: str, geometry: dict[str, Any], rng: np.random.Generator, mass: int) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    for label, center in enumerate(geometry["centers"]):
        if topology == "compact_gaussian":
            block = rng.normal(size=(mass, 2)) * geometry["sigmas"][label] + center
        elif topology == "anisotropic_ellipses":
            angle = geometry["orientations"][label]
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            block = rng.normal(size=(mass, 2)) * np.array([geometry["major_sigma"], geometry["minor_sigma"]])
            block = block @ rotation.T + center
        elif topology == "moon_arcs":
            theta = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=mass)
            arc = np.column_stack((geometry["curve_radius"] * np.cos(theta), geometry["curve_radius"] * np.sin(theta)))
            normal = np.column_stack((-np.cos(theta), -np.sin(theta))) * rng.normal(0.0, geometry["tube_sigma"], size=(mass, 1))
            angle = geometry["orientations"][label]
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            block = (arc + normal) @ rotation.T + center
        else:
            theta = rng.uniform(0.0, 2.0 * np.pi, size=mass)
            radius = geometry["ring_radius"] + rng.normal(0.0, geometry["tube_sigma"], size=mass)
            block = np.column_stack((radius * np.cos(theta), radius * np.sin(theta))) + center
        blocks.append(block)
    return blocks


def generate(*, topology: str, density_ratio: float, noise_fraction: float, geometry_seed: int, sampling_seed: int, mass_per_target: int = 300, minimum_local_contrast_at_99pct: float = 50.0) -> Dataset:
    if noise_fraction not in NOISE_FRACTIONS:
        raise ValueError("noise_fraction must be a prespecified sweep level")
    geometry = fixed_geometry(topology=topology, density_ratio=density_ratio, geometry_seed=geometry_seed, mass_per_target=mass_per_target, minimum_local_contrast_at_99pct=minimum_local_contrast_at_99pct)
    signal_count = 6 * mass_per_target
    background_count = _background_count(signal_count, noise_fraction)
    rng = np.random.default_rng(sampling_seed)
    blocks = _sample_targets(topology, geometry, rng, mass_per_target)
    background = rng.uniform(-geometry["window_side"] / 2.0, geometry["window_side"] / 2.0, size=(background_count, 2))
    X = np.vstack([*blocks, background]).astype(np.float64, copy=False)
    y = np.concatenate([*[np.full(mass_per_target, label, dtype=np.int64) for label in range(6)], np.full(background_count, -1, dtype=np.int64)])
    order = rng.permutation(X.shape[0])
    background_density = background_count / geometry["window_area"]
    local_densities = np.asarray(geometry["local_peak_densities"], dtype=float)
    metadata = {
        "generator": "replicated_topology_noise_v1", "topology": topology, "density_ratio_requested": density_ratio,
        "geometry_seed": geometry_seed, "sampling_seed": sampling_seed, "geometry_instance_id": geometry["geometry_instance_id"],
        "noise_fraction_requested": noise_fraction, "noise_fraction_realized": background_count / X.shape[0],
        "target_count": 6, "mass_per_target": mass_per_target, "cluster_masses": [mass_per_target] * 6,
        "signal_count": signal_count, "background_count": background_count, "total_n": int(X.shape[0]),
        "background_support": "fixed axis-aligned square", "background_window_area": geometry["window_area"], "background_window_side": geometry["window_side"], "background_density": float(background_density),
        "local_target_density_definition": "Gaussian peak for compact/elliptical targets; central tube density for arcs/rings",
        "local_target_peak_or_tube_densities": local_densities.tolist(), "local_target_background_contrasts": (local_densities / background_density).tolist(),
        "minimum_local_target_background_contrast": float(np.min(local_densities / background_density)),
        "maximum_local_target_background_contrast": float(np.max(local_densities / background_density)),
        "minimum_center_distance": geometry["minimum_center_distance"], "minimum_edge_gap": geometry["minimum_edge_gap"],
        "support_radii": geometry["support_radii"], "nested_background_prefix": True,
    }
    for key in ("sigmas", "major_sigma", "minor_sigma", "aspect_ratio", "curve_radius", "ring_radius", "tube_sigma", "curve_length", "orientations"):
        if key in geometry: metadata[key] = np.asarray(geometry[key]).tolist() if isinstance(geometry[key], np.ndarray) else geometry[key]
    return Dataset(X=X[order], y=y[order], metadata=metadata)
