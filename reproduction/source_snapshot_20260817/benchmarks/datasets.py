from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stratascan.synthetic import (
    make_density_contrast,
    make_multidensity_2d,
    make_overlapping_density_16d,
    make_ultrasparse_16d,
)


@dataclass(slots=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    target_names: list[str]
    metadata: dict[str, Any]


def _robust_scale(X: np.ndarray) -> np.ndarray:
    median = np.nanmedian(X, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(X - median), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return np.asarray((X - median) / scale, dtype=np.float32, order="C")


def _shuffle(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(X.shape[0])
    return X[order], y[order]


def _add_uniform_background(
    signal: np.ndarray,
    labels: np.ndarray,
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_noise = n - signal.shape[0]
    if n_noise < 0:
        raise ValueError("signal contains more rows than requested dataset size")
    if n_noise:
        lo = np.min(signal, axis=0) - 2.5
        hi = np.max(signal, axis=0) + 2.5
        noise = rng.uniform(lo, hi, size=(n_noise, signal.shape[1])).astype(np.float32)
        X = np.vstack([signal, noise])
        y = np.concatenate([labels, np.full(n_noise, -1, dtype=np.int64)])
    else:
        X, y = signal, labels
    return _shuffle(X, y, seed)


def _make_moons_or_rings(spec: dict[str, Any], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_circles, make_moons

    noise_fraction = float(spec["noise_fraction"])
    n_signal = int(round(n * (1.0 - noise_fraction)))
    if n_signal < 10:
        raise ValueError("synthetic signal is too small")
    if spec["family"] == "moons":
        signal, labels = make_moons(n_samples=n_signal, noise=0.08, random_state=seed)
    else:
        signal, labels = make_circles(
            n_samples=n_signal, factor=0.35, noise=0.04, random_state=seed
        )
    return _add_uniform_background(
        signal.astype(np.float32), labels.astype(np.int64), n=n, seed=seed
    )


def _make_gaussian(spec: dict[str, Any], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dimension = int(spec["dimension"])
    noise_fraction = float(spec["noise_fraction"])
    n_signal = int(round(n * (1.0 - noise_fraction)))
    n_clusters = 6 if spec["family"] == "gaussian_overlap" else 8
    if n_signal < n_clusters * 20:
        raise ValueError("every synthetic truth cluster must contain at least 20 points")
    if spec["family"] == "gaussian_imbalanced":
        weights = np.geomspace(1.0, 10.0, n_clusters)
    else:
        weights = np.ones(n_clusters)
    counts = rng.multinomial(n_signal, weights / weights.sum())
    centers = rng.normal(0.0, 4.0, size=(n_clusters, dimension))
    if spec["family"] == "gaussian_overlap":
        centers *= 0.55
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for cluster, count in enumerate(counts):
        scales = rng.uniform(0.35, 1.1, size=dimension)
        if spec["family"] == "gaussian_overlap":
            scales *= 1.45
        block = rng.normal(size=(count, dimension)) * scales + centers[cluster]
        blocks.append(block.astype(np.float32))
        labels.append(np.full(count, cluster, dtype=np.int64))
    return _add_uniform_background(
        np.vstack(blocks), np.concatenate(labels), n=n, seed=seed
    )


def _make_global_contamination_invariance(
    spec: dict[str, Any], n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fixed targets plus constant-intensity background on expanding support.

    This family isolates *global* contamination burden. Increasing the declared
    noise fraction adds points to two remote strips at a fixed areal intensity;
    it does not shrink the targets, move them, or increase background density
    near them. Background points form a deterministic prefix across fractions
    for a given seed, which supports paired trajectory checks without selecting
    favorable random realizations.
    """

    if int(spec.get("dimension", 2)) != 2:
        raise ValueError("global_contamination_invariance is defined only in 2D")
    signal_size = int(spec["signal_size"])
    if signal_size < 3 * 40:
        raise ValueError("signal_size must provide at least 40 points per target")
    noise_fraction = float(spec["noise_fraction"])
    if not 0.0 < noise_fraction < 1.0:
        raise ValueError("noise_fraction must lie strictly between zero and one")
    expected_n = signal_size + int(round(signal_size * noise_fraction / (1.0 - noise_fraction)))
    if n != expected_n:
        raise ValueError(
            "global_contamination_invariance requires n consistent with "
            f"signal_size and noise_fraction; expected {expected_n}, found {n}"
        )

    intensity = float(spec.get("background_intensity", 12.0))
    strip_height = float(spec.get("background_strip_height", 10.0))
    inner_edge = float(spec.get("background_inner_edge", 6.0))
    if not np.isfinite(intensity) or intensity <= 0.0:
        raise ValueError("background_intensity must be positive and finite")
    if not np.isfinite(strip_height) or strip_height <= 0.0:
        raise ValueError("background_strip_height must be positive and finite")
    if not np.isfinite(inner_edge) or inner_edge < 5.0:
        raise ValueError("background_inner_edge must be finite and at least 5")

    signal_seed, background_seed, shuffle_seed = np.random.SeedSequence(seed).spawn(3)
    signal_rng = np.random.default_rng(signal_seed)
    centers = np.array([[-3.0, -2.0], [3.0, -2.0], [0.0, 3.0]], dtype=np.float64)
    covariance = np.array([[0.35, 0.08], [0.08, 0.22]], dtype=np.float64)
    counts = np.full(3, signal_size // 3, dtype=np.int64)
    counts[: signal_size % 3] += 1
    blocks = [
        signal_rng.multivariate_normal(center, covariance, int(count)).astype(np.float32)
        for center, count in zip(centers, counts, strict=True)
    ]
    labels = [np.full(int(count), index, dtype=np.int64) for index, count in enumerate(counts)]

    n_background = n - signal_size
    # Alternate between the right and left strips.  Position j occupies one
    # intensity-normalized slice, so prefixes retain the same nominal density.
    background_rng = np.random.default_rng(background_seed)
    uniforms = background_rng.random((n_background, 2))
    index = np.arange(n_background, dtype=np.int64)
    within_side = index // 2
    side = np.where(index % 2 == 0, 1.0, -1.0)
    x = side * (
        inner_edge + (within_side + uniforms[:, 0]) / (intensity * strip_height)
    )
    y = (uniforms[:, 1] - 0.5) * strip_height
    background = np.column_stack([x, y]).astype(np.float32)

    X = np.vstack([*blocks, background])
    truth = np.concatenate([*labels, np.full(n_background, -1, dtype=np.int64)])
    order = np.random.default_rng(shuffle_seed).permutation(n)
    support_area = n_background / intensity
    metadata = {
        "signal_size": signal_size,
        "background_size": n_background,
        "declared_noise_fraction": noise_fraction,
        "realized_noise_fraction": n_background / n,
        "background_intensity": intensity,
        "background_support_area": support_area,
        "background_strip_height": strip_height,
        "background_inner_edge": inner_edge,
        "n_targets": 3,
        "controlled_factor": "global_background_support_at_fixed_local_intensity",
        "paired_background": "seeded_prefix",
    }
    return X[order], truth[order], metadata


def load_synthetic(spec: dict[str, Any], seed: int) -> Dataset:
    n = int(spec["n"])
    family = spec["family"]
    family_metadata: dict[str, Any] = {}
    if family == "multidensity":
        X, y = make_multidensity_2d(n, seed=seed)
    elif family == "ultrasparse":
        X, y = make_ultrasparse_16d(n, seed=seed)
    elif family == "overlapping_density":
        X, y = make_overlapping_density_16d(
            n,
            seed=seed,
            signal_fraction=float(spec.get("signal_fraction", 0.20)),
        )
    elif family == "density_contrast":
        X, y = make_density_contrast(
            n,
            seed=seed,
            dimension=int(spec["dimension"]),
            density_ratio=float(spec["density_ratio"]),
            shape=str(spec.get("shape", "gaussian")),
            noise_fraction=float(spec.get("noise_fraction", 0.50)),
            n_clusters=int(spec.get("n_clusters", 6)),
        )
    elif family in {"moons", "rings"}:
        X, y = _make_moons_or_rings(spec, n, seed)
    elif family in {"gaussian_overlap", "gaussian_imbalanced"}:
        X, y = _make_gaussian(spec, n, seed)
    elif family == "global_contamination_invariance":
        X, y, family_metadata = _make_global_contamination_invariance(spec, n, seed)
    else:
        raise ValueError(f"unknown synthetic family: {family}")
    names = [f"cluster_{value}" for value in sorted(np.unique(y[y >= 0]).tolist())]
    return Dataset(
        np.asarray(X, dtype=np.float32, order="C"),
        np.asarray(y, dtype=np.int64),
        names,
        {"family": family, "n": n, "dimension": int(X.shape[1]), **family_metadata},
    )


def load_cytometry(spec: dict[str, Any], repo: Path) -> Dataset:
    import pandas as pd

    features = pd.read_csv(repo / spec["features"])
    feature_metadata = pd.read_csv(repo / spec["feature_metadata"])
    metadata = pd.read_csv(repo / spec["metadata"])
    if not np.array_equal(features["event_id"].to_numpy(), metadata["event_id"].to_numpy()):
        raise ValueError("cytometry features and metadata event_id columns do not align")
    labels = metadata[spec["label_column"]].fillna("unlabeled").astype(str).to_numpy()
    backgrounds = set(map(str, spec["background_labels"]))
    targets = sorted(set(labels) - backgrounds)
    mapping = {name: index for index, name in enumerate(targets)}
    y = np.array([mapping.get(label, -1) for label in labels], dtype=np.int64)
    selected_features = feature_metadata.loc[
        feature_metadata["marker_class"].isin(spec["feature_classes"]), "feature"
    ].astype(str).tolist()
    missing_features = sorted(set(selected_features) - set(features.columns))
    if not selected_features or missing_features:
        raise ValueError(
            f"invalid cytometry feature selection; selected={len(selected_features)}, "
            f"missing={missing_features}"
        )
    X = features[selected_features].to_numpy(dtype=np.float32)
    transform = spec["transform"]
    if transform["name"] != "arcsinh" or float(transform["cofactor"]) <= 0:
        raise ValueError(f"unsupported cytometry transform: {transform}")
    X = np.arcsinh(X / float(transform["cofactor"]))
    X = _robust_scale(X)
    if not np.all(np.isfinite(X)):
        raise ValueError("cytometry preprocessing produced non-finite values")
    return Dataset(
        X,
        y,
        targets,
        {
            "dataset_id": spec["id"],
            "n": int(X.shape[0]),
            "dimension": int(X.shape[1]),
            "background_labels": sorted(backgrounds),
            "selected_features": selected_features,
            "transform": transform,
        },
    )


def gaia_field_ids(data_root: Path) -> list[str]:
    fields = data_root / "raw_open_cluster_fields" / "gaia_dr3_cone_fields"
    available = {path.stem.removeprefix("gaia_cone_") for path in fields.glob("gaia_cone_*.csv")}
    import pandas as pd

    table1 = pd.read_csv(data_root / "benchmark_reference_tables" / "ocfinder_table1.csv")
    table2 = pd.read_csv(data_root / "benchmark_reference_tables" / "ocfinder_table2.csv")
    referenced = set(table1["Cluster"].astype(str)) & set(table2["Cluster"].astype(str))
    return sorted(available & referenced)


def load_gaia(spec: dict[str, Any], repo: Path) -> Dataset:
    import pandas as pd

    root = repo / spec["data_root"]
    field_id = str(spec["field_id"])
    field = pd.read_csv(
        root / "raw_open_cluster_fields" / "gaia_dr3_cone_fields" / f"gaia_cone_{field_id}.csv"
    )
    table1 = pd.read_csv(root / "benchmark_reference_tables" / "ocfinder_table1.csv")
    table2 = pd.read_csv(root / "benchmark_reference_tables" / "ocfinder_table2.csv")
    columns = ["ra", "dec", "parallax", "pmra", "pmdec", "ruwe", "source_id"]
    valid = np.all(np.isfinite(field[columns].to_numpy(dtype=np.float64)), axis=1)
    valid &= field["ruwe"].to_numpy(dtype=float) <= float(spec["ruwe_max"])
    field = field.loc[valid].reset_index(drop=True)
    if field.shape[0] < 33:
        raise ValueError(f"Gaia field {field_id} has fewer than 33 usable rows")

    ra = field["ra"].to_numpy(dtype=float)
    dec = field["dec"].to_numpy(dtype=float)
    parallax = field["parallax"].to_numpy(dtype=float)
    pmra = field["pmra"].to_numpy(dtype=float)
    pmdec = field["pmdec"].to_numpy(dtype=float)
    profile = spec["preprocessing"]
    if profile == "unsupervised_field":
        ra0, dec0 = float(np.median(ra)), float(np.median(dec))
        raw = np.column_stack([
            (ra - ra0) * np.cos(np.deg2rad(dec0)), dec - dec0, parallax, pmra, pmdec
        ])
        X = _robust_scale(raw)
    elif profile == "catalog_guided_recovery":
        row = table1.loc[table1["Cluster"].astype(str) == field_id]
        if row.empty:
            raise ValueError(f"Gaia field {field_id} is missing from table1")
        ref = row.iloc[0]
        ra0, dec0 = float(ref["RA_ICRS"]), float(ref["DE_ICRS"])
        raw = np.column_stack([
            (ra - ra0) * np.cos(np.deg2rad(dec0)),
            dec - dec0,
            parallax - float(ref["plx"]),
            pmra - float(ref["pmRA"]),
            pmdec - float(ref["pmDE"]),
        ])
        X = _robust_scale(raw)
    else:
        raise ValueError(f"unknown Gaia preprocessing profile: {profile}")

    known = set(
        table2.loc[table2["Cluster"].astype(str) == field_id, "GaiaEDR3"]
        .dropna()
        .astype(np.int64)
        .tolist()
    )
    y = np.array(
        [0 if int(source_id) in known else -1 for source_id in field["source_id"]],
        dtype=np.int64,
    )
    if not np.any(y >= 0):
        raise ValueError(f"Gaia field {field_id} has no reference members after filtering")
    return Dataset(
        X,
        y,
        [field_id],
        {
            "field_id": field_id,
            "preprocessing": profile,
            "n": int(X.shape[0]),
            "dimension": int(X.shape[1]),
            "reference_members": int(np.sum(y >= 0)),
        },
    )


def load_dataset(job: dict[str, Any], repo: Path) -> Dataset:
    suite = job["suite"]
    if suite == "synthetic":
        return load_synthetic(job["dataset"], int(job["seed"]))
    if suite == "cytometry":
        return load_cytometry(job["dataset"], repo)
    if suite == "gaia":
        return load_gaia(job["dataset"], repo)
    raise ValueError(f"unknown benchmark suite: {suite}")
