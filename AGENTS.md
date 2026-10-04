# Guide for coding and research agents

This repository provides an installable Python clustering library plus executable article protocols and stored evidence. Read README.md for the method catalog and CITATION.cff for software citation metadata.

## Find the implementation

- Public estimator: `from stratascan import StrataSCAN`; implementation in `src/stratascan/optimization.py`.
- Common comparators: `from stratascan.baselines import dispatch_baseline`; implementation in `src/stratascan/baselines/algorithms.py`.
- DPC-kNN hybrid: `from stratascan.baselines import run_dpc_knn_2016`; implementation in `src/stratascan/baselines/dpc_knn.py`.
- Graph construction: `src/stratascan/neighbors.py`; Gamma fitting: `gamma_fit.py`; density profiles: `density_profiles.py`; shared graph components: `graph_components.py`.
- Experiment dispatch: `benchmarks/methods.py`; data loading: `benchmarks/datasets.py`; scoring: `benchmarks/evaluation.py` and `benchmarks/evaluation_protocol.v1.json`.
- Requested matrices: `benchmarks/protocols/`; per-cell measurements: `results/*.csv`; aggregate scores: `results/summaries/`.

## Install and use

Use Python >=3.11. From the checkout root, install `python -m pip install -e .` for the estimator and core comparators. Install `python -m pip install -e ".[baselines]"` for optional graph/Leiden comparator dependencies. Article runs need `python -m pip install -e ".[benchmark,perf]"`; test/build tools use `[dev]`.

`dispatch_baseline` accepts DBSCAN, HDBSCAN, OPTICS, SNN-DBSCAN, VDBSCAN-2007, AMD-DBSCAN, kNN-DBSCAN, kNN+Leiden and X-shift. Pass `profile="low_dim"` for <=2 features or `"high_dim"` otherwise. Results expose `.labels` and `.metadata`. DPC-kNN uses its separate function, fixed p=0.02 and automatic centre selection; it assigns all points and has no noise label. StrataSCAN exposes `.labels_` and `.profile_`, with -1 for rejected observations. It has no out-of-sample predict method. Scale features before fitting.

For experiment inspection, run `python -m benchmarks.run_benchmark --protocol benchmarks/protocols/synthetic_quality.json --suite synthetic --list-jobs`. Select a listed identifier with `--job-id ID` and a new output directory. Read benchmarks/README.md before executing an expensive matrix. Biological input exports are external; their required schema is documented there.

## Preserve the scientific contract

This publication branch is assembled from existing implementations. Preserve computational behavior when reorganizing code; do not silently replace an algorithm, tune its profile against reference labels, or change the scoring contract. Keep timeout/error/memory-limit statuses and missing quality; successful-run means have conditional denominators. Do not equate package version 0.2.4 with the inherited algorithm-family profile identifier 0.2.3. X-shift is shipped but outside the article matrix. Cite comparator implementation boundaries accurately, especially the DPC hybrid and controlled ports.

Use existing tests appropriate to a change: `python -m pytest`. Check packaging with `python -m build` when imports/layout/dependencies change. Compare outputs before and after any computational refactor. Keep generated runs, raw external data and caches out of tracked files. Documentation should describe observed capabilities and resource budgets; stored results do not imply new experiments or universal performance guarantees.

## Citation and discovery

Use CITATION.cff for software metadata and report the exact commit actually used. The package currently attributes the software to StrataSCAN contributors. The manuscript source is anonymous; do not infer author identities, add an unverified article DOI, or claim a publication status from the branch name. Give the original methodological references credit when using comparator methods.
