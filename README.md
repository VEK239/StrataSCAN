# StrataSCAN

![StrataSCAN: neighbourhood shells, density strata, and adaptive cluster extraction](assets/method_overview.png)

**Adaptive density-stratified clustering on sparse kNN graphs.** Finds clusters at different densities and rejects background, without a global radius or a requested cluster count.

**Accepted at [OEDM 2026](https://fan-meng.github.io/ICDM-Workshop/OEDM-26/index.html), a workshop at IEEE ICDM 2026.**

[PyPI](https://pypi.org/project/stratascan/) · [Install](#install) · [Quick start](#quick-start) · [Methods](#method-catalog) · [Experiments](#article-experiments) · [Citation](#citation) · [MIT license](LICENSE)

## Install

Python **3.11+**:

```bash
pip install stratascan
```

Optional dependencies: `pip install "stratascan[perf]"` for FAISS neighbour search, or `pip install "stratascan[baselines]"` for FAISS and Leiden comparators. Pin `stratascan==0.2.4` to use the article software release.

## Quick start

```python
import numpy as np
from stratascan import StrataSCAN

rng = np.random.default_rng(42)
X = np.vstack([
    rng.normal(-3.0, 0.25, size=(160, 4)),
    rng.normal(3.0, 0.25, size=(160, 4)),
    rng.uniform(-8.0, 8.0, size=(180, 4)),
]).astype(np.float32)

model = StrataSCAN(backend="brute").fit(X)
labels = model.labels_  # -1: noise; 0, 1, ...: clusters
print(model.n_clusters_)
```

Pass a finite numeric array `(n_samples, n_features)` with more than 32 observations under the default `k=32`. Scale features before fitting; distances are Euclidean. `fit_predict(X)` returns labels directly.

## How it works

1. **Build one kNN graph.** Neighbourhood shells at ranks 4, 8, 16 and 32 capture local density at several scales.
2. **Separate density strata.** A Gamma mixture groups similar neighbourhood profiles; semantic ICL selects its order.
3. **Extract supported cores.** Within each stratum, select a local radius using density and connectivity evidence minus description cost. Search the initial background for additional supported groups.
4. **Grow from dense to sparse.** Extend existing clusters without merging them; attach border points within selected core radii and leave the remainder unassigned.

[Implementation](src/stratascan/optimization.py) · [Default settings](benchmarks/protocols/release_defaults.json)

<details>
<summary>Algorithm settings and execution</summary>

The Gamma search starts at up to eight bases and expands by four to a bound of 24. A largest-log-rate-gap split selects candidate strata and initial background. Core extraction sweeps fourth-neighbour distances, handling ties together; slots 1–4 construct components and slots 5–8 score connectivity. Components need positive description gain. Background recovery compares candidate groups with background windows and their local surroundings, then repeats after removing accepted groups. The criterion is MDL-inspired; it is not a calibrated significance test or a global optimization guarantee.

`backend="auto"` selects KD-tree for up to two features, FAISS HNSW in higher dimensions if installed, and brute-force otherwise. HNSW is approximate; backend choice can change results. Brute-force is suitable for the small example above and can be expensive on large data. The first fit includes Numba compilation; `n_jobs=1` is the default.

Fitted attributes include `labels_`, `n_clusters_`, `core_sample_indices_`, `stratification_` and `profile_`. `StrataSCAN` aliases `OptimizationStrataSCAN`; configuration classes are `GammaMDLConfig` and `OptimizationStrictCoreConfig`. For `fit_from_graph`, pass `ambient_dimension=X.shape[1]`. There is no out-of-sample `predict`. The inherited `profile_["algorithm_version"]` value `0.2.3` identifies the algorithm family, not the package version.

Density overlap, the mixture search bound and available memory can limit extraction. Background recovery does not guarantee a false-discovery rate.

</details>

## Method catalog

The package also ships independently usable clustering implementations and adapters.

| Method / dispatch ID | Implementation |
| --- | --- |
| `DBSCAN`, `HDBSCAN`, `OPTICS` | scikit-learn adapters with automatic benchmark profiles |
| `SNN-DBSCAN` | Shared nearest-neighbour clustering with a Numba kernel |
| `VDBSCAN-2007` | Variable-density DBSCAN with multiple radii |
| `AMD-DBSCAN` | Adaptive multi-density DBSCAN; stores pairwise distances |
| `kNN-DBSCAN` | Graph-based nearest-neighbour DBSCAN |
| `kNN+Leiden` | kNN graph with igraph/leidenalg communities |
| `X-shift` | Python port using angular geometry; outside the article comparison |
| DPC-kNN | 2016 density-peaks core + 2020 automatic centre selector; fixed `p=0.02` |

```python
from stratascan.baselines import dispatch_baseline, run_dpc_knn_2016

profile = "low_dim" if X.shape[1] <= 2 else "high_dim"
result = dispatch_baseline("DBSCAN", X, profile, seed=42)
labels = result.labels
settings = result.metadata

dpc = run_dpc_knn_2016(X)  # automatic centres; assigns every point
```

These are controlled ports and adapters with fixed label-free profiles. The graph comparators use KD-tree in low dimensions and FAISS HNSW in high dimensions. DPC-kNN uses exact blockwise pairwise distances and has no noise label. See [comparator sources](src/stratascan/baselines/) for implementation details.

## Article experiments

| Experiment | Recorded result for StrataSCAN |
| --- | --- |
| Seven synthetic families, 20,000 points, three seeds | Target F1 **0.828**, discovery **0.827**; completed 21/21 |
| Density contrast, 16–256-fold | Median target F1 **0.829–0.929** across 12 scenarios; completed 36/36 |
| Rare dense targets, up to 99% background | All three targets discovered in every run; median F1 **0.979** at 99% |
| Seven-family scaling | Completed all families through **5 million** points |
| Focused 16-D scaling, 95% background | **22.83 min** at 5 million points, one CPU; comparison timeout 15 h |
| Levine / Mosmann / Nilsson cytometry | Target F1 **0.190 / 0.565 / 0.517** |
| Single-density-layer ablation | Target F1 **0.828 → 0.735**; discovery **0.827 → 0.687** |

The synthetic comparison includes ten methods, including DPC-kNN. Scores use Hungarian target matching; discovery requires purity ≥0.90 and coverage ≥0.10. Read completion counts alongside quality: AMD-DBSCAN's F1 **0.843** covers 12/21 successful cells, while StrataSCAN's **0.828** covers all 21. Timings apply to the recorded datasets and resource budgets.

[Results and scoring](results/README.md) · [Reproduction instructions](benchmarks/README.md) · [Evaluation contract](benchmarks/evaluation_protocol.v1.json)

<details>
<summary>Article result figures</summary>

![Target F1 and discovery across seven synthetic families](assets/synthetic_validation.png)

![Density contrast and rare targets amid background](assets/density_and_noise.png)

![Runtime comparison on the 16-D ultra-sparse family](assets/ultrasparse_scaling.png)

</details>

To run protocols, clone the repository and install its experiment dependencies:

```bash
git clone https://github.com/VEK239/StrataSCAN.git
cd StrataSCAN
python -m pip install -e ".[benchmark,perf]"
python -m benchmarks.run_benchmark --protocol benchmarks/protocols/synthetic_quality.json --suite synthetic --list-jobs
```

Synthetic inputs are generated locally. Cytometry requires [external data preparation](benchmarks/README.md#external-data). The PyPI package contains the library; the checkout contains protocols, results and figures.

## Citation

Article citation details will be added when available. For the software, use [CITATION.cff](CITATION.cff):

```bibtex
@software{stratascan_software,
  author = {{StrataSCAN contributors}},
  title = {StrataSCAN: Adaptive Density-Stratified Clustering and Reference Baselines},
  version = {0.2.4},
  year = {2026},
  url = {https://github.com/VEK239/StrataSCAN/tree/v0.2.4}
}
```

Record the package version or source commit used. When using comparators, also cite their original methodological publications and identify the implementation used here.

## Development

From the checkout, preferably in a virtual environment:

```bash
python -m pip install -e ".[dev,benchmark]"
python -m pytest
python -m build
```

[src/stratascan/](src/stratascan/) — library · [benchmarks/](benchmarks/) — protocols and runner · [results/](results/) — measurements · [tests/](tests/) — tests · [AGENTS.md](AGENTS.md) — guide for coding and research agents
