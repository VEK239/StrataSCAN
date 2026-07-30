# StrataSCAN

StrataSCAN 0.2.1 is a variable-density clustering algorithm based on a
full-data multiscale Gamma model and a unified description-length clustering
objective. It adaptively selects Gamma density bases, separates semantic signal
strata from one aggregated background state, optimizes supported core radii,
and retains unsupported points as noise. `StrataSCAN` is the public alias for
the explicit `OptimizationStrataSCAN` estimator.

## Installation

```bash
pip install stratascan
```

For the FAISS HNSW backend used by the high-dimensional benchmark:

```bash
pip install "stratascan[perf]"
```

## Usage

```python
from stratascan import StrataSCAN

labels = StrataSCAN().fit_predict(X)
```

The estimator accepts NumPy-compatible two-dimensional input and follows the
usual `fit`, `fit_predict`, `labels_`, and `n_clusters_` conventions. A
prebuilt graph can be supplied with `fit_from_graph` or
`fit_predict_from_graph`.

## 0.2.1 method

1. Build a 32-neighbour graph. The default uses exact KD-tree search in 2D,
   FAISS HNSW above 2D when FAISS is installed, and brute-force search as the
   dependency-free fallback.
2. Fit full-data local-Poisson Gamma shell profiles at neighbour ranks 4, 8,
   16, and 32.
3. Select the number of density bases with semantic ICL. Search starts at eight
   bases and expands in blocks of four when the optimum is near the boundary,
   up to a hard diagnostic bound of 24.
4. Use the largest fitted log-rate gap to retain a dense signal prefix and
   aggregate all remaining Gamma bases into one semantic background state.
5. Optimize each signal stratum over observed fourth-neighbour radii using
   Gamma posterior log-odds, held-out neighbour-window topology evidence, and
   explicit description-length component costs.
6. Scan the background separately for finitely supported overdensities using
   rank-window and accessible-volume evidence.
7. Form density-ordered, non-merging connected components and attach border
   points only inside a selected core radius.

The released background model is `gamma_components`. Continuous Gamma-rate,
lognormal-rate, and inverse-Gamma-rate backgrounds remain opt-in research
controls. `GammaMDLConfig` and `OptimizationStrictCoreConfig` expose the two
released configuration stages. The 0.1.2
`PredictiveMultiscaleStrataSCAN` class remains available explicitly for
historical reproduction.

## Release evidence

The 0.2.1 release decision and known limitations are recorded in
[`results/published/v0.2.1/RESULTS.md`](results/published/v0.2.1/RESULTS.md).
Its machine-readable defaults are frozen in
[`benchmarks/protocol.v0.2.1-release.json`](benchmarks/protocol.v0.2.1-release.json).
The frozen 0.1.2 evaluation remains in
[`results/published/v0.1.2/RESULTS.md`](results/published/v0.1.2/RESULTS.md).

The 0.2.0 formulation and development evidence remain available in
[`docs/V0.2.0_OPTIMIZATION.md`](docs/V0.2.0_OPTIMIZATION.md),
[`docs/V0.2.0_DEV10_RESULTS.md`](docs/V0.2.0_DEV10_RESULTS.md),
[`docs/V0.2.0_DEV11_VALIDATION.md`](docs/V0.2.0_DEV11_VALIDATION.md), and
[`docs/V0.2.0_DEV12_BACKGROUND_VALIDATION.md`](docs/V0.2.0_DEV12_BACKGROUND_VALIDATION.md).

## Development

```bash
python -m pip install -e ".[dev,perf,benchmark]"
python -m pytest
python -m build
```

Use the benchmark runner with `StrataSCAN` to evaluate the 0.2.1 default. Use
`StrataSCAN-PredictiveMultiscale` when reproducing the 0.1.2 estimator.

### Samusik real-data benchmark

The Samusik workflow downloads the official `Samusik_all` ExperimentHub object,
exports the 10 mouse samples independently, screens all nine methods on sample
01, and then measures per-population stability across samples 02--10 for the
methods that completed screening.

```bash
python scripts/download_samusik.py
Rscript scripts/prepare_samusik_data.R
PYTHONPATH=src python -m benchmarks.run_benchmark --protocol benchmarks/protocol.samusik-01.json --suite cytometry --output-dir results/runs/samusik_01
PYTHONPATH=src python -m benchmarks.run_benchmark --protocol benchmarks/protocol.samusik-all.json --suite cytometry --output-dir results/runs/samusik_all_samples_02_10 --max-workers 3
python scripts/analyze_samusik_stability.py results/runs/samusik_01/results.csv results/runs/samusik_all_samples_02_10/results.csv --output-dir results/samusik_stability
```

## License

MIT
