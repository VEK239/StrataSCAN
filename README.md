# StrataSCAN

StrataSCAN 0.1.2 is a variable-density clustering algorithm. The released
`PredictiveMultiscaleStrataSCAN` estimator fits a constrained four-shell Gamma
mixture, selects its complexity by repeated-holdout predictive likelihood, and
combines the resulting density strata with strict kNN-DBSCAN core seeds and a
wider, core-radius border assignment. `StrataSCAN` is the short public alias for
the same estimator.

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
from stratascan import PredictiveMultiscaleStrataSCAN

labels = PredictiveMultiscaleStrataSCAN().fit_predict(X)
```

The estimator accepts NumPy-compatible two-dimensional input and follows the
usual `fit`, `fit_predict`, `labels_`, and `n_clusters_` conventions. A
prebuilt graph can be supplied with `fit_from_graph` or
`fit_predict_from_graph`.

## 0.1.2 method

1. Build a 32-neighbour graph. The default uses exact KD-tree search in 2D,
   FAISS HNSW above 2D when FAISS is installed, and brute-force search as the
   dependency-free fallback.
2. Fit constrained local-Poisson Gamma rate profiles to the shell-volume
   increments at neighbour ranks 4, 8, 16, and 32.
3. Select 2--8 mixture components using three repeated 80/20 holdouts. Choose
   the smallest stable model within one standard error of the best predictive
   log likelihood, then refit it on up to 10,000 representative observations.
4. Use `q95(d4)` for core detection in supported dense strata.
5. In the tail, test the lowest 2% of `d4 / d32` against 49 equally sized rank
   windows using the largest induced 4-NN component. Activate only at
   empirical `p < 0.05`.
6. A significant tail uses `q5(d4)`. Only the q5 component overlapping the
   significant probe may seed a cluster.
7. Attach border points within `1.25 * core epsilon`.

There is no fallback to scalar kNN-DBSCAN and no post-hoc structural recovery.
`PredictiveMultiscaleConfig` exposes the predictive mixture selection settings
when non-default configuration is required.

## Release evidence

The frozen 0.1.2 evaluation is recorded in
[`results/published/v0.1.2/RESULTS.md`](results/published/v0.1.2/RESULTS.md).
Earlier release evidence remains available under `results/published`.

## Development

```bash
python -m pip install -e ".[dev,perf,benchmark]"
python -m pytest
python -m build
```

Use the benchmark runner with `StrataSCAN` to evaluate the 0.1.2 default.

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
