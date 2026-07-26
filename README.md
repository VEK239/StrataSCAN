# StrataSCAN

StrataSCAN 0.1.1 is a variable-density clustering algorithm. It combines a
four-shell local-Poisson Gamma stratifier with strict kNN-DBSCAN core seeds and
a wider, core-radius border assignment. The earlier standalone multiscale
estimator is now the public `StrataSCAN` algorithm.

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

## 0.1.1 method

1. Build a 32-neighbour graph. The default uses exact KD-tree search in 2D,
   FAISS HNSW above 2D when FAISS is installed, and brute-force search as the
   dependency-free fallback.
2. Fit a local-Poisson Gamma mixture to the volume increments at neighbour
   ranks 4, 8, 16, and 32. The fitting diagnostics use `log(d4)` and anchored
   ratios `log(d4/d8)`, `log(d4/d16)`, and `log(d4/d32)`.
3. Use `q95(d4)` for core detection in supported dense strata.
4. In the tail, test the lowest 2% of `d4 / d32` against 49 equally sized rank
   windows using the largest induced 4-NN component. Activate only at
   empirical `p < 0.05`.
5. A significant tail uses `q5(d4)`. Only the q5 component overlapping the
   significant probe may seed a cluster.
6. Attach border points within `1.25 * core epsilon`.

There is no fallback to scalar kNN-DBSCAN and no post-hoc structural recovery.
`MultiscaleConfig` exposes the multiscale mixture settings when non-default
configuration is required.

## Release evidence

The 0.1.0 frozen evidence is retained as an archival baseline in
[`results/published/v0.1.0/RESULTS.md`](results/published/v0.1.0/RESULTS.md).
It does not evaluate the new 0.1.1 multiscale default.

## Development

```bash
python -m pip install -e ".[dev,perf,benchmark]"
python -m pytest
python -m build
```

Use the benchmark runner with `StrataSCAN` to evaluate the 0.1.1 default.

## License

MIT
