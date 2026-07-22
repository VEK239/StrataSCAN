# StrataSCAN

StrataSCAN 0.1.0 is a single frozen variable-density clustering algorithm. It
combines Gamma density strata with strict kNN-DBSCAN core seeds and a wider,
core-radius border assignment. Earlier experimental stratifiers, selectors,
fallbacks, vetoes, and structural-recovery paths are not part of the package.

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

## Frozen 0.1.0 method

1. Build a 32-neighbour graph. The default uses exact KD-tree search in 2D,
   FAISS HNSW above 2D when FAISS is installed, and brute-force search as the
   dependency-free fallback.
2. Fit a local-Poisson Gamma mixture to fourth-neighbour volume and collapse
   sparse mixture components into one background tail.
3. Use `q95(d4)` for core detection in supported dense strata.
4. In the tail, test the lowest 2% of `d4 / d32` against 49 equally sized rank
   windows using the largest induced 4-NN component. Activate only at
   empirical `p < 0.05`.
5. A significant tail uses `q5(d4)`. Only the q5 component overlapping the
   significant probe may seed a cluster.
6. Attach border points within `1.25 * core epsilon`.

There is no fallback to scalar kNN-DBSCAN and no post-hoc structural recovery.

## Release evidence

The frozen acceptance scope contains seven synthetic datasets, Levine
cytometry, and 24 Gaia fields. Mosmann and Nilsson are intentionally outside
the 0.1.0 evaluation scope. Results and limitations are recorded in
[`results/published/v0.1.0/RESULTS.md`](results/published/v0.1.0/RESULTS.md).

## Development

```bash
python -m pip install -e ".[dev,perf,benchmark]"
python -m pytest
python -m build
```

The frozen release protocol is
[`benchmarks/protocol.gamma-strict-core.json`](benchmarks/protocol.gamma-strict-core.json).

## License

MIT
