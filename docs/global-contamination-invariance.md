# Global-contamination invariance benchmark

`benchmarks/protocol.v0.2.4-target-discovery-v1-global-contamination.json`
defines a controlled positive benchmark for the released StrataSCAN defaults.
It asks a narrow question: when recoverable targets and their local environment
do not change, does adding more remote, low-density background alter target
discovery?

## Controlled construction

Every cell contains the same three anisotropic Gaussian targets and exactly
1,200 signal observations. Within a seed, target coordinates are identical at
all noise fractions. Known synthetic background is placed in two strips beyond
`|x| = 6`. The strips have a fixed height of 10 and a fixed nominal intensity
of 12 observations per square unit. A larger noise fraction therefore expands
the background support rather than increasing its local density. Background
coordinates are a seeded prefix, so the paired trajectory adds observations
without replacing earlier ones.

The six fractions (25%, 50%, 75%, 90%, 95%, and 99%) and five seeds (211, 223,
227, 229, and 233) are frozen before the declared-seed evaluation. The 99%
endpoint contains 1,200 signal and 118,800 background observations (120,000
total). All 30 cells must be retained; seeds may not be dropped or replaced.

This construction intentionally does **not** claim robustness to background
that becomes denser around or overlaps a target. The existing fixed-total-size
moons noise sweep changes local contamination while shrinking signal and is
retained as complementary negative/stress evidence.

## Evaluation and interpretation

The frozen `target-discovery-v1` evaluator uses one-to-one Hungarian matching,
macro target aggregation, and known-synthetic-noise semantics. A cell discovers
a target only when the matched cluster has at least 0.90 purity and 0.10
coverage. Before running the declared seeds, “good” was fixed as:

- every job completes;
- every target is discovered in every cell;
- macro target F1 is at least 0.80 in every cell; and
- known-noise evidence F1 is at least 0.80 in every cell.

These are minimum-cell criteria, not averages, so a favorable seed cannot hide
a failure. Passing supports only global-contamination invariance over the stated
geometry, intensity, fractions, sizes, and defaults.

Run or inspect the matrix with:

```text
python -m benchmarks.run_benchmark --protocol benchmarks/protocol.v0.2.4-target-discovery-v1-global-contamination.json --evaluation-protocol benchmarks/evaluation_protocol.v1.json --suite synthetic --list-jobs
python -m benchmarks.run_benchmark --protocol benchmarks/protocol.v0.2.4-target-discovery-v1-global-contamination.json --evaluation-protocol benchmarks/evaluation_protocol.v1.json --suite synthetic --output-dir results/runs/v0.2.4-global-contamination-invariance
```

## Completed five-seed result

The frozen 30-cell matrix was executed locally on 2026-08-17 in
`results/runs/v0.2.4-global-contamination-invariance-full`. All 30 jobs
completed successfully and the run validator reported an exact 30/30 matrix
with no missing, duplicate, malformed, or invalid evaluation artifacts.

| Noise | Total rows | Completed | Minimum target F1 | Median target F1 | Minimum discovery | Minimum noise F1 | Median runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 1,600 | 5/5 | 0.927 | 0.948 | 1.000 | 0.831 | 0.529 |
| 50% | 2,400 | 5/5 | 0.907 | 0.913 | 1.000 | 0.921 | 0.507 |
| 75% | 4,800 | 5/5 | 0.911 | 0.920 | 1.000 | 0.972 | 0.642 |
| 90% | 12,000 | 5/5 | 0.965 | 0.982 | 1.000 | 0.997 | 0.799 |
| 95% | 24,000 | 5/5 | 0.941 | 0.977 | 1.000 | 0.997 | 1.007 |
| 99% | 120,000 | 5/5 | 0.929 | 0.935 | 1.000 | 0.999 | 3.971 |

Thus every cell passed the criteria frozen above. The narrow supported result
is that the unchanged 0.2.4 defaults are invariant to the declared increase in
remote global contamination when target size, target geometry, and local
background intensity are controlled. This does not weaken or supersede the
existing fixed-support noise result, where increasing local contamination
causes target recovery to fail.

For integrity checks, the SHA-256 values are:

- `results.csv`: `f57d94f9a06bc3a5d9081aeecf76ac25f81d6371838ab142ec6e6869eee49ae2`
- `manifest.json`: `358f162b0cce41033f5bb46786dd98b688bbeb326149f76e291c6c25f8099716`
- `validation.json`: `c6ec1948e78c43e0e147e842d894cc1fa78b14cee940015cd4e43065ddb507e8`

This run remains separate, noncanonical development evidence until it is
explicitly added to a manuscript evidence policy and frozen artifact set.
