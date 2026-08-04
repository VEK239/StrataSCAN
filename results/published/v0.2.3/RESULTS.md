# StrataSCAN 0.2.3 release decision

Version 0.2.3 promotes the compiled constrained-Gamma solver and guarded local
confirmation as the supported StrataSCAN implementation. The public estimator
and clustering objective remain `OptimizationStrataSCAN`; the pre-optimization
implementation is retained only for historical compatibility and comparison.

## Decision evidence

- 175 of 175 StrataSCAN seven-family robustness jobs completed, including all
  10 cells that timed out under the conservative implementation.
- Across 165 successful paired reference jobs, the median runtime ratio is
  0.0442, corresponding to a 22.6x median speedup.
- Median pairwise F1 changes by +0.0007 and mean pairwise F1 by +0.0268.
- Failure-adjusted median pairwise F1 over the full 175-cell grid is 0.8555;
  the strongest baseline, HDBSCAN, records 0.8516 with 160 of 175 jobs complete.
- The guarded duplicate-profile confirmation restores the two severe
  imbalanced-20k regressions to 0.937 and 0.888 pairwise F1.

## Limitations

The optimized and conservative solvers can select different local optima.
Four non-imbalanced cells retain pairwise-F1 losses above 0.05. The largest-rate
gap semantic grouping and the 24-component hard bound also remain known model
limitations.

Eleven guarded reevaluations ran with four parallel workers. Their quality and
completion results are valid, but their resource readings are conservative
validation measurements. The serial four-phase v0.2.3 campaign is frozen in
`benchmarks/campaign.v0.2.3-release.json` and has not been executed in full at
release time.

## Release outcome

The speedup, recovered timeouts, deterministic guard, and preserved median
quality justify a patch release. Remaining negative cells are reported rather
than excluded. The supported package version is 0.2.3.
