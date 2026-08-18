# Analysis-ready performance tables

Snapshot: 2026-08-06T12:45:41.617Z

This folder contains tidy CSV tables plus one formatted Excel workbook. Blank metric cells are unavailable measurements, not zeros.

## Requested result blocks

1. Standard synthetic benchmark: nine methods, seven generator families, three fresh seeds, n=20,000. F1 summaries follow the frozen failure policy (non-ok quality cells contribute 0); resource and granularity summaries use successful jobs only.
2. Noise-aware benchmark: six distinct 99% diffuse-background architectures from the separate **implement-stratascan-pilot-benchmark** worktree: Gaussian 4x, 16x, 64x density ranges, moon arcs, rings, and anisotropic ellipses. Four seeds. Timeouts and AMD resource skips remain missing rather than being converted to zero.
3. Varying-noise matrix: the replicated compact-Gaussian 4x design at 50%, 75%, 90%, 95%, 97.5%, and 99% diffuse background.
4. Scalability through 5,000,000 observations: the accepted StrataSCAN envelope plus a timestamped snapshot of the in-progress failure-gated baseline grid. Pending and skipped cells remain explicit. Per-process RSS is not aggregate host memory.
5. Real data: all nine methods on 13 cytometry datasets and 359 Gaia fields. Gaia's primary endpoint is best-cluster F1; target/noise/pairwise values are retained as requested diagnostics.

## Granularity

`granularity_ratio = predicted_clusters / truth_clusters` (1 is exact). `granularity_log2_ratio` is signed (0 exact, positive over-segmentation, negative under-segmentation). `granularity_abs_log2_error` ignores direction and is 0 at exact granularity.

## Aggregation

F1 columns ending in `_mean` are arithmetic means. SD is sample SD. Runtime and RSS medians are conditional on successful completion. Coverage columns must be shown beside performance whenever failures, timeouts, skips, or pending cells occur. See `source_provenance.csv` for exact sources and commits.
