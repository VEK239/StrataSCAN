# Final target-discovery-v1 statistical summary

This generated report is accepted only after exact identity and evaluator-semantic checks.

## Locked evaluation contract

- Hungarian one-to-one maximum pairwise F1; unmatched targets score zero.
- Primary target score: equal-target macro F1.
- Discovery: purity >= 0.90 and coverage >= 0.10.
- Synthetic true-noise evidence is separate from biological/Gaia abstention diagnostics.

## Biological validation

| Method | Equal-study target F1 | Purity | Coverage | Discovery rate | Abstention F1 diagnostic | Successful datasets |
|---|---:|---:|---:|---:|---:|---:|
| DBSCAN | 0.011 | 0.091 | 0.008 | 0.000 | 0.789 | 13/13 |
| HDBSCAN | 0.065 | 0.043 | 0.370 | 0.000 | 0.243 | 11/13 |
| OPTICS | 0.003 | 0.136 | 0.002 | 0.000 | 0.766 | 12/13 |
| SNN-DBSCAN | 0.136 | 0.441 | 0.093 | 0.006 | 0.822 | 13/13 |
| VDBSCAN-2007 | 0.025 | 0.135 | 0.353 | 0.000 | 0.155 | 11/13 |
| AMD-inspired | NA | NA | NA | NA | NA | 0/13 |
| kNN-DBSCAN | 0.010 | 0.092 | 0.008 | 0.000 | 0.789 | 13/13 |
| kNN+Leiden | 0.210 | 0.150 | 0.699 | 0.000 | 0.000 | 11/13 |
| StrataSCAN | 0.358 | 0.392 | 0.454 | 0.011 | 0.390 | 13/13 |

## Gaia validation

- DBSCAN: 349/359 fields discovered; median target F1/purity/coverage 0.894/1.000/0.810; median unmatched candidates 51.000; successful fields 359/359.
- HDBSCAN: 312/359 fields discovered; median target F1/purity/coverage 1.000/1.000/1.000; median unmatched candidates 1.000; successful fields 359/359.
- OPTICS: 336/359 fields discovered; median target F1/purity/coverage 0.582/1.000/0.410; median unmatched candidates 26.000; successful fields 359/359.
- SNN-DBSCAN: 340/359 fields discovered; median target F1/purity/coverage 1.000/1.000/1.000; median unmatched candidates 44.000; successful fields 359/359.
- VDBSCAN-2007: 280/359 fields discovered; median target F1/purity/coverage 1.000/1.000/1.000; median unmatched candidates 4.000; successful fields 359/359.
- AMD-inspired: 330/359 fields discovered; median target F1/purity/coverage 0.997/1.000/1.000; median unmatched candidates 34.000; successful fields 359/359.
- kNN-DBSCAN: 349/359 fields discovered; median target F1/purity/coverage 0.889/1.000/0.800; median unmatched candidates 54.000; successful fields 359/359.
- kNN+Leiden: 310/359 fields discovered; median target F1/purity/coverage 1.000/1.000/1.000; median unmatched candidates 7.000; successful fields 359/359.
- StrataSCAN: 330/359 fields discovered; median target F1/purity/coverage 0.983/1.000/0.991; median unmatched candidates 7.000; successful fields 359/359.

All numerical prose and figure data must be generated from the companion JSON.
