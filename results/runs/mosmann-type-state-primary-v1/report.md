# StrataSCAN benchmark protocol v0.2.4-target-discovery-v1-mosmann-type-state-primary

Suite: `cytometry`. Expected jobs: 8. Structural validation: **complete**.

Resources: 1 thread(s), 1800 s timeout, 8192 MB RSS limit.

| Suite | Method | Jobs | OK | Failures | Median primary metric | Median runtime, s |
|---|---|---:|---:|---:|---:|---:|
| cytometry | DBSCAN | 1 | 1 | 0 | 0.0000 | 671.315 |
| cytometry | HDBSCAN | 1 | 0 | 1 | — | — |
| cytometry | OPTICS | 1 | 0 | 1 | — | — |
| cytometry | SNN-DBSCAN | 1 | 1 | 0 | 0.4295 | 76.079 |
| cytometry | StrataSCAN | 1 | 1 | 0 | 0.5655 | 297.789 |
| cytometry | VDBSCAN-2007 | 1 | 0 | 1 | — | — |
| cytometry | kNN+Leiden | 1 | 0 | 1 | — | — |
| cytometry | kNN-DBSCAN | 1 | 1 | 0 | 0.0000 | 61.544 |

Controlled failures remain in the denominator. This report is generated only from job records listed in the frozen manifest.
