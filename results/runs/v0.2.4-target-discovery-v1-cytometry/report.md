# StrataSCAN benchmark protocol 0.2.4-target-discovery-v1-biological

Suite: `cytometry`. Expected jobs: 117. Structural validation: **complete**.

Resources: 1 thread(s), 1800 s timeout, 8192 MB RSS limit.

| Suite | Method | Jobs | OK | Failures | Median primary metric | Median runtime, s |
|---|---|---:|---:|---:|---:|---:|
| cytometry | AMD-DBSCAN | 13 | 0 | 13 | — | — |
| cytometry | DBSCAN | 13 | 13 | 0 | 0.0080 | 37.942 |
| cytometry | HDBSCAN | 13 | 11 | 2 | 0.0347 | 939.981 |
| cytometry | OPTICS | 13 | 13 | 0 | 0.0004 | 124.573 |
| cytometry | SNN-DBSCAN | 13 | 13 | 0 | 0.0520 | 8.119 |
| cytometry | StrataSCAN | 13 | 13 | 0 | 0.2021 | 13.434 |
| cytometry | VDBSCAN-2007 | 13 | 12 | 1 | 0.0384 | 70.647 |
| cytometry | kNN+Leiden | 13 | 11 | 2 | 0.2840 | 29.659 |
| cytometry | kNN-DBSCAN | 13 | 13 | 0 | 0.0080 | 7.403 |

Controlled failures remain in the denominator. This report is generated only from job records listed in the frozen manifest.
