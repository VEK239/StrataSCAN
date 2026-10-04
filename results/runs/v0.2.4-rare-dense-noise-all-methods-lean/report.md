# StrataSCAN benchmark protocol 0.2.4-target-discovery-v1-rare-dense-noise-all-methods

Suite: `synthetic`. Expected jobs: 270. Structural validation: **complete**.

Resources: 1 thread(s), 1200 s timeout, 8192 MB RSS limit.

| Suite | Method | Jobs | OK | Failures | Median primary metric | Median runtime, s |
|---|---|---:|---:|---:|---:|---:|
| synthetic | AMD-DBSCAN | 30 | 28 | 2 | 0.8975 | 0.083 |
| synthetic | DBSCAN | 30 | 30 | 0 | 0.9950 | 0.028 |
| synthetic | HDBSCAN | 30 | 30 | 0 | 1.0000 | 0.063 |
| synthetic | OPTICS | 30 | 30 | 0 | 0.9446 | 2.770 |
| synthetic | SNN-DBSCAN | 30 | 30 | 0 | 0.9916 | 0.035 |
| synthetic | StrataSCAN | 30 | 30 | 0 | 0.9300 | 0.604 |
| synthetic | VDBSCAN-2007 | 30 | 30 | 0 | 0.9300 | 0.044 |
| synthetic | kNN+Leiden | 30 | 30 | 0 | 1.0000 | 0.095 |
| synthetic | kNN-DBSCAN | 30 | 30 | 0 | 0.9579 | 0.016 |

Controlled failures remain in the denominator. This report is generated only from job records listed in the frozen manifest.
