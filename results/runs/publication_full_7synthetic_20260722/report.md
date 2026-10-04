# StrataSCAN benchmark protocol 0.1.0-full-legacy-grid-7synthetic

Suite: `all`. Expected jobs: 5777. Structural validation: **complete**.

Resources: 1 thread(s), 600 s timeout, 8192 MB RSS limit.

| Suite | Method | Jobs | OK | Failures | Median primary metric | Median runtime, s |
|---|---|---:|---:|---:|---:|---:|
| cytometry | AMD-DBSCAN | 3 | 0 | 3 | — | — |
| cytometry | DBSCAN | 3 | 3 | 0 | 0.0000 | 60.609 |
| cytometry | HDBSCAN | 3 | 1 | 2 | 0.1036 | 21.570 |
| cytometry | OPTICS | 3 | 1 | 2 | 0.0000 | 15.834 |
| cytometry | SNN-DBSCAN | 3 | 3 | 0 | 0.0591 | 23.446 |
| cytometry | StrataSCAN | 9 | 9 | 0 | 0.0000 | 52.071 |
| cytometry | VDBSCAN-2007 | 3 | 2 | 1 | 0.0096 | 120.011 |
| cytometry | kNN+Leiden | 9 | 3 | 6 | 0.1385 | 9.241 |
| cytometry | kNN-DBSCAN | 3 | 3 | 0 | 0.0000 | 20.241 |
| gaia | AMD-DBSCAN | 359 | 359 | 0 | 0.9966 | 0.767 |
| gaia | DBSCAN | 359 | 359 | 0 | 0.8936 | 0.280 |
| gaia | HDBSCAN | 359 | 359 | 0 | 1.0000 | 0.322 |
| gaia | OPTICS | 359 | 359 | 0 | 0.5818 | 0.904 |
| gaia | SNN-DBSCAN | 359 | 359 | 0 | 1.0000 | 0.235 |
| gaia | StrataSCAN | 1077 | 1077 | 0 | 1.0000 | 2.690 |
| gaia | VDBSCAN-2007 | 359 | 359 | 0 | 1.0000 | 0.602 |
| gaia | kNN+Leiden | 1077 | 1077 | 0 | 1.0000 | 0.657 |
| gaia | kNN-DBSCAN | 359 | 359 | 0 | 0.8889 | 0.194 |
| synthetic | AMD-DBSCAN | 119 | 57 | 62 | 0.1521 | 1.285 |
| synthetic | DBSCAN | 119 | 119 | 0 | 0.4053 | 0.787 |
| synthetic | HDBSCAN | 119 | 116 | 3 | 0.8504 | 6.082 |
| synthetic | OPTICS | 119 | 119 | 0 | 0.1823 | 14.424 |
| synthetic | SNN-DBSCAN | 119 | 119 | 0 | 0.0060 | 0.605 |
| synthetic | StrataSCAN | 119 | 119 | 0 | 0.9699 | 13.628 |
| synthetic | VDBSCAN-2007 | 119 | 118 | 1 | 0.0433 | 0.824 |
| synthetic | kNN+Leiden | 119 | 119 | 0 | 0.1700 | 2.310 |
| synthetic | kNN-DBSCAN | 119 | 119 | 0 | 0.4072 | 0.193 |

Controlled failures remain in the denominator. This report is generated only from job records listed in the frozen manifest.
