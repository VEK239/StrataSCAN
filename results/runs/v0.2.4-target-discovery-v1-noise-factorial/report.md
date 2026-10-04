# StrataSCAN benchmark protocol 0.2.4-target-discovery-v1-noise-sweep

Suite: `synthetic`. Expected jobs: 270. Structural validation: **complete**.

Resources: 1 thread(s), 1200 s timeout, 8192 MB RSS limit.

| Suite | Method | Jobs | OK | Failures | Median primary metric | Median runtime, s |
|---|---|---:|---:|---:|---:|---:|
| synthetic | AMD-DBSCAN | 30 | 30 | 0 | 0.0360 | 5.246 |
| synthetic | DBSCAN | 30 | 30 | 0 | 0.2978 | 0.183 |
| synthetic | HDBSCAN | 30 | 30 | 0 | 0.6665 | 2.911 |
| synthetic | OPTICS | 30 | 30 | 0 | 0.0130 | 15.815 |
| synthetic | SNN-DBSCAN | 30 | 30 | 0 | 0.1745 | 0.225 |
| synthetic | StrataSCAN | 30 | 30 | 0 | 0.7629 | 0.776 |
| synthetic | VDBSCAN-2007 | 30 | 30 | 0 | 0.0558 | 0.285 |
| synthetic | kNN+Leiden | 30 | 30 | 0 | 0.3390 | 1.044 |
| synthetic | kNN-DBSCAN | 30 | 30 | 0 | 0.2927 | 0.070 |

Controlled failures remain in the denominator. This report is generated only from job records listed in the frozen manifest.
