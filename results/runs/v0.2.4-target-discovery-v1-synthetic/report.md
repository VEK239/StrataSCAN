# StrataSCAN benchmark protocol 0.2.4-target-discovery-v1-synthetic

Suite: `synthetic`. Expected jobs: 504. Structural validation: **complete**.

Resources: 1 thread(s), 1200 s timeout, 8192 MB RSS limit.

| Suite | Method | Jobs | OK | Failures | Median primary metric | Median runtime, s |
|---|---|---:|---:|---:|---:|---:|
| synthetic | AMD-DBSCAN | 56 | 33 | 23 | 0.9251 | 4.836 |
| synthetic | DBSCAN | 56 | 56 | 0 | 0.5407 | 1.646 |
| synthetic | HDBSCAN | 56 | 56 | 0 | 0.9582 | 3.984 |
| synthetic | OPTICS | 56 | 56 | 0 | 0.3418 | 5.649 |
| synthetic | SNN-DBSCAN | 56 | 56 | 0 | 0.0684 | 0.818 |
| synthetic | StrataSCAN | 56 | 56 | 0 | 0.9338 | 2.218 |
| synthetic | VDBSCAN-2007 | 56 | 56 | 0 | 0.6705 | 2.257 |
| synthetic | kNN+Leiden | 56 | 56 | 0 | 0.5985 | 1.869 |
| synthetic | kNN-DBSCAN | 56 | 56 | 0 | 0.5372 | 0.670 |

Controlled failures remain in the denominator. This report is generated only from job records listed in the frozen manifest.
