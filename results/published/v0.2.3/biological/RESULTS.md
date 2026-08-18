# StrataSCAN 0.2.3 biological revalidation — cytometry

This evidence records the completed cytometry portion of the 0.2.3 biological
revalidation.  It evaluates the released estimator with fixed defaults on
Levine, Mosmann, Nilsson, all ten Samusik samples, and the full Gaia DR3
open-cluster field matrix.

## Protocol and comparator

- Protocol: `benchmarks/protocol.v0.2.3-biological.json`
- Run date: 2026-08-04
- Resource contract: one thread per isolated job, 1,800 s timeout, 8,192 MB
  RSS limit; two isolated jobs were run concurrently.
- Comparator: frozen baseline result records from the 0.1.0 biological
  campaign and the Samusik follow-up runs.
- Baseline source audit: baseline implementations have no source changes after
  commit `5071f3237a76cfd5c7c6c8179df446c0335127b7` (the 0.1.0 release).

## Completion

All 13 of 13 cytometry jobs completed successfully.

| Dataset | Macro target F1 | Runtime, s |
|---|---:|---:|
| Levine | 0.1902 | 78.713 |
| Mosmann | 0.0018 | 205.389 |
| Nilsson | 0.5169 | 23.710 |
| Samusik 01 | 0.2334 | 23.057 |
| Samusik 02 | 0.0125 | 21.851 |
| Samusik 03 | 0.0162 | 24.887 |
| Samusik 04 | 0.2399 | 26.633 |
| Samusik 05 | 0.2902 | 26.310 |
| Samusik 06 | 0.2769 | 24.249 |
| Samusik 07 | 0.2595 | 35.333 |
| Samusik 08 | 0.2855 | 30.573 |
| Samusik 09 | 0.2311 | 25.862 |
| Samusik 10 | 0.0220 | 20.321 |

The median macro target F1 is **0.2334**.

## Frozen-baseline context

The historical StrataSCAN row, evaluated on the identical 13-dataset coverage,
has median macro target F1 0.0402.  The unchanged baseline implementations
remain the comparison set: kNN+Leiden has the highest frozen median (0.3357,
11/13 successful jobs), followed by HDBSCAN (0.1036, but only 1/4 successful
jobs) and VDBSCAN-2007 (0.0738, 12/13).  StrataSCAN 0.2.3 completes all 13
jobs and improves materially on its historical estimator result; this report
does not claim that it exceeds kNN+Leiden on the cytometry endpoint.

## Gaia DR3

All **359 of 359** Gaia `unsupervised_field` jobs completed successfully.  The
released estimator has median best-cluster F1 **0.9836** with a 2.282 s median
runtime.  The frozen historical StrataSCAN row has median best-cluster F1
1.0000 and a 2.726 s median runtime on the same 359 fields.  The comparison
therefore records a small quality decrease while reducing median runtime by
about 16%; it does not claim superiority over the unchanged baselines, whose
best-cluster-F1 medians range from 0.5818 (OPTICS) to 1.0000 (HDBSCAN,
SNN-DBSCAN, VDBSCAN-2007, and kNN+Leiden).

The reproducible raw job records, run manifest, validation record, and CSV are
available locally under `results/runs/v0.2.3-biological-20260804/cytometry/`.
They are excluded from the repository by policy because per-job benchmark
outputs are large and reproducible.
