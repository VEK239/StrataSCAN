# RQ3 — Million-scale performance

## Research question

**Does StrataSCAN remain computationally practical on datasets containing millions of observations?**

## Manuscript-ready Results subsection

### Million-scale performance

We evaluated the released StrataSCAN configuration on seven heterogeneous synthetic families at 0.5, 1, 2, and 5 million observations (one thread, 3,600-s timeout, 8,192-MiB process-memory limit, seed 42). The frozen 28-cell campaign was structurally complete. StrataSCAN completed all seven families at both one and two million observations and five of seven families at five million observations. Median runtime increased from 4.00 min at one million to 8.17 min at two million and 23.07 min among the five successful five-million-point runs (range 4.91–32.68 min). At five million, median process peak RSS was 6.29 GiB (maximum 7.66 GiB among successes); the rings and 8-D Gaussian-overlap cases reached the 8-GiB limit.

The completed five-million-point runs retained strong target recovery on the 16-D imbalanced, 16-D ultrasparse, and two-moons families, with macro target F1 values of 0.949, 0.995, and 0.811, respectively, and target-discovery rate 1.0 in all three. The multidensity and overlapping-density endpoints were weaker (F1 0.413 and 0.457), so they are retained in the audit table but are not used to support a universal quality-preservation claim.

The baseline scaling campaign used the same machine, one-thread limit, 3,600-s timeout, and 8-GiB memory cap, but its five-million-point ladder is structurally incomplete (177 of 224 expected records) and its scaling records do not contain clustering-quality metrics. At one and two million observations, StrataSCAN and kNN-DBSCAN were the only methods that completed all seven recorded families at both sizes; SNN-DBSCAN completed 7/7 and 6/7, respectively, whereas HDBSCAN and OPTICS completed 0/7 at either size after timeouts and gated skips. In the one directly matched five-million-point imbalanced case, StrataSCAN completed in 30.64 min, compared with 34.39 min for kNN-DBSCAN and 36.20 min for SNN-DBSCAN; kNN+Leiden reached the memory limit. Thus, the evidence supports million-scale suitability and a favorable execution envelope, but not universal speed superiority.

**One-sentence summary:** StrataSCAN completed every tested family at one and two million observations and reached five million on five of seven families, demonstrating a broad million-scale execution envelope while retaining high target recovery on the strongest completed large-scale cases.

## Figure caption

**Fig. X. Million-scale StrataSCAN execution envelope.** Runtime across seven synthetic families under one thread, a 3,600-s timeout, and an 8,192-MiB process-memory limit. The line is the median over successful jobs, the band spans the successful-family range, and labels report successful/attempted families. At five million observations, five families completed; the rings and 8-D Gaussian-overlap jobs reached the memory limit. Failed jobs are not assigned artificial runtimes.

## Compact comparison-table caption and note

**TABLE X. Completion and successful-run runtime at million scale.** Entries at 1M and 2M are successful/recorded families. The five-million baseline ladder is structurally incomplete; starred entries therefore describe only recorded endpoints and must not be interpreted as full-suite completion. The 2M runtime column is the median over successes and must be read jointly with completion. M denotes memory limit and P denotes a gated skip following a prior failure.

## Claim boundary and exclusions

- Use the canonical v0.2.4 StrataSCAN campaign for StrataSCAN runtime, memory, completion, and target-discovery quality.
- Do not state that StrataSCAN completed all seven families at five million; the verified result is 5/7 under the 8-GiB cap.
- Do not state that StrataSCAN was universally fastest. kNN-DBSCAN and SNN-DBSCAN were faster in their fully completed 1M and 2M summaries.
- A narrow matched-case claim is supported: on the five-million-point imbalanced family, StrataSCAN was 10.9% faster than kNN-DBSCAN and 15.3% faster than SNN-DBSCAN.
- Do not aggregate baseline results at five million: the baseline campaign contains 177/224 expected records and usually only one recorded five-million family per method.
- Do not compare baseline clustering quality in the scaling section: the baseline scaling job records contain no quality metrics.
- Exclude the v0.2.3 two-worker StrataSCAN runtime campaign from direct serial runtime comparisons.
- Exclude the older prototype/reference-configuration endpoints because they correspond to different algorithm snapshots.
- Exclude Gaia and the abandoned optimization study.
- Retain low-quality and failed StrataSCAN endpoints in the source table; exclude them only from the narrow high-quality example claim, not from the completion denominator.

## Exact five-million StrataSCAN endpoints

| Family | Status | Runtime (min) | Peak RSS (GiB) | Macro target F1 | Discovery |
|---|---:|---:|---:|---:|---:|
| Imbalanced, 16-D | OK | 30.64 | 7.62 | 0.949 | 1.000 |
| Moons, 2-D | OK | 4.91 | 6.29 | 0.811 | 1.000 |
| Multidensity, 2-D | OK | 12.48 | 5.83 | 0.413 | 0.667 |
| Gaussian overlap, 8-D | Memory limit | — | 8.00 | — | — |
| Overlapping density, 16-D | OK | 32.68 | 7.66 | 0.457 | 0.000 |
| Rings, 2-D | Memory limit | — | 8.08 | — | — |
| Ultrasparse, 16-D | OK | 23.07 | 5.52 | 0.995 | 1.000 |

