# RQ4 verified claims and exclusions

## Exact supported claims

- Canonical evidence: `results/published/v0.2.4/target-discovery-v1/cytometry.csv`, SHA-256 `9dd7519d4f80490aaaa622eab12a584707b23c6928f578897a6ba39a9f58904e`.
- Design: 13 cytometry datasets, nine methods, one frozen seed, 117 attempted cells. Mosmann uses seven type plus seven state markers; the other biological datasets use their declared type-marker panels.
- Evaluation: native `target-discovery-v1`; Hungarian one-to-one matching; pairwise F1 assignment; purity threshold 0.90 and coverage threshold 0.10.
- Statuses: 97 successful, 13 errors, six timeouts, and one memory-limit termination.
- StrataSCAN: 13/13 successful; median macro target F1 0.203343 (IQR 0.190183--0.247567); median purity 0.437110; median coverage 0.244475; mean strict discovery rate 0.035256; median runtime 13.4337 s.
- StrataSCAN leads Levine (0.190183), Mosmann (0.565476), and Nilsson (0.516880).
- On Mosmann, the best successful baseline is SNN-DBSCAN at 0.429530; neither method passes the 0.90 strict-purity threshold.
- kNN+Leiden has the higher median over its 11 successful runs, 0.284008 (IQR 0.265172--0.301911), and leads all ten Samusik samples.
- Dataset winners: kNN+Leiden 10 and StrataSCAN three.
- HDBSCAN, VDBSCAN-2007, and kNN+Leiden complete 11/13; OPTICS completes 12/13; AMD-DBSCAN completes 0/13.
- StrataSCAN has nonzero strict discovery in seven Samusik samples, SNN-DBSCAN in six, and kNN+Leiden in none. Mean discovery is 0.035256 for StrataSCAN and 0.019231 for SNN-DBSCAN.

## Required exclusions and wording limits

- Exclude Gaia completely.
- Do not use the archived type-only Mosmann rows in the primary comparison.
- Do not call background abstention “noise F1” on biological data.
- Do not impute zero quality for failed cells; use NA and report completion separately.
- Do not claim that StrataSCAN is best on the Samusik series or has the highest successful-run median.
- Do not claim strict discovery of Mosmann: the 14-marker recovery is strong, but purity is below 0.90.
- Do not average AMD-DBSCAN biological quality because it has no successful cells.

## Preferred manuscript emphasis

Foreground complete 13/13 execution, the highest median target F1 among full-completion methods, and leading F1 on all three non-Samusik datasets. Present kNN+Leiden's stronger Samusik results in the same subsection.
