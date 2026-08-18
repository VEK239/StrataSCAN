# RQ4 verified claims and exclusions

## Exact supported claims

- Canonical evidence: `results/published/v0.2.4/target-discovery-v1/cytometry.csv`, SHA-256 `bb1e7c144c1f4623f4b6d9409086608f862b2478d0d8004be369a5e7dea9cd94`.
- Design: 13 cytometry datasets, nine methods, one frozen seed, 117 attempted cells.
- Evaluation: native `target-discovery-v1`; Hungarian one-to-one matching; pairwise F1 assignment objective; purity threshold 0.90 and coverage threshold 0.10.
- Statuses: 99 successful, 13 errors, four timeouts, and one memory-limit termination.
- StrataSCAN: 13/13 successful; median macro target F1 0.202083 (IQR 0.008318--0.243984); median purity 0.437110; median coverage 0.244475; mean strict discovery rate 0.035256; median runtime 13.4337 s.
- StrataSCAN has the highest median target F1 among methods completing all 13 datasets.
- kNN+Leiden has the higher median over its 11 successful runs: 0.284008 (IQR 0.265172--0.301911), but timed out on Levine and Mosmann.
- Overall dataset winners: kNN+Leiden 10, StrataSCAN two (Levine and Nilsson), SNN-DBSCAN one (Mosmann).
- Levine: StrataSCAN 0.190183 versus best successful baseline 0.035748 (DBSCAN).
- Nilsson: StrataSCAN 0.516880 versus best baseline 0.132747 (kNN+Leiden).
- Across ten Samusik samples, kNN+Leiden has higher macro target F1 than StrataSCAN in all ten.
- StrataSCAN has a nonzero strict discovery rate in seven of ten Samusik samples; kNN+Leiden has zero in all ten completed Samusik cells; SNN-DBSCAN has nonzero discovery in six.
- SNN-DBSCAN has the highest mean strict discovery rate (0.096154), driven principally by complete discovery of the single Mosmann target; StrataSCAN's mean is 0.035256.
- AMD-DBSCAN completed 0/13 and therefore has no biological quality aggregate.

## Required exclusions and wording limits

- Exclude Gaia completely.
- Do not use old manuscript-derived rows that encode failed executions as zeros.
- Do not call background abstention “noise F1” on biological data. It is a diagnostic only and is absent from the primary figure/table.
- Do not impute a zero quality score for timeout, memory-limit, or error cells; use NA and report completion separately.
- Do not claim that StrataSCAN is best on the Samusik series, best on every dataset, or has the highest overall successful-run median; those claims are false under the canonical evidence.
- Do not claim broadly high target discovery: the strict rates are low. The supported statement is nonzero discovery on seven Samusik samples and a purity-oriented trade-off relative to kNN+Leiden.
- Do not average AMD-DBSCAN biological quality because it has no successful cells.
- Do not treat unlabeled cytometry events as verified background/noise.

## Preferred manuscript emphasis

Foreground three verified advantages: complete 13/13 execution, highest median target F1 among full-completion methods, and leading target F1 on Levine and Nilsson. Present kNN+Leiden's stronger Samusik F1 and SNN-DBSCAN's stronger Mosmann/discovery result in the same subsection so that the scope of the advantage is explicit.

