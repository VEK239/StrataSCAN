# Recorded article evidence

These are existing measurements collected from the experiment branches and local outputs. No new scientific experiments were substituted for the article evidence. Tables retain method identities, seeds, scores and completion statuses; verbose profiles, unavailable prediction references, absolute paths and tracebacks were removed.

| CSV | Rows | Current article panel |
| --- | ---: | --- |
| [synthetic_quality.csv](synthetic_quality.csv) | 210 | 7 families × 3 fresh seeds × 10 methods, including DPC-kNN |
| [stratification_ablation.csv](stratification_ablation.csv) | 42 | Full StrataSCAN and the single published ablation on 21 identical inputs |
| [density_contrast.csv](density_contrast.csv) | 36 | 12 scenarios × 3 seeds; StrataSCAN |
| [rare_dense_noise.csv](rare_dense_noise.csv) | 300 | 6 background fractions × 5 seeds × 10 methods |
| [scaling.csv](scaling.csv) | 28 | StrataSCAN, 7 families × 4 sizes; all complete |
| [scaling_comparison.csv](scaling_comparison.csv) | 24 | 16-D ultra-sparse family, 8 methods × 3 sizes; includes timeouts |
| [cytometry.csv](cytometry.csv) | 27 | Levine, Mosmann, Nilsson × 9 methods; Mosmann uses type + state markers |

[summaries/](summaries/) contains compact synthetic method/family scores, the published ablation, density contrast, and rare-target trajectories. Means use successful cells; `completed` and `total` make that denominator explicit. Failed-run quality is never imputed. AMD-DBSCAN completes 12/21 fresh synthetic cells; its conditional mean 0.843 cannot be ranked directly against the full-panel StrataSCAN mean 0.828.

The latest density-contrast figure contains the full 12-scenario factorial. Nine later completion cells supplement the earlier 27-cell run, giving 36 cells. Earlier manuscript prose still counted 27; this release describes the complete figure and its stored measurements.

## Columns and timing

`macro_target_f1` and `target_discovery_rate` follow the [evaluation contract](../benchmarks/evaluation_protocol.v1.json). `noise_evidence_f1` assesses the known synthetic background; reference-negative biological events have a different interpretation. Keep `ok`, `timeout`, `error` and `memory_limit` distinct. Empty fields denote unavailable measurements.

`runtime_seconds` is method timing; `process_runtime_seconds` includes process-level work. Peak memory fields are interpreted as MiB. The family-wide scaling table is a runtime source table, not a quality or memory measurement table. Its per-row budget fields distinguish the 0.5-million controls from the later large runs. The focused comparison records its final 54,000-second censoring limit: earlier successful cells retain their measured runtimes, and slow cells come from later retries. Do not treat these as synchronized runs under identical hardware conditions.

`parameters_json` and `dataset_metadata_json`, where recorded, preserve input and method context. Historical `package_version`, `evidence_stage` and `evaluation_generation` distinguish native runs from reused and rescored evidence. Missing per-observation prediction archives mean these compact tables cannot support arbitrary new matching thresholds; use the executable protocols and original inputs for re-execution.
