# Recorded article evidence

These are curated existing measurements, not newly executed experiments. Numeric scores, method identities, seeds, and completion statuses were preserved. Runtime profiles, per-target JSON payloads, tracebacks, job identifiers, and references to unavailable per-job artifacts were removed to keep the tables useful and compact. [integrity.json](integrity.json) records input and curated CSV checksums and row counts.

| CSV | Rows | Interpretation |
| --- | ---: | --- |
| [synthetic_quality.csv](synthetic_quality.csv) | 504 | Seven families × eight seeds × nine methods; distinguish development and fresh seeds |
| [density_contrast.csv](density_contrast.csv) | 27 | Nine scenarios × three seeds; StrataSCAN only |
| [rare_dense_noise.csv](rare_dense_noise.csv) | 270 | Six background fractions × five seeds × nine methods; fixed rare targets |
| [scaling.csv](scaling.csv) | 28 | Seven families × four sizes; StrataSCAN, 8 GiB / 1 h budget |
| [cytometry.csv](cytometry.csv) | 117 | Thirteen datasets × nine methods; AMD-DBSCAN excluded from article comparisons |
| [gaia.csv](gaia.csv) | 3,231 | 359 fields × nine methods; external-method legacy measurements were rescored |
| [noise_sweep.csv](noise_sweep.csv) | 270 | Supplementary factorial, separate from fixed rare-target experiment |

## Start with the summaries

The [summaries/](summaries/) directory retains compact existing article tables: fresh-seed synthetic families and method summaries, density contrast, 99% background, scaling by size/method, and cytometry scores and sensitivity. `scaling_comparison.csv` contains successful runtime observations used by the expanded comparison figure; it is not a complete raw status matrix. `scaling_by_method_size.csv` includes completion information where available. Summaries can pool distinct resource budgets: use the experiment context before comparing runtime columns.

## Read the measurements correctly

- `status`: keep `ok`, `timeout`, `error`, and `memory_limit` separate. Missing quality is not a measured zero.
- `macro_target_f1`: primary target-level score, under one-to-one Hungarian matching.
- `target_discovery_rate`: fraction of targets satisfying purity >= 0.90 and coverage >= 0.10.
- `runtime_seconds` / `process_runtime_seconds`: method timing / process timing where recorded; the two are not interchangeable.
- `process_peak_rss_mb`: recorded peak process memory; values are interpreted as MiB in article summaries.
- `package_version`, `evidence_stage`, `evaluation_generation`: distinguish native 0.2.4 runs from reused and rescored evidence.
- `parameters_json` and `dataset_metadata_json`: retained method settings and dataset context where present.

AMD-DBSCAN is retained in raw evidence for transparency and excluded from comparative article summaries. The synthetic development seeds are 23, 42, 73, 101, and 151; fresh seeds are 211, 223, and 227. Report those groups separately. Cytometry and Gaia use different background semantics. See the [evaluation contract](../benchmarks/evaluation_protocol.v1.json) for the exact policy.

The source code, protocols, figures, and scores support inspection and new runs. Removed per-observation prediction archives mean these CSVs alone cannot reproduce arbitrary new target matching or discovery thresholds. Original data are also required for full re-execution.
