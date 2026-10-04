# Running the article experiments

Run commands from the repository root after installing `python -m pip install -e ".[dev,benchmark]"`. Install `[perf]` as well for FAISS HNSW experiments. The runner generates synthetic inputs, executes methods in isolated processes, and records metrics under the [evaluation contract](evaluation_protocol.v1.json).

## Inspect before running

```bash
python -m benchmarks.run_benchmark --protocol benchmarks/protocols/synthetic_quality.json --suite synthetic --list
python -m benchmarks.run_benchmark --protocol benchmarks/protocols/synthetic_quality.json --suite synthetic --list-jobs
```

`--list` prints the job count without executing the matrix. `--list-jobs` exposes exact identifiers. Use `--job-id ID` to select one existing job; repeat the flag for multiple jobs.

## Run

```bash
python -m benchmarks.run_benchmark --protocol benchmarks/protocols/synthetic_quality.json --suite synthetic --output-dir runs/synthetic_quality
```

Use a new output directory for each protocol. `--resume` continues a compatible run; `--validate-only` checks an existing manifest and results. Inputs and sources are hashed by the runner. Generated `runs/` and external `data/` are ignored by Git. Execution can be expensive: the full quality matrix has 504 jobs, not a quick example. For a quick algorithm check, use the root README example.

| Protocol in `protocols/` | Purpose | Recorded budget per job |
| --- | --- | --- |
| `release_defaults.json` | Algorithm settings reference; not an experiment matrix | 1 thread, 8 GiB, 2 h |
| `synthetic_quality.json` | 7 families, 20,000 points, 5 development + 3 fresh seeds | 1 thread, 8 GiB, 20 min |
| `density_contrast.json` | 9 density scenarios, 3 seeds; article evidence uses the StrataSCAN subset | 1 thread, 8 GiB, 20 min |
| `rare_dense_noise.json` | Three fixed 100-point targets, 6 background fractions, 5 seeds, 9 methods | 1 thread, 8 GiB, 20 min |
| `scaling.json` | StrataSCAN across 7 families and 0.5/1/2/5 million points | 1 thread, 8 GiB, 1 h |
| `scaling_comparison.json` | Expanded comparison across 7 families at 1/2/5 million points | 1 thread, 192 GiB, 15 h |
| `cytometry.json` | 13 datasets, seed 42 | 1 thread, 8 GiB, 30 min |
| `gaia.json` | StrataSCAN on 359 fields, seed 42 | 1 thread, 8 GiB, 30 min |
| `noise_sweep.json` | Supplementary noise factorial | See protocol |
| `global_contamination.json` | StrataSCAN contamination-invariance control | See protocol |

The density-contrast config contains comparator jobs as well; the recorded article table contains only its 27 StrataSCAN cells. Select those identifiers with `--job-id` when recreating that subset.

Names and layout have been cleaned up. Data-generation and method parameters retain their existing definitions. Historical comparisons were removed from configs because their original paths are not distributed here. Source edits to the runner only select the relocated default protocol and a clean output directory. Renamed protocol paths change manifest identities; start a new run instead of resuming a historical manifest.

## Comparator and evaluation boundaries

`baselines/algorithms.py` contains the existing controlled implementations. They should not be presented as official author implementations. AMD-DBSCAN was excluded from the article comparison after methodological screening; its raw rows remain identified in the CSVs. Some external-method measurements were reused and rescored with the frozen evaluation contract rather than rerun under package 0.2.4. Consult `evidence_stage`, `evaluation_generation`, and `package_version` where present.

Macro target F1 uses Hungarian one-to-one matching; unmatched targets receive zero. Target discovery requires purity >= 0.90 and coverage >= 0.10. The versioned evaluation JSON is authoritative. Keep timeout, error, and memory-limit statuses when assessing completion; successful-only runtime summaries do not establish that all methods completed.

The expanded scaling protocol specifies a complete requested matrix, whereas the stored summary contains available measurements. It is not evidence that every requested job finished. Never combine the 8 GiB and 192 GiB budgets into one performance claim.

## External data

Primary datasets are not bundled. Synthetic suites work without them. Paths and feature selection are specified in `cytometry.json` and `gaia.json`.

For cytometry, each `data/processed/DATASET/` directory needs:

- `features.csv` (or the configured `.csv.gz`): numeric marker columns and `event_id`.
- `event_metadata.csv` (or `.csv.gz`): aligned `event_id` and `population_id`.
- `feature_metadata.csv`: `feature` and `marker_class`, identifying the configured `type` / `state` markers.

The loader checks event alignment and selects markers from metadata. It applies the configured arcsinh transform and robust scaling. Mosmann uses type + state markers; the other primary datasets use type markers. Missing population labels and declared background labels become background.

Existing Samusik preparation tools are available:

```bash
python scripts/download_samusik.py
Rscript scripts/prepare_samusik_data.R
```

The R export requires `SummarizedExperiment` from Bioconductor. These tools prepare Samusik samples 01–10; automated preparation tools for Levine, Mosmann, and Nilsson are not included. Supply those exports in the schema above. This is a remaining setup step for reproducing the full biological matrix.

For Gaia, place the MCTNC data release under the `data_root` configured in `gaia.json`. Required subpaths are:

- `raw_open_cluster_fields/gaia_dr3_cone_fields/gaia_cone_FIELD.csv`: `ra`, `dec`, `parallax`, `pmra`, `pmdec`, `ruwe`, and `source_id`.
- `benchmark_reference_tables/ocfinder_table1.csv`.
- `benchmark_reference_tables/ocfinder_table2.csv`: `Cluster` and `GaiaEDR3` reference membership columns.

The article profile uses median-centred positions, robust feature scaling, and RUWE <= 1.6. Reference membership is used for evaluation. Non-members are not a verified physical-noise catalogue. Data acquisition and licensing remain separate from this MIT-licensed software.
