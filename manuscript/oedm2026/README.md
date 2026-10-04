# OEDM 2026 manuscript package

This directory contains the anonymous IEEE workshop manuscript and its
fail-closed reproducibility pipeline for StrataSCAN 0.2.4.

## Canonical evaluation contract

The submission uses `target-discovery-v1`: Hungarian one-to-one target--cluster
matching maximizes pairwise F1; target F1, purity, coverage, discovery rate
(purity >= 0.90 and coverage >= 0.10), and candidate burden remain separate.
Known-noise metrics are valid only for synthetic data. Cytometry and Gaia
reference negatives support an explicitly diagnostic background-abstention
measure, not a physical-noise claim. Failed executions retain missing quality,
and paired effects use joint successes with their declared denominators.

Exactly five same-stem CSV/provenance pairs under
`results/published/v0.2.4/target-discovery-v1/` are build inputs:

1. `synthetic_release.csv` (development and fresh rows distinguished by
   `evidence_stage`);
2. `synthetic_scaling.csv`;
3. `cytometry.csv`;
4. `gaia.csv`;
5. `noise_factorial.csv`.

Each provenance JSON records the CSV SHA-256, evaluator version, Hungarian
matching strategy and objective, and both discovery thresholds. The scripts
reject absent or mismatched metadata, unexpected identities, unrescored legacy
rows, non-synthetic `noise_f1` fields, and incomplete matrices.

Synthetic release, noise-sweep, scaling, and cytometry evidence is native
target-discovery-v1. Gaia alone combines native 0.2.4 StrataSCAN with legacy
external predictions after an exact single-target rescore; the equivalence
proof, source generations, and seed aggregation are recorded in its provenance.
Historical model-specification and solver-predecessor studies are noncanonical
and do not support submission claims.

The raw method key `AMD-DBSCAN` is displayed as **AMD-inspired**. Its frozen
controlled implementation retains the published dense distance stage but does
not reproduce the later AMD parameter-adaptation procedure; it must not be
interpreted as the authors' official implementation. Its 0/13 cytometry
completion is reported without fabricated quality. See
`docs/V0.2.4_BASELINE_INTEGRITY_AUDIT.md` for the attribution audit.

## Build

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File manuscript\oedm2026\scripts\build.ps1
```

The build runs final statistics, keyed manuscript synchronization, figures,
the predeclared discovery-threshold sensitivity table, the synchronization
check, LaTeX, and the artifact-manifest write/check. It
stops before figure or PDF generation until all five evidence blocks pass.
Generated summaries, figures, and manifests become canonical only after a
successful complete build.
