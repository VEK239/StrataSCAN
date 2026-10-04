# Final experiment and evidence policy

The claim-bearing method is StrataSCAN 0.2.4. Every canonical row uses
`target-discovery-v1`, Hungarian one-to-one assignment with pairwise-F1
objective, macro target aggregation, and discovery thresholds purity >= 0.90
and coverage >= 0.10.

Synthetic rows use `known_synthetic_noise` and may report
`noise_evidence_f1` plus partition pairwise F1. Cytometry uses
`heterogeneous_biological_background`; Gaia uses
`reference_negative_background`. Their abstention fields are diagnostic and
must never be described as physical-noise accuracy.

## Required freezes

All build inputs live at `results/published/v0.2.4/target-discovery-v1/`:

| Freeze | Declared matrix | Run lineage |
|---|---|---|
| `synthetic_release.csv` | 7 cases x (5 development + 3 fresh seeds) x 9 methods at 20k = 504 cells | `results/runs/v0.2.4-target-discovery-v1-synthetic` |
| `synthetic_scaling.csv` | 7 cases x 4 sizes x StrataSCAN x seed 42 | `results/runs/v0.2.4-target-discovery-v1-scaling` |
| `cytometry.csv` | 13 datasets x 9 methods | `results/runs/v0.2.4-target-discovery-v1-cytometry` |
| `gaia.csv` | 359 fields x 9 methods | `results/runs/v0.2.4-target-discovery-v1-gaia` |
| `noise_factorial.csv` | 6 known-noise fractions x 5 seeds x 9 methods at 20k = 270 cells | `results/runs/v0.2.4-target-discovery-v1-noise-factorial` |

Every CSV has a same-stem `-provenance.json` whose digest and semantic fields
are checked before aggregation.

Development and fresh seeds remain separate. Failed quality is missing;
effects pair joint successes only. There is no canonical five-size all-method
release grid. The 0.5--5M panel is a separate 28-cell StrataSCAN-only envelope.
Biological equal-dataset and equal-study
summaries remain separate. Gaia is descriptive.

Synthetic release, noise-sweep, and cytometry are native 0.2.4 all-method
runs. Scaling is native 0.2.4 StrataSCAN. Gaia alone merges native 0.2.4
StrataSCAN with exact single-target-rescored legacy external predictions; its
provenance must retain the equivalence proof and seed aggregation. Historical model-specification and solver-predecessor
comparisons used legacy semantics and are excluded from claim-bearing outputs.
