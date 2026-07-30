# OEDM 2026 supplemental diagnostics

The main manuscript uses three nonredundant figures: the method pipeline, the
five-seed synthetic recovery/granularity matrix, and the dataset-level
biological analysis. Three earlier standalone plots repeated values already in
Tables II–V. They are replaced by one consolidated diagnostic so that the
negative findings remain inspectable without competing with the primary
figures.

![Consolidated secondary diagnostics](figures/figS1_secondary_diagnostics.png)

**Figure S1 — Secondary evidence and sensitivity checks.** (a) Post-hoc
target–background harmonic F1 after first averaging the ten Samusik samples and
then weighting the four biological studies equally. The method is second to
SNN-DBSCAN, so the dataset-weighted first-place impression is not robust.
(b) Signed best-cluster-F1 changes on the 24 prespecified Gaia fields: six
improve, ten tie, and eight regress. (c) Mean paired changes on the two locked
synthetic seeds per family. Gains on moons, rings, and Gaussian overlap coexist
with multi-density and overlap-density regressions in at least one metric.
(d) End-to-end runtime against dataset size for the directly evaluated
fixed-range biological and Gaia pilots. The largest job is Mosmann (396,460
cells, 517 s); this is the largest scale attributable to the submitted
snapshot.

## Machine-readable evidence

- `tables/method_comparison_matrix.csv` contains both dataset- and
  study-weighted biological summaries.
- `tables/gaia24_field_results.csv` retains all 24 field-level outcomes.
- `tables/locked_synthetic_by_family.csv` retains target, pairwise, and
  injected-noise deltas.
- `tables/scalability_evidence_boundaries.csv` identifies which algorithm
  snapshot supports every reported scale observation.
- `tables/biological_segmentation_by_dataset.csv` retains fragmentation,
  merging, cluster-count, target, and reference-background diagnostics for all
  13 biological datasets.

The supplement does not introduce a new endpoint, test, seed, or prediction.
It only reorganizes diagnostics computed from the frozen artifacts.
