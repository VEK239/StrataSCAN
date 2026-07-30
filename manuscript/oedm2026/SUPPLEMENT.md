# OEDM 2026 supplemental diagnostics

The main manuscript uses three nonredundant figures: the method pipeline, the
five-seed synthetic recovery/granularity matrix, and the dataset-level
biological analysis. Three earlier standalone plots repeated values already in
Tables II–V. They are replaced by one consolidated diagnostic so that the
negative findings remain inspectable without competing with the primary
figures.

![Consolidated secondary diagnostics](figures/figS1_secondary_diagnostics.png)

**Figure S1 — Secondary evidence and sensitivity checks.** (a) Study-weighted
biological TB-HF1 and the stricter TBS-HF1, which also requires pairwise
structure. SNN-DBSCAN leads under TB-HF1, whereas MDL-StrataSCAN leads under
TBS-HF1; the reversal shows that no post-hoc composite gives a metric-invariant
winner. (b) Signed best-cluster-F1 changes on the 24 prespecified Gaia fields:
six improve, ten tie, and eight regress, with UBC1322 and UBC1278 naming the
largest loss and gain. (c) Both locked seed effects for target and pairwise F1,
instead of family means alone. Moons reverses sign; Gaussian overlap and rings
improve on both seeds. (d) End-to-end runtime against dataset size for the directly evaluated
fixed-range biological and Gaia pilots. The largest job is Mosmann (396,460
cells, 517 s); this is the largest scale attributable to the submitted
snapshot.

## Machine-readable evidence

- `tables/method_comparison_matrix.csv` contains both dataset- and
  study-weighted biological summaries for TB-HF1 and TBS-HF1.
- `tables/biological_pareto_by_dataset.csv` shows that only Levine improves
  target F1, background F1, and pairwise F1 simultaneously; the other twelve
  outputs are trade-offs.
- `tables/biological_study_leave_one_out.csv` reports the TBS-HF1 method rank
  after omitting each biological study in turn.
- `tables/gaia24_field_results.csv` retains all 24 field-level outcomes.
- `tables/gaia_paired_field_effects.csv` retains field identities and signed
  predecessor-to-MDL changes.
- `tables/locked_synthetic_by_family.csv` retains target, pairwise, and
  injected-noise deltas.
- `tables/locked_synthetic_seed_effects.csv` retains the two paired effects
  separately rather than only their family means.
- `tables/scalability_evidence_boundaries.csv` identifies which algorithm
  snapshot supports every reported scale observation.
- `tables/biological_segmentation_by_dataset.csv` retains fragmentation,
  merging, cluster-count, target, and reference-background diagnostics for all
  13 biological datasets.

The supplement does not introduce a new endpoint, seed, or prediction. TBS-HF1,
Pareto counts, and leave-one-study-out ranks are explicitly post-hoc sensitivity
analyses computed from the frozen artifacts; they do not alter the declared
target-only endpoint.
