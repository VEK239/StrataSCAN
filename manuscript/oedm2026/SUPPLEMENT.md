# Supplementary figure guide

The canonical `target-discovery-v1` supplement contains:

- S3: development synthetic target F1, discovery rate, and true-noise F1;
- S4: dataset-level cytometry one-to-one target F1;
- S5: Gaia target discovery, purity, coverage, target F1, unmatched-candidate
  burden, and completion;
- S7: the 270-cell, nine-method controlled synthetic known-noise sweep.

All panels use the five v0.2.4 freezes. Failed quality remains missing. S4 does
not treat ten Samusik samples as ten independent studies. S5 calls
reference-negative agreement a background-abstention diagnostic and records
the sole mixed-provenance boundary: native 0.2.4 StrataSCAN plus exact
single-target-rescored legacy external predictions. S7 is not
generalized beyond its declared synthetic mechanism.

`tables/discovery_threshold_sensitivity.csv` and its concise Markdown rendering
recompute discovery from the frozen per-target matches at (0.90,0.05),
(0.90,0.10), (0.90,0.25), and (0.95,0.10). The locked 0.90/0.10 pair remains
primary; this analysis changes neither assignments nor fitted parameters.

Historical S1 galleries, S2 model-specification perturbations, and S6
solver-predecessor comparisons are noncanonical because they were not rescored
under this contract. They are absent from submission claims and manifests.
