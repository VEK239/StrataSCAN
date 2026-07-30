# StrataSCAN 0.2.2 evidence decision

Version 0.2.2 is an evaluation-hardening and manuscript-review release. It
does not change the 0.2.1 clustering algorithm, public defaults, frozen seeds,
predictions, or declared endpoints.

## Evidence added

- A post-hoc target-background-structure harmonic F1 (TBS-HF1) requires target
  recovery, reference-background F1, and pairwise structure simultaneously.
  It is reported only as a sensitivity diagnostic.
- A non-composite Pareto check shows that MDL-StrataSCAN improves all three
  constituent axes over 0.1.2 on only Levine; the other 12 biological outputs
  are trade-offs.
- Equal-study TBS-HF1 is 0.094 for MDL-StrataSCAN, 0.051 for 0.1.2, and 0.017
  for SNN-DBSCAN. MDL remains first in each leave-one-study-out analysis, but
  its absolute mean ranges only from 0.057 to 0.125.
- The two locked synthetic effects are now shown separately. Moons reverses
  from +0.332 macro target F1 at seed 197 to -0.062 at seed 251; both
  Gaussian-overlap and rings seeds improve.
- Gaia field identities are retained. UBC1322 is the largest best-cluster-F1
  loss (1.000 to 0.000), while UBC1278 is the largest gain (0.013 to 0.900).

## Implementation hardening

Pairwise metrics no longer allocate arrays indexed by raw label magnitude.
Sparse large integer labels are handled by observed-value counts, and metric
inputs now receive explicit dimensionality, shape, and integer-type checks.
This changes failure behavior for invalid inputs, not valid benchmark scores.

## Decision

The manuscript remains a defensible weak-accept workshop submission. The new
diagnostics make the limitations sharper rather than improving the frozen
results retrospectively. Component ablations, additional independent studies,
the full Gaia matrix, and direct million-point evaluation of the submitted
snapshot remain outstanding.
