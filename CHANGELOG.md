# Changelog

## 0.2.4

- Made the exact core-distance sweep score only complete groups of tied
  observed thresholds, so the optimized prefix and inclusive threshold
  materialization are identical.
- Made guarded Gamma confirmations use independent cold starts at every
  confirmed order; ordinary candidate searches retain split-warm continuation.
- Replaced many-to-one target scoring with a versioned one-to-one Hungarian
  evaluation contract reporting target F1, purity, coverage, discovery, and
  candidate burden separately.
- Separated known synthetic-noise evidence from biological and Gaia
  reference-background abstention diagnostics.
- Added lossless prediction/contingency artifacts, checksum-validated resume,
  and a fail-closed submission evidence freeze.

## 0.2.3

- Compiled the constrained multiscale-Gamma EM loop with Numba and removed
  repeated per-iteration matrix allocations.
- Changed the ordinary candidate search to one split-warm initialization with
  tolerance `1e-4`, while retaining guarded tighter confirmation for ambiguous
  high-order fits.
- Added duplicate-signal-profile detection and a local three-order confirmation
  that prevents deterministic over-fragmentation on imbalanced data.
- Completed 175 of 175 StrataSCAN robustness jobs. The paired median runtime
  ratio against the conservative implementation is 0.0442 (22.6x faster), with
  a median pairwise-F1 change of +0.0007.
- Promoted Numba to a required runtime dependency and removed the unreachable
  uncompiled solver and optional-Numba fallbacks.
- Added a single release-campaign launcher, exact job selection for clean
  partial evaluations, frozen worker-count provenance, and source-drift checks
  that prevent unsafe in-place benchmark resumption.

## 0.2.2

- Preserved the frozen clustering outputs and public 0.2.1 algorithm defaults
  while hardening evaluation, evidence interpretation, and label validation.
- Added post-hoc target-background-structure harmonic F1 and a non-composite
  Pareto check so target recovery cannot hide background or segmentation loss.
- Added study leave-one-out sensitivity, named Gaia extreme fields, and
  seed-resolved locked synthetic effects; no seed, prediction, or declared
  endpoint changed.
- Reworked the method diagram to distinguish fixed choices, conditional models,
  the exact within-block sweep, the heuristic scan, and the non-merging
  invariant.
- Made core metrics safe for sparse, very large integer label identifiers and
  added explicit shape/type validation with regression tests.
- Rebuilt and visually reviewed the OEDM manuscript and supplemental figure
  after a fourth adversarial review round.

## 0.2.1

- Preserved the 0.2.0 clustering decisions and defaults while correcting
  release and benchmark provenance for the frozen fixed-range manuscript
  snapshot versus the public adaptive 8--24 component search.
- Replaced ambiguous adaptive-model labels in the fixed-range benchmark
  profile with exact Gamma-basis and semantic-background identifiers.
- Added an explicit scalability evidence boundary: the submitted snapshot is
  directly monitored through 396,460 cells, while million- and five-million-
  point records belong to earlier algorithm generations.
- Consolidated three redundant secondary plots into one supplemental figure
  and retained all negative biological, Gaia, synthetic, and resource results
  in machine-readable tables.
- Revised the OEDM manuscript, supplement, artifact manifest, and final PDF
  after a fresh adversarial review; no benchmark seed or frozen prediction was
  changed.

## 0.2.0

- Promoted `OptimizationStrataSCAN` to the released algorithm and made the
  public `StrataSCAN` name an alias for it. The package and estimator provenance
  now report 0.2.0.
- Fixed the release background formulation to the original discrete Gamma
  density bases with semantic largest-rate-gap grouping. Continuous latent-rate
  backgrounds remain explicit research options and are not release defaults.
- Kept the released 0.1.2 implementation frozen at Git tag `v0.1.2`.
- Started the optimization research line with a proposed MDL-calibrated
  StrictCore objective that preserves the `k`-NN graph and Gamma shell mixture
  while jointly selecting supported strata, core radii, border assignments,
  and background points.
- Documented replacements for the fixed dense/tail quantiles, tail test,
  minimum cluster size, graph width, and border multiplier before changing the
  public estimator.
- Implemented the opt-in `OptimizationStrataSCAN` dev9 prototype: exact
  observed-radius event sweeps, Gamma shell log-odds, held-out k-NN topology
  Bayes factors, MDL component codes, and learned border/noise assignment.
- Verified sparse execution through one million points. The prototype improved
  mean synthetic metrics but regressed on Gaia border recall during development.
- Reworked the opt-in estimator as dev10: full-data ICL Gamma selection replaces
  the repeated holdouts, MAP component assignment replaces the posterior gate,
  and an adaptive coded background scan replaces the inherited 2% tail probe.
- Added density-ordered, non-merging graph growth and corrected DBSCAN-like
  connectivity to use all 32-NN edges contained by optimized core radii.
- Froze a fresh two-seed synthetic audit. Dev10 improved mean macro target F1
  by 0.0771 but retained a worst paired regression of 0.0619.
- Added the dev11 validation layer for adaptive Gamma-basis expansion,
  split-warm candidate fitting, semantic signal/background diagnostics,
  uniform-noise null checks, and an experimental continuous-rate background
  distribution. The initial alternatives expose a quality-versus-null-control
  tradeoff and are not approved for promotion.
- Added the dev12 background-only comparison with Gamma, lognormal, and
  inverse-Gamma latent-rate laws while keeping the signal Gamma mixture fixed.
  Gamma remained the strongest continuous-rate control; lognormal reduced
  component pressure but lost signal quality, and inverse-Gamma failed
  high-dimensional nulls. None of the continuous-rate alternatives is promoted.

## 0.1.2

- Promoted `PredictiveMultiscaleStrataSCAN` to the released algorithm and made
  the public `StrataSCAN` name an alias for it.
- Replaced BIC selection of the multiscale Gamma mixture with constrained
  repeated-holdout predictive likelihood and the one-standard-error rule.
- Added `PredictiveMultiscaleConfig` and the explicit
  `PredictiveMultiscaleStrataSCAN` class to the public API.
- Froze the completed evaluation over seven synthetic scenarios, three classic
  cytometry datasets, ten Samusik samples, all 359 Gaia fields, and synthetic
  scaling through five million observations.

## 0.1.1

- Promoted the four-shell multiscale Gamma stratification from the experimental
  `MultiscaleStrataSCAN` estimator to the public `StrataSCAN` default.
- The primary estimator now fits local-Poisson Gamma shells at neighbour ranks
  4, 8, 16, and 32, using `log(d4)` and anchored distance ratios to separate
  dense strata from the uniform background tail.
- Added `MultiscaleConfig` to the public API for explicit configuration of the
  primary algorithm. The experimental multiscale estimator has been removed;
  `ResidualStrataSCAN` remains experimental.

## 0.1.0

- Released the frozen Gamma-StrictCore StrataSCAN algorithm as the only public
  clustering path.
- Added Gamma/uniform-tail density stratification, q95 supported-stratum cores,
  the empirical `d4 / d32` tail component test at `p < 0.05`, q5 significant
  tail cores, and `1.25x` core-radius border assignment.
- Added exact KD-tree and block-queried FAISS HNSW neighbour backends.
- Removed the experimental GMM, merge, adaptive, knee, plateau, percolation,
  distance-gap, component-adaptive, RedCEA, structural-recovery, routing,
  fallback, and fragmentation-veto implementations from the runtime package.
- Froze the release evaluation scope to seven synthetic cases, Levine
  cytometry, and 24 Gaia fields. Mosmann and Nilsson are excluded from 0.1.0.
