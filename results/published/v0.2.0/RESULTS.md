# StrataSCAN 0.2.0 release record

This record freezes the primary algorithm identity and default configuration
for StrataSCAN 0.2.0. The public `StrataSCAN` name is an alias for
`OptimizationStrataSCAN`. The 0.1.2 `PredictiveMultiscaleStrataSCAN` estimator
remains available explicitly for historical reproduction.

## Released formulation

Version 0.2.0 uses the original semantic Gamma-basis background formulation:

- a 32-neighbour graph and shell ranks 4, 8, 16, and 32;
- full-data Gamma candidate fitting;
- semantic ICL with an adaptive search starting at eight and bounded at 24;
- deterministic split warm starts plus three initializations;
- `background_distribution="gamma_components"`;
- dense-prefix signal roles selected by the largest adjacent fitted-rate gap;
- one aggregated semantic background state;
- exact observed-radius StrictCore event sweeps and coded topology evidence;
- a separate coded background overdensity scan;
- density-ordered non-merging connectivity and strict core-radius borders.

The exact machine-readable defaults are frozen in
`benchmarks/protocol.v0.2.0-release.json`.

## Evidence inherited by the release decision

The dev10 locked two-seed synthetic audit reported a mean macro-target-F1 gain
of 0.0771 over 0.1.2, with a worst paired regression of 0.0619. The later dev11
work introduced adaptive component expansion, split warm starts, semantic
signal/background accounting, and explicit null diagnostics.

The dev12 background-only screen compared continuous Gamma-rate, lognormal-rate,
and inverse-Gamma-rate background laws while leaving signal Gamma shells fixed.
None was suitable as the default: lognormal reduced component pressure but lost
signal quality and increased runtime, while inverse-Gamma failed
high-dimensional uniform nulls. Consequently the release retains the original
discrete Gamma-component background rather than promoting a continuous-rate
alternative.

## Known limitations

This release decision does not erase the development warnings:

- The largest-rate-gap semantic role model activated false signal and clusters
  on the small uniform-null screen used in dev11.
- Increasing the component ceiling is diagnostic rather than curative. Some
  candidate searches continue upward, and the imbalanced development case
  reached 16 states in the continuous-background cap check.
- Component count is a density-basis resolution, not a scientific cluster
  count.
- The adaptive post-dev12 default has not yet completed a new full cytometry,
  Gaia, scaling, and multi-seed release matrix. The release protocol records
  this follow-up as pending rather than presenting development results as a
  completed confirmatory benchmark.

These limitations should be reported with 0.2.0 results. The fixed release
means that subsequent validation must evaluate this exact configuration rather
than tune it on the evaluation labels.
