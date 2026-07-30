# StrataSCAN 0.2.1 release record

Version 0.2.1 is a provenance and evidence-reporting patch over 0.2.0. It does
not change clustering decisions or the public estimator defaults. The public
`StrataSCAN` alias still selects `OptimizationStrataSCAN` with semantic ICL,
adaptive component expansion from 8 to at most 24 bases, Gamma-component
backgrounds, and density-ordered non-merging extraction.

## Corrections

- The OEDM experiments retain their historical `StrataSCAN-Optimization`
  result identifier and fixed 1–8 component search. They are now dispatched
  and described explicitly as `0.2.0-dev10-oedm2026-evaluated`, rather than
  through the mutable public alias.
- Fixed-range benchmark profiles no longer call their Gamma-basis or
  background models “adaptive.” The public adaptive search and frozen paper
  search are distinguishable from machine-readable provenance alone.
- Macro target F1 remains the declared cytometry endpoint. The
  target–background harmonic F1 is emitted only as a clearly identified
  post-hoc diagnostic.

## Evidence boundaries

No seed, prediction, or benchmark outcome changed in this patch. The revised
paper distinguishes the following scale records:

- the evaluated fixed-range snapshot: 14 paired synthetic jobs at 5,000
  observations, seven one-seed engineering jobs at 20,000, and direct
  biological/Gaia pilots through the 396,460-cell Mosmann dataset;
- an earlier dev9 prototype: a one-million-point moons smoke test;
- the 0.1.2 predecessor: seven-family scaling through five million points.

The largest directly monitored job for the submitted snapshot completed in
517.0 seconds with 646.7 MiB peak process RSS. The one-million- and
five-million-point records support the sparse algorithm lineage but are not
attributed to the submitted fixed-range solver.

## Manuscript revision

The OEDM manuscript now emphasizes biological segmentation and
reference-background agreement separately, preserves the study-weighted rank
reversal, reports the Gaia non-improvement, and adds a scalability evidence
table. Three redundant uncited figures were consolidated into one supplemental
diagnostic while their underlying tables and negative cases were retained.

The exact public defaults are frozen in
`benchmarks/protocol.v0.2.1-release.json`. The broader modeling limitations in
the 0.2.0 release record remain in force.
