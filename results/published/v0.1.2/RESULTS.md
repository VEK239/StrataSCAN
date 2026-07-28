# StrataSCAN 0.1.2 results

This document freezes the completed evidence for the 0.1.2
`PredictiveMultiscaleStrataSCAN` release. `StrataSCAN` is an alias for the same
estimator. The source tables and detailed analysis are retained in
`results/predictive_full_20260727`, `results/predictive_lower_q_20260727`, and
`results/predictive_scaling_5m_20260727`.

## Evaluation scope

- Seven synthetic scenarios at 5k, 20k, and 50k observations with five seeds
  (105 quality runs).
- The same scenarios at 100k and 200k observations (14 scalability runs).
- Levine, Mosmann, and Nilsson cytometry datasets.
- Ten independently processed Samusik mouse samples.
- All 359 available Gaia benchmark fields.
- Seven synthetic scenarios at 500k, 1M, 2M, and 5M observations (28 extended
  scalability runs).

All prespecified 0.1.2 estimator jobs in these matrices completed successfully.

## Quality summary

Across the 21 synthetic scenario-size conditions, 0.1.2 reached mean macro
target F1 0.806 and median pairwise F1 0.964. The 0.1.1 reference reached 0.790
and 0.973. The paired mean macro-F1 delta was +0.016 with a condition-bootstrap
95% interval of -0.006 to +0.041; six conditions improved and fifteen degraded.

On Levine, Mosmann, and Nilsson together, mean macro target F1 was 0.091 versus
0.032 for the 0.1.1 reference. The per-dataset values were 0.055 on Levine,
0.014 on Mosmann, and 0.203 on Nilsson.

Across ten Samusik samples, mean macro target F1 was 0.074 versus 0.063 for the
reference. Seven of ten samples improved; the paired bootstrap interval for the
mean delta was -0.015 to +0.034.

Across 359 Gaia fields, mean best-cluster F1 was 0.894, median F1 was 1.000, and
the F1 >= 0.80 success rate was 0.889. The 0.1.1 reference values were 0.897,
1.000, and 0.889. The paired bootstrap interval for the mean delta was -0.022
to +0.015.

## Runtime and scaling

The median synthetic quality-run runtime was 2.13 times the 0.1.1 reference;
the median Gaia runtime was 12.0 times the reference.

Extended-scaling median runtimes were 106.5 seconds at 500k, 189.7 seconds at
1M, 400.3 seconds at 2M, and 1013.0 seconds at 5M observations. Memory records
are end-of-task RSS measurements and must not be interpreted as continuously
sampled peak memory.

## Parameter-sensitivity guardrail

A labelled sensitivity sweep found q=0.95 to be the best single fixed dense
core quantile across the 13 cytometry datasets and q=0.25 across the 359 Gaia
fields. Dataset-specific optima for Mosmann and Nilsson were q=0.35 and q=0.30.
These values were selected on the same annotations used for scoring, so they
remain supervised sensitivity evidence rather than an unbiased automatic
parameter policy. The released default remains q=0.95.

## Interpretation

Version 0.1.2 freezes the predictive estimator as the released algorithm, but
the evidence does not establish a universal quality improvement over 0.1.1.
The main benefits are selected synthetic conditions, Nilsson, and a modest
Samusik gain. The principal limitations are slower execution, unchanged Gaia
aggregate success, and rare large failure flips. These limitations are part of
the frozen release record and should not be omitted when reporting 0.1.2.
