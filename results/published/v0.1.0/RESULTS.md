# StrataSCAN 0.1.0 results

This document records the fixed results used for the 0.1.0 release. The method
is the exact public `StrataSCAN()` default. No dataset-specific parameters or
fallback paths are used.

## Synthetic matrix

Mean pairwise F1 over three seeds, 20,000 points per dataset:

| Dataset | kNN-DBSCAN | StrataSCAN 0.1.0 |
|---|---:|---:|
| imbalanced 16D | 0.395243 | **0.898579** |
| moons 2D | 0.003003 | **0.973595** |
| multidensity 2D | **0.981510** | 0.977350 |
| Gaussian overlap 8D | 0.055583 | **0.314307** |
| overlapping density 16D | 0.268963 | **0.282898** |
| rings 2D | 0.757940 | **0.759756** |
| ultrasparse 16D | **1.000000** | 0.997336 |

StrataSCAN wins five of seven cases. The two losses are small regressions on
near-perfect baseline cases.

## Levine cytometry

| Metric | kNN-DBSCAN | StrataSCAN 0.1.0 |
|---|---:|---:|
| Macro target F1 | 0.040722 | **0.095650** |
| Signal coverage | 0.081039 | **0.179173** |
| Background rejection | **0.942432** | 0.865488 |

The gain is recall-driven and the absolute clustering remains weak: none of
the 14 annotated populations reaches F1 0.8. Mosmann and Nilsson are not part
of the 0.1.0 evaluation scope.

## Gaia matrix

Primary metric: best-cluster F1 over 24 stable-hash fields.

| Method | Mean | Median |
|---|---:|---:|
| kNN-DBSCAN | 0.822975 | 0.889113 |
| StrataSCAN 0.1.0 | **0.939188** | **1.000000** |

Against kNN-DBSCAN, StrataSCAN records 22 wins, one tie, and one loss. The
single major failure is UBC1209 (0.024843 versus 0.839506): recall is 1.0 but
precision is 0.012578 because the reference population is absorbed into a
large component.

| Field | kNN-DBSCAN | StrataSCAN 0.1.0 |
|---|---:|---:|
| UBC1003 | 0.871795 | **0.988506** |
| UBC1023 | 0.571429 | **0.857143** |
| UBC1040 | 0.000000 | **0.730159** |
| UBC1058 | 0.875000 | **1.000000** |
| UBC1071 | 0.969697 | **1.000000** |
| UBC1081 | 0.960000 | **1.000000** |
| UBC1124 | **1.000000** | **1.000000** |
| UBC1125 | 0.838710 | **1.000000** |
| UBC1138 | 0.774194 | **1.000000** |
| UBC1159 | 0.800000 | **1.000000** |
| UBC1162 | 0.685714 | **0.977778** |
| UBC1196 | 0.800000 | **0.990654** |
| UBC1197 | 0.903226 | **1.000000** |
| UBC1204 | 0.971429 | **1.000000** |
| UBC1209 | **0.839506** | 0.024843 |
| UBC1251 | 0.909091 | **1.000000** |
| UBC1259 | 0.787879 | **1.000000** |
| UBC1260 | 0.909091 | **0.971429** |
| UBC1268 | 0.533333 | **1.000000** |
| UBC1272 | 0.960000 | **1.000000** |
| UBC1330 | 0.943396 | **1.000000** |
| UBC1332 | 0.965517 | **1.000000** |
| UBC1336 | 0.905660 | **1.000000** |
| UBC1349 | 0.976744 | **1.000000** |

All frozen runs completed successfully: 21/21 synthetic jobs, 1/1 in-scope
cytometry job, and 24/24 Gaia jobs.
