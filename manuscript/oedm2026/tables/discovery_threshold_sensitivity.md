# Discovery-threshold sensitivity

Assignments and target matches are frozen. The locked primary remains purity 0.90 and coverage 0.10; the other three pairs are descriptive sensitivity checks.

| Evidence block | Method | Locked rate | Sensitivity range | Maximum absolute rank shift | Successful/declared executions |
|---|---|---:|---:|---:|---:|
| cytometry | AMD-inspired | NA | NA | NA | 0/13 |
| cytometry | DBSCAN | 0.000 | 0.000--0.000 | 1 | 13/13 |
| cytometry | HDBSCAN | 0.000 | 0.000--0.004 | 0 | 11/13 |
| cytometry | OPTICS | 0.000 | 0.000--0.000 | 1 | 13/13 |
| cytometry | SNN-DBSCAN | 0.096 | 0.013--0.103 | 1 | 13/13 |
| cytometry | StrataSCAN | 0.035 | 0.013--0.035 | 1 | 13/13 |
| cytometry | VDBSCAN-2007 | 0.000 | 0.000--0.000 | 1 | 12/13 |
| cytometry | kNN+Leiden | 0.000 | 0.000--0.000 | 1 | 11/13 |
| cytometry | kNN-DBSCAN | 0.000 | 0.000--0.000 | 1 | 13/13 |
| gaia | AMD-inspired | 0.919 | 0.911--0.919 | 1 | 359/359 |
| gaia | DBSCAN | 0.972 | 0.961--0.972 | 0 | 359/359 |
| gaia | HDBSCAN | 0.869 | 0.861--0.869 | 1 | 359/359 |
| gaia | OPTICS | 0.936 | 0.758--0.961 | 5 | 359/359 |
| gaia | SNN-DBSCAN | 0.947 | 0.930--0.950 | 1 | 359/359 |
| gaia | StrataSCAN | 0.919 | 0.914--0.919 | 1 | 359/359 |
| gaia | VDBSCAN-2007 | 0.780 | 0.780--0.780 | 1 | 359/359 |
| gaia | kNN+Leiden | 0.864 | 0.852--0.864 | 1 | 359/359 |
| gaia | kNN-DBSCAN | 0.972 | 0.958--0.972 | 1 | 359/359 |
| noise_sweep | AMD-inspired | 0.167 | 0.000--0.167 | 1 | 30/30 |
| noise_sweep | DBSCAN | 0.000 | 0.000--0.017 | 4 | 30/30 |
| noise_sweep | HDBSCAN | 0.167 | 0.050--0.167 | 1 | 30/30 |
| noise_sweep | OPTICS | 0.000 | 0.000--0.000 | 4 | 30/30 |
| noise_sweep | SNN-DBSCAN | 0.017 | 0.000--0.133 | 3 | 30/30 |
| noise_sweep | StrataSCAN | 0.183 | 0.150--0.183 | 1 | 30/30 |
| noise_sweep | VDBSCAN-2007 | 0.200 | 0.167--0.200 | 1 | 30/30 |
| noise_sweep | kNN+Leiden | 0.133 | 0.000--0.133 | 1 | 30/30 |
| noise_sweep | kNN-DBSCAN | 0.133 | 0.000--0.217 | 4 | 30/30 |
| synthetic_development | AMD-inspired | 0.643 | 0.548--0.643 | 1 | 21/35 |
| synthetic_development | DBSCAN | 0.535 | 0.399--0.656 | 4 | 35/35 |
| synthetic_development | HDBSCAN | 0.602 | 0.455--0.602 | 1 | 35/35 |
| synthetic_development | OPTICS | 0.376 | 0.338--0.393 | 0 | 35/35 |
| synthetic_development | SNN-DBSCAN | 0.169 | 0.048--0.306 | 0 | 35/35 |
| synthetic_development | StrataSCAN | 0.833 | 0.721--0.837 | 0 | 35/35 |
| synthetic_development | VDBSCAN-2007 | 0.562 | 0.515--0.562 | 2 | 35/35 |
| synthetic_development | kNN+Leiden | 0.361 | 0.218--0.361 | 0 | 35/35 |
| synthetic_development | kNN-DBSCAN | 0.544 | 0.451--0.569 | 0 | 35/35 |
| synthetic_fresh | AMD-inspired | 0.667 | 0.569--0.667 | 0 | 12/21 |
| synthetic_fresh | DBSCAN | 0.563 | 0.411--0.639 | 2 | 21/21 |
| synthetic_fresh | HDBSCAN | 0.603 | 0.433--0.603 | 3 | 21/21 |
| synthetic_fresh | OPTICS | 0.347 | 0.319--0.385 | 1 | 21/21 |
| synthetic_fresh | SNN-DBSCAN | 0.159 | 0.032--0.290 | 0 | 21/21 |
| synthetic_fresh | StrataSCAN | 0.827 | 0.756--0.827 | 0 | 21/21 |
| synthetic_fresh | VDBSCAN-2007 | 0.579 | 0.520--0.579 | 1 | 21/21 |
| synthetic_fresh | kNN+Leiden | 0.357 | 0.214--0.357 | 1 | 21/21 |
| synthetic_fresh | kNN-DBSCAN | 0.540 | 0.474--0.567 | 2 | 21/21 |
| synthetic_scaling | StrataSCAN | 0.723 | 0.691--0.748 | 0 | 26/28 |

The CSV contains all four predeclared threshold pairs and both execution-macro and target-micro rates.
