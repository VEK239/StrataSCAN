# Rare dense target recovery under increasing background

Three fixed 100-observation targets were evaluated against low-density background at six fractions and five paired seeds. The empirical 16-neighbour signal/background density ratio remained approximately 8.4--17.2.

| Background | Total n | Target F1 median [min, max] | Discovery median [min, max] | True-noise F1 median [min, max] |
|---:|---:|---:|---:|---:|
| 25% | 400 | 0.981 [0.798, 0.995] | 1.000 [1.000, 1.000] | 0.823 [0.086, 0.985] |
| 50% | 600 | 0.922 [0.790, 0.992] | 1.000 [1.000, 1.000] | 0.933 [0.855, 0.992] |
| 75% | 1,200 | 0.926 [0.787, 0.993] | 1.000 [1.000, 1.000] | 0.978 [0.946, 0.998] |
| 90% | 3,000 | 0.894 [0.852, 0.993] | 1.000 [1.000, 1.000] | 0.990 [0.986, 0.998] |
| 95% | 6,000 | 0.852 [0.838, 0.991] | 1.000 [1.000, 1.000] | 0.993 [0.993, 1.000] |
| 99% | 30,000 | 0.979 [0.949, 0.992] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] |

StrataSCAN completed all 30 cells and discovered all three targets in every cell under the locked purity >= 0.90 and coverage >= 0.10 definition. This supports stability to increasing *global low-density background burden* at fixed target size and density contrast; it does not support robustness to locally overlapping background of comparable density.

AMD-DBSCAN completed 268/270 matrix cells overall; its two 99% failures were retained as memory-allocation errors. All other methods completed 30/30 cells.
