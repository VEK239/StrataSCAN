# Changelog

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
