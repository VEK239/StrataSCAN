# Changelog

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
