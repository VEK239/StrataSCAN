# PILOT / replicated topology and density stress benchmark

Manifest rows: 684. Statuses: {'ok': 560, 'skipped_resource_limit': 76, 'timeout': 48}.

Sampling seeds: [23, 42, 73, 151]; four workers; geometry seed: 3571.

The Gaussian reference has six noise levels. The topology screens are deliberately separate from the density-ratio screens, so topology is not confounded with the 16x/64x density stress.

All cells retain six 300-point targets and fixed support within topology. The design audit records total n, local target/background contrast, support area, background density, edge separation, and geometry identity. AMD-DBSCAN is an explicit skip in every cell.

Figures are exploratory; metrics are aggregated over four sampling seeds where methods completed. Timeouts and errors remain in status tables and are not treated as missing favorable results.
