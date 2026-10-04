# RQ3 scaling deliverables

This directory contains the audited source data and IEEE single-column artifacts for the million-scale Results subsection.

- `rq3_results.md`: manuscript-ready subsection, one-sentence answer, caption, exclusions, and exact 5M endpoints.
- `fig_rq3_million_scale.pdf`: 3.45 × 2.18 in vector figure.
- `fig_rq3_million_scale.png`: 600-dpi preview.
- `fig_rq3_million_scale.svg`: editable vector copy.
- `table_rq3_scaling.tex`: compact IEEE-style comparison table.
- `stratascan_scaling_cells.csv`: all 28 canonical StrataSCAN cells.
- `stratascan_by_size.csv`: per-size StrataSCAN summary.
- `scaling_by_method_size.csv`: per-method/size completion and success-only runtime audit.
- `five_million_imbalanced_matched.csv`: the only directly matched recorded 5M family across StrataSCAN, kNN-DBSCAN, SNN-DBSCAN, and other recorded outcomes.
- `validation.json`: structural checks and supported claim.
- `source_provenance.csv`: source paths, sizes, and SHA-256 fingerprints.
- `build_rq3_scaling.py`: reproducible builder with assertions.

The baseline campaign is explicitly marked incomplete at 5M, and its scaling jobs do not provide clustering-quality metrics. This prevents unsupported universal speed or quality claims.
