# Final figure specification

Every numerical panel is generated from the five v0.2.4
`target-discovery-v1` freezes after semantic and provenance validation.

| Stem | Purpose |
|---|---|
| `fig1_method_pipeline` | Method blocks and exact local threshold event sweep. |
| `fig2_synthetic_validation` | Development/fresh joint-success target-F1 contrasts and absolute target-F1/discovery anchors. |
| `fig3_execution_envelope` | StrataSCAN-only 0.5--5M runtime, RSS, and quality envelope. |
| `fig4_biological_validation` | Equal-dataset/equal-study target F1, purity, coverage, discovery, burden, and completion. |
| `figS3_synthetic_metric_matrix` | Synthetic target F1, discovery, and true-noise F1 matrices. |
| `figS4_biological_metric_matrix` | Dataset-level biological one-to-one target F1. |
| `figS5_gaia_validation` | Gaia discovery, target components, unmatched candidates, and completion; current 0.2.4 StrataSCAN is separated in provenance from exact single-target-rescored legacy external predictions. |
| `figS7_noise_factorial` | Nine-method sensitivity at six controlled synthetic known-noise fractions. |

No panel imputes zero quality for failures. Synthetic true-noise F1 is not
relabeled as biological/Gaia noise. S2 model-specification and S6
solver-predecessor figures are historical, noncanonical, and absent.
