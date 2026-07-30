# Adversarial review and revision record

This record treats the paper as an IEEE ICDM OEDM submission and separates
problems that were corrected from limitations that must remain visible to a
reviewer. No benchmark was rerun with replacement seeds. The original
five-seed development matrix and the frozen post-revision outputs are
unchanged; new numbers below are diagnostics recomputed from those outputs.

## Round 1 — initial recommendation: reject / major revision

| Severity | Reviewer finding | Why it threatened validity | Correction |
|---|---|---|---|
| Critical | The target–background harmonic score was presented as a biological primary endpoint even though it was motivated after observing that Leiden assigns every point. | This was undisclosed post-hoc endpoint selection. | Restored macro target F1 as the declared cytometry endpoint. Renamed the new score TB-HF1, marked it post-hoc in the abstract, metrics, tables, figures, protocol, and artifact manifest, and retained its two components in every dataset row. |
| Critical | Ten Samusik samples were counted as ten independent datasets, while Levine, Mosmann, and Nilsson counted once each. | The nominal 13-dataset average gives one study 77% of the weight and can manufacture a rank. | Added an equal-study sensitivity analysis that averages Samusik first. The conclusion now reports the rank reversal: MDL-StrataSCAN is 0.160 versus 0.163 for SNN-DBSCAN, with only four study units and no robust first-place claim. |
| Critical | The manuscript threshold code used `-log |V_s|`, but the evaluated implementation uses `-log n`. | The stated objective was not the evaluated algorithm. | Corrected the mathematical statement to `-log n` and explained that the threshold identity is coded against the global sample size. |
| Critical | The mutable `StrataSCAN` method alias could select a newer adaptive component search rather than the evaluated 1–8 component snapshot. | The paper could not be reproduced from its protocol identifier. | Paper protocols now use explicit versioned identifiers. `StrataSCAN-Optimization` is dispatched with `max_components=8`, `hard_max_components=8`, `adaptive_components=False`, and `entropy_scope="basis"`; the predecessor also has an explicit alias. |
| Major | “Noise” in cytometry and Gaia was treated as physical truth. | Reference-negative or “other” objects can contain legitimate unannotated biology or astronomy. | Biological terminology now uses reference background, reports background precision/recall/F1, and states that this label is not verified physical noise. Gaia receives the same caveat. Synthetic injected noise remains the only verified noise label. |
| Major | Macro target F1 permits several target classes to select the same predicted cluster. | An all-assigned or merged partition can score well without providing a useful segmentation. | Disclosed the many-to-one matching rule and paired it with pairwise F1, predicted/annotated cluster ratio, target fragmentation, target merging, and background-majority cluster counts. |
| Major | The graph topology term was described too strongly as a Bayes factor / full MDL probability. | kNN edges are dependent, so the score is not a calibrated likelihood for the joint partition. | Reframed it as an integrated edge-surprisal or composite codelength heuristic and explicitly limited the global-optimality claim to blockwise decisions. |
| Major | A full algorithm revision was called an ablation. | Multiple components changed, so causal attribution to a single mechanism was unsupported. | Renamed it a post-freeze synthetic revision comparison and added “controlled component ablations” to required future work. |
| Major | The biological aggregate hid dataset-specific segmentation failures. | Levine fragmentation and Samusik over-segmentation contradicted the aggregate recovery claim. | Added one row per dataset, four diagnostic panels, and dataset-specific interpretation for Levine, Mosmann, Nilsson, and Samusik 01–10. |
| Major | Synthetic runtime, memory, and baseline evidence had been displaced by the biological discussion. | The submission no longer supported its efficiency and baseline context. | Restored the complete nine-method, five-seed synthetic matrix with quality, pairwise F1, predicted-cluster ratio, runtime, incremental memory, failures, and separate locked revision results. |

### Round-1 algorithm changes

- Added the canonical `target_background_hmean_f1` diagnostic with finite-value
  and range validation; the old function name remains only as a compatibility
  alias.
- Restored `macro_target_f1` as the historical benchmark primary, so future
  runs cannot silently optimize the post-hoc diagnostic.
- Pinned the evaluated estimator configuration in method dispatch and in both
  replay protocols.
- Added an artifact manifest with result and protocol SHA-256 hashes.

## Round 2 — recommendation after corrections: weak accept

The corrected paper has a defensible workshop contribution: an explicit
blockwise codelength formulation, an exact within-stratum event sweep, and a
useful negative result showing that target recovery, background rejection, and
segmentation granularity can disagree. It no longer supports universal
superiority, and it does not claim that the post-hoc biological diagnostic is
confirmatory.

The second pass found two remaining fixable issues:

1. The biological diagnostic was deferred by IEEE wide-float rules until after
   the conclusion. Wide-float placement was adjusted so the dataset table and
   figure now precede the concluding material.
2. A forced page break left most of the penultimate page empty. It was removed,
   and the final content and references were balanced. The checked paper is now
   nine US-Letter pages without smaller body text or rasterized tables.

### Remaining weaknesses that are disclosed, not papered over

- The locked synthetic revision comparison contains two unseen seeds per
  family. The earlier baseline matrix keeps the original five seeds, but it is
  development evidence and is not pooled with the locked comparison.
- The four-study biological analysis is too small for a strong inferential
  claim. Its paired interval spans zero and the method is second under equal
  study weighting.
- The current algorithm over-segments Levine and most Samusik samples, loses
  background recall on 12 of 13 biological outputs, and does not improve the
  prespecified Gaia pilot primary.
- No controlled component ablation isolates the Gamma model, event sweep,
  topology code, background scan, and border rule.
- The fixed `k=32`, shell ranks `(4,8,16,32)`, ambient-dimension Poisson model,
  and 1–8 component search are modeling and numerical controls, not learned
  universally optimal values.
- The dependent-edge topology score has no family-wise false-activation
  guarantee for the repeated background scan and no measured global-optimality
  gap.
- The full 359-field Gaia study was paused and is intentionally absent from all
  claims.

These limitations prevent a strong accept. They do not invalidate the narrower
OEDM paper because the title, abstract, evaluation, discussion, and conclusion
now state the same scoped contribution and expose the principal failure modes.

## Round 3 — post-tag evidence audit

The preceding manuscript and code were frozen at commit `8c3db22` and annotated
tag `v0.2.0` before this round. A fresh review against that immutable baseline
found three additional presentation/provenance problems:

1. The paper's sparse complexity bound was correct, but the narrative mixed a
   20,000-point run of the submitted fixed-range snapshot, a one-million-point
   dev9 prototype, and five-million-point 0.1.2 predecessor runs. The revision
   adds an evidence-boundary table. The largest job attributable to the
   submitted snapshot is now reported directly: Mosmann, 396,460 cells, 517.0
   seconds, and 646.7 MiB monitored peak process RSS. The larger records are
   explicitly labeled as different algorithms.
2. The fixed-range benchmark profile called its Gamma-basis and background
   models “adaptive,” although component expansion was disabled. The profile
   strings now describe the frozen configuration exactly; the public adaptive
   8–24 search and the evaluated 1–8 search are machine-distinguishable.
3. Three uncited figures duplicated main-table ranks and paired effects. They
   were consolidated into one supplemental diagnostic covering study-weighted
   biology, signed Gaia field changes, locked synthetic trade-offs, and direct
   pilot runtime versus size. All source tables and negative cases remain.

No frozen seed, prediction, score, or endpoint changed. The resulting 0.2.1
work is a provenance, reporting, and artifact-organization revision rather than
a retrospectively tuned algorithm.

## Verification checklist

- Focused release-profile, public-API, and optimization tests: 19 passed.
- Full project test suite: 44 passed.
- Offline source and wheel builds completed for version 0.2.1.
- Both replay protocols and the artifact manifest parse as JSON.
- Every table and figure is regenerated from the frozen result files.
- The stable PDF compiles without unresolved references or horizontal overflow.
- Every page of the exact stable PDF is rendered and visually inspected.
- Anonymous author text and PDF metadata contain no author identity.
