# OEDM 2026 manuscript package

This directory contains the anonymous IEEE ICDM OEDM draft, its compiled PDF,
publication figures, result tables, bibliography, analysis code, and visual-QA
renders.

## Primary artifacts

- `manuscript.tex` — IEEE conference manuscript source.
- `references.bib` — bibliography used by the manuscript.
- `MDL-StrataSCAN_OEDM2026_draft.pdf` — stable compiled draft for review.
- `figures/` — every figure in vector PDF/SVG and high-resolution PNG form.
- `tables/` — manuscript-ready LaTeX tables plus machine-readable CSV/JSON data.
- `scripts/analyze_results.py` — regenerates the aggregate figures, tables,
  confidence intervals, paired tests, and method-ranking matrices.
- `scripts/revision_results.py` — regenerates the per-dataset biological
  segmentation analysis and original five-seed synthetic quality/resource
  matrix.
- `scripts/build.ps1` — regenerates analysis artifacts and compiles the PDF.
- `build/` — LaTeX outputs and analysis console output.
- `artifact_manifest.json` — evaluated configuration, endpoint provenance, and
  immutable result/protocol hashes.
- `REVIEW_AND_REVISION.md` — adversarial reviewer findings, corrections, and
  remaining limitations.
- `SUPPLEMENT.md` and `supplement/figures/` — consolidated secondary
  diagnostics and their interpretation.
- `qa/submission_verified/` — 150-dpi renders of every page of the exact stable
  PDF used for final visual inspection.

## Evidence included in this draft

The evidence is deliberately separated by evaluation stage:

- the new 0.2.3 release robustness grid: seven families, five sizes, and five
  seeds (175 StrataSCAN jobs), with all jobs completed, a 22.6x median paired
  speedup against the conservative solver, and retained negative cells;
- completed serial 0.2.3 multidensity scaling measurements through 2,000,000
  points; the remaining five-million-point campaign cells are explicitly
  incomplete and are not reported as endpoints;
- the original 315-job synthetic reference matrix: seven families, five
  pre-freeze seeds, nine methods, `n = 5,000`, with quality, segmentation,
  runtime, and process-memory measurements;
- 14 post-freeze synthetic revision pairs: unseen seeds 197 and 251 at `n = 5,000`,
  comparing MDL-StrataSCAN only with its immediate 0.1.2 predecessor;
- all 13 biological/cytometry datasets (1,547,871 cells total), reported
  separately with target recovery, reference-background identification, and
  segmentation diagnostics; macro target F1 is the declared endpoint, while
  target--background and target--background--structure harmonic F1 are
  explicitly post-hoc sensitivity diagnostics;
- the prespecified 24-field Gaia DR3 pilot;
- conventional graph/density baselines where protocol coverage permits.

The scalability evidence is deliberately separated by algorithm generation and
protocol. The 0.2.3 release has completed direct serial multidensity runs
through 2,000,000 points; its other scaling cells remain in progress. The
historical fixed-range snapshot has direct monitored runs through the
396,460-cell Mosmann dataset. The one-million-point dev9 smoke test and the
five-million-point 0.1.2 matrix are retained as lineage evidence, not attributed
to the submitted solver. The historical scalability evidence table records
this distinction explicitly.

The incomplete 359-field Gaia sweep is not included in any result, table, test,
or conclusion. Its paused checkpoints remain outside this manuscript directory
and can be resumed as a separate validation stage.

The five-seed synthetic matrix is explicitly described as diagnostic/development
evidence; it is not pooled with the locked two-seed ablation. No newly generated
baseline measurements at seeds 197 or 251 are used.

## Rebuild

From the repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File manuscript\oedm2026\scripts\build.ps1
```

The analysis requires the repository's Python scientific stack. The build script
uses `.tmp/tectonic/tectonic.exe`, the project-local Tectonic binary used for the
checked PDF. The final output is copied to
`manuscript/oedm2026/MDL-StrataSCAN_OEDM2026_draft.pdf`.

## Submission status

The draft is formatted in IEEE two-column workshop style and uses anonymous
author metadata for triple-blind review. Before submission, replace or confirm
the workshop year/track metadata requested by the final EasyChair form, archive
immutable benchmark manifests and result hashes, and prepare an anonymized code
artifact. The manuscript distinguishes strong biological recovery evidence from
segmentation/background limitations and the mixed Gaia pilot rather than making
a universal-superiority claim.

The checked 0.2.3 PDF is ten US-Letter pages including references. Its SHA-256
digest is `95fd4970b9925677081403ceecd0e26cea4fbcb5afed9077bb7c68b370322a15`.
