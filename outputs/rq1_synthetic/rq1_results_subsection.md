# RQ1 synthetic-quality result (audited draft)

## Research question

**RQ1. Does StrataSCAN provide competitive target recovery across heterogeneous synthetic datasets?**

## Manuscript-ready subsection

### A. Recovery across heterogeneous synthetic structures

We evaluated StrataSCAN and eight baselines on seven synthetic families spanning
non-convex geometry, multiple densities, severe class imbalance, high-dimensional
overlap, and sparse targets. All methods were evaluated with the frozen
`target-discovery-v1` contract: predicted clusters were matched one-to-one to
targets by Hungarian assignment maximizing pairwise F1. The analysis below uses
the three prespecified fresh seeds (21 dataset--seed cells per method); the five
development seeds were kept separate.

StrataSCAN completed all 21 fresh cells and obtained mean macro target F1 of
0.828, median macro target F1 of 0.931, and mean target-discovery rate of 0.827.
Among methods completing the entire matrix, this was the highest mean target F1
and the highest mean discovery rate. In cell-wise paired comparisons,
StrataSCAN had a positive median F1 difference against seven of eight baselines:
DBSCAN (+0.221), OPTICS (+0.448), SNN-DBSCAN (+0.870), VDBSCAN-2007 (+0.065),
kNN-DBSCAN (+0.112), kNN+Leiden (+0.146), and AMD-DBSCAN (+0.001 among 12 jointly
successful cells). The only negative paired median was against HDBSCAN
(-0.007), which also exceeded StrataSCAN on 12 of 21 cells. AMD-DBSCAN completed
only 12 of 21 fresh cells because nine runs reached the 8-GiB memory limit, so
its conditional quality must not be interpreted as full-matrix performance.
Together, these results support competitive, robust target recovery across the
heterogeneous benchmark rather than universal superiority on every family.

**One-sentence answer.** StrataSCAN achieved the strongest average target
recovery among methods completing all fresh synthetic cells and a positive
paired median F1 difference against seven of eight baselines, while HDBSCAN
retained a small median advantage.

## Recommended main-paper display

Use one compact single-column table; do not add a separate RQ1 figure. The table
reports all nine methods, completion, and conditional quality on successful
fresh cells. Conditional AMD-DBSCAN values are marked with a dagger because 9/21
cells failed by memory limit.

| Method | Complete | Mean target F1 | Median target F1 | Mean discovery |
|---|---:|---:|---:|---:|
| AMD-DBSCAN | 12/21 | 0.843† | 0.956† | 0.667† |
| DBSCAN | 21/21 | 0.551 | 0.539 | 0.563 |
| HDBSCAN | 21/21 | 0.776 | **0.964** | 0.603 |
| OPTICS | 21/21 | 0.374 | 0.393 | 0.347 |
| SNN-DBSCAN | 21/21 | 0.087 | 0.068 | 0.159 |
| **StrataSCAN** | **21/21** | **0.828** | 0.931 | **0.827** |
| VDBSCAN-2007 | 21/21 | 0.671 | 0.815 | 0.579 |
| kNN+Leiden | 21/21 | 0.619 | 0.595 | 0.357 |
| kNN-DBSCAN | 21/21 | 0.539 | 0.536 | 0.540 |

† Conditional on successful cells; AMD-DBSCAN reached the prespecified 8-GiB
memory limit in nine fresh cells. Failures are `NA`, not zero-quality results.

### IEEE LaTeX table

```latex
\begin{table}[t]
\caption{Synthetic target recovery on three fresh seeds across seven families.}
\label{tab:synthetic_fresh}
\centering
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrr}
\toprule
Method & Done & Mean F1 & Med. F1 & Disc.\\
\midrule
AMD-DBSCAN$^{\dagger}$ & 12/21 & .843 & .956 & .667\\
DBSCAN                 & 21/21 & .551 & .539 & .563\\
HDBSCAN                & 21/21 & .776 & \textbf{.964} & .603\\
OPTICS                  & 21/21 & .374 & .393 & .347\\
SNN-DBSCAN              & 21/21 & .087 & .068 & .159\\
\textbf{StrataSCAN}     & \textbf{21/21} & \textbf{.828} & .931 & \textbf{.827}\\
VDBSCAN-2007            & 21/21 & .671 & .815 & .579\\
kNN+Leiden              & 21/21 & .619 & .595 & .357\\
kNN-DBSCAN              & 21/21 & .539 & .536 & .540\\
\bottomrule
\multicolumn{5}{p{0.96\columnwidth}}{\scriptsize $^{\dagger}$Conditional on successful cells; nine runs reached the 8-GiB memory limit. Failures are not scored as zero.}
\end{tabular}
\end{table}
```

## Caption

**Table I. Synthetic target recovery on three fresh seeds across seven
families.** Macro target F1 uses one-to-one Hungarian target matching. Means,
medians, and discovery rates are conditional on successful cells, with
completion reported explicitly. Bold denotes the best value among methods that
completed all 21 cells; AMD-DBSCAN values are conditional on 12 completed cells.

## Claims that must be excluded

- Do not claim StrataSCAN is best on every family: it is not the top fresh-seed
  family median in any of the seven families under this table's best-baseline
  definition.
- Do not claim a universal advantage over HDBSCAN: HDBSCAN has a 0.007 paired
  median advantage and wins 12/21 jointly successful fresh cells.
- Do not compare AMD-DBSCAN's conditional mean directly as if it covered all 21
  cells; nine fresh runs failed by memory limit.
- Do not replace failures with target F1 = 0 in this table. Report completion
  and conditional quality separately.
- Do not mix development and fresh seeds into one inferential sample. The
  protocol declares them as separate evidence stages.
- Do not use old many-to-one, AMI-only, ARI-only, or legacy noise-aware metrics
  as the primary result. Use `macro_target_f1` from `target-discovery-v1`.
- Do not use Gaia or the abandoned optimization study in RQ1.
- Do not say the seven-family matrix proves rare-target robustness through 99%
  noise; that claim belongs to the separate controlled rare-dense experiment.

## Evidence and reproducibility

- Canonical rows: `results/published/v0.2.4/target-discovery-v1/synthetic_release.csv`
- Canonical SHA-256: `37a52224a55b7d9bc0723a095dba8f1b9fe6cc82dbc240b42b83a46dc23a37c7`
- Provenance: `results/published/v0.2.4/target-discovery-v1/synthetic_release-provenance.json`
- Protocol: `benchmarks/protocol.v0.2.4-target-discovery-v1-synthetic.json`
- Audit script: `outputs/rq1_synthetic/audit_rq1_synthetic.py`
- Derived all-method summary: `outputs/rq1_synthetic/rq1_method_summary.csv`
- Derived family audit: `outputs/rq1_synthetic/rq1_fresh_family_table.csv`
- Derived paired comparisons: `outputs/rq1_synthetic/rq1_fresh_paired_deltas.csv`

Validation command:

```powershell
python outputs/rq1_synthetic/audit_rq1_synthetic.py
```

The script verifies the canonical file hash, row and status counts, method and
seed coverage, evidence-stage partition, evaluation protocol, Hungarian
matching, and cell uniqueness before producing the summaries.
