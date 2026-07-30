# Locked synthetic quality, noise, and ablation evidence

`benchmarks/protocol.v0.2.2-synthetic-evidence.json` is the reproducible
synthetic matrix for the v0.2.2 estimator.  It fixes five generated-data seeds,
12 scenario/noise conditions (four families at 25%, 50%, and 75% uniform
background), and all eight supported baseline implementations.  It also
includes the released estimator and four explicitly named controls: a fixed
component bound, a continuous background model, a rate-changepoint semantic
role, and the existing wide-graph control.

The resulting matrix is 780 jobs: 13 methods × 12 conditions × 5 seeds.  Each
job is isolated by the common runner.  Timeout, memory, and numerical failures
are saved as job records rather than removed from the matrix.

Run it from an editable environment that exposes this checkout's `src/`:

```powershell
python -m pip install -e ".[benchmark,perf]"
python -m benchmarks.run_benchmark `
  --protocol benchmarks/protocol.v0.2.2-synthetic-evidence.json `
  --suite synthetic `
  --output-dir results/runs/v0.2.2-locked-synthetic-evidence `
  --max-workers 1
```

Resume an interrupted matrix with the identical command plus `--resume`.  Do
not merge job directories from different checkouts or protocol revisions.

Only after the runner's `validation.json` is structurally complete, generate
the descriptive evidence package:

```powershell
python scripts/analyze_locked_synthetic_evidence.py `
  results/runs/v0.2.2-locked-synthetic-evidence `
  --output results/runs/v0.2.2-locked-synthetic-evidence/evidence
```

This produces seed-level records, condition summaries with median/IQR and a
deterministic bootstrap mean interval, all StrataSCAN-minus-comparator deltas,
reversal counts, and a target/background/structure figure.  Failed jobs receive
the protocol's zero penalty for quality measures only; structural counts remain
missing for failed jobs.  The report is descriptive: seed variation is not
treated as independent biological replication, and it does not choose away
reversals or failures.

No generated table or figure is committed with the protocol because a complete
v0.2.2 matrix is not available in this source snapshot.
