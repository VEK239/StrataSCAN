$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = "$repo;$repo\src"

$protocol = Join-Path $repo "benchmarks\protocol.v0.2.3-final-figures-stratascan-robustness.json"
$evidence = Join-Path $repo "results\published\v0.2.3\final_synthetic_robustness.csv"
$galleryDirectory = Join-Path $repo "results\published\v0.2.3\figures"
$finalDirectory = Join-Path $repo "manuscript\oedm2026\final_figures"

& python (Join-Path $repo "scripts\freeze_final_figure_evidence.py")
& python (Join-Path $repo "scripts\make_baseline_parameter_profiles.py")
& python (Join-Path $repo "manuscript\oedm2026\scripts\final_manuscript_statistics.py")

& python (Join-Path $repo "scripts\make_algorithm_behavior_grid.py") `
    --protocol $protocol `
    --benchmark-results $evidence `
    --output (Join-Path $galleryDirectory "algorithm-behavior-grid-n50000-a") `
    --n 50000 `
    --seed 42 `
    --max-display-points 10000 `
    --method DBSCAN `
    --method HDBSCAN `
    --method OPTICS `
    --method SNN-DBSCAN `
    --method VDBSCAN-2007 `
    --method StrataSCAN

& python (Join-Path $repo "scripts\make_algorithm_behavior_grid.py") `
    --protocol $protocol `
    --benchmark-results $evidence `
    --output (Join-Path $galleryDirectory "algorithm-behavior-grid-n50000-b") `
    --n 50000 `
    --seed 42 `
    --max-display-points 10000 `
    --method AMD-DBSCAN `
    --method kNN-DBSCAN `
    --method kNN+Leiden `
    --method StrataSCAN

& python (Join-Path $repo "manuscript\oedm2026\scripts\final_manuscript_figures.py") `
    --output $finalDirectory

Write-Output "Final figure set written to $finalDirectory"
