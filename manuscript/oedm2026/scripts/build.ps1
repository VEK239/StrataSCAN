$ErrorActionPreference = 'Stop'

$manuscriptRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repositoryRoot = (Resolve-Path (Join-Path $manuscriptRoot '..\..')).Path
$buildDirectory = Join-Path $manuscriptRoot 'build'
$finalFigures = Join-Path $manuscriptRoot 'final_figures'
$tectonic = Join-Path $repositoryRoot '.tmp\tectonic\tectonic.exe'
$matplotlibCache = Join-Path $repositoryRoot '.tmp-mpl'
$promotedPdf = Join-Path $manuscriptRoot 'StrataSCAN_OEDM2026_draft.pdf'

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & python $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed with exit code $LASTEXITCODE`: $Script"
    }
}

if (-not (Test-Path -LiteralPath $tectonic)) {
    throw "Tectonic was not found at $tectonic"
}

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $matplotlibCache | Out-Null
$env:MPLCONFIGDIR = $matplotlibCache
$env:PYTHONPATH = "$repositoryRoot;$repositoryRoot\src"

Push-Location $repositoryRoot
try {
    # The JSON summary is the sole numerical prose source. Synchronization is
    # deliberately performed before figures and checked again before LaTeX.
    Invoke-CheckedPython (Join-Path $manuscriptRoot 'scripts\final_manuscript_statistics.py')
    Invoke-CheckedPython (Join-Path $manuscriptRoot 'scripts\discovery_threshold_sensitivity.py')
    Invoke-CheckedPython (Join-Path $manuscriptRoot 'scripts\synchronize_final_manuscript_statistics.py')

    Invoke-CheckedPython (Join-Path $manuscriptRoot 'scripts\final_manuscript_figures.py') `
        --output $finalFigures
    # Build the plain study-by-method RQ4 table from canonical evidence.
    Invoke-CheckedPython (Join-Path $repositoryRoot 'scripts\build_biological_validation_latex_table.py')
    Invoke-CheckedPython (Join-Path $repositoryRoot 'scripts\build_revised_results_presentation.py')
    Invoke-CheckedPython (Join-Path $repositoryRoot 'outputs\rq3_scaling\build_ieee_ultrasparse_16d_scaling.py')
    Invoke-CheckedPython (Join-Path $manuscriptRoot 'scripts\synchronize_final_manuscript_statistics.py') `
        --check
}
finally {
    Pop-Location
}

Push-Location $manuscriptRoot
try {
    & $tectonic -X compile manuscript.tex --outdir build --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic compilation failed with exit code $LASTEXITCODE"
    }
    $builtPdf = Join-Path $buildDirectory 'manuscript.pdf'
    if (-not (Test-Path -LiteralPath $builtPdf)) {
        throw "Tectonic did not create $builtPdf"
    }
    Copy-Item -LiteralPath $builtPdf -Destination $promotedPdf -Force
}
finally {
    Pop-Location
}

Push-Location $repositoryRoot
try {
    Invoke-CheckedPython (Join-Path $manuscriptRoot 'scripts\generate_artifact_manifest.py')
    Invoke-CheckedPython (Join-Path $manuscriptRoot 'scripts\generate_artifact_manifest.py') `
        --check
}
finally {
    Pop-Location
}

Write-Output "Built and manifested: $promotedPdf"

