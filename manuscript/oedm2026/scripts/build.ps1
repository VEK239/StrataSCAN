$ErrorActionPreference = 'Stop'

$manuscriptRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repositoryRoot = (Resolve-Path (Join-Path $manuscriptRoot '..\..')).Path
$analysisScript = Join-Path $manuscriptRoot 'scripts\analyze_results.py'
$revisionScript = Join-Path $manuscriptRoot 'scripts\revision_results.py'
$buildDirectory = Join-Path $manuscriptRoot 'build'
$tectonic = Join-Path $repositoryRoot '.tmp\tectonic\tectonic.exe'
$matplotlibCache = Join-Path $repositoryRoot '.tmp-mpl'

if (-not (Test-Path -LiteralPath $tectonic)) {
    throw "Tectonic was not found at $tectonic"
}

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $matplotlibCache | Out-Null
$env:MPLCONFIGDIR = $matplotlibCache

Push-Location $repositoryRoot
try {
    python $analysisScript | Tee-Object -FilePath (Join-Path $buildDirectory 'analysis_stdout.txt')
    python $revisionScript
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
    Copy-Item -LiteralPath (Join-Path $buildDirectory 'manuscript.pdf') `
        -Destination (Join-Path $manuscriptRoot 'MDL-StrataSCAN_OEDM2026_draft.pdf') -Force
}
finally {
    Pop-Location
}

Write-Output "Built: $(Join-Path $manuscriptRoot 'MDL-StrataSCAN_OEDM2026_draft.pdf')"
