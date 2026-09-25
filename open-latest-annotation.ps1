[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Match = '*-annotation.html',

    [switch]$PrintOnly
)

$ErrorActionPreference = 'Stop'

$artifactDirectory = Join-Path $PSScriptRoot 'artifacts\unihan-import'
if (-not (Test-Path -LiteralPath $artifactDirectory -PathType Container)) {
    throw "Annotation directory does not exist: $artifactDirectory"
}

$candidatePages = @(
    Get-ChildItem -LiteralPath $artifactDirectory -File -Filter $Match |
        Where-Object Name -Like '*-annotation.html' |
        Sort-Object -Property `
            @{ Expression = 'LastWriteTimeUtc'; Descending = $true },
            @{ Expression = 'Name'; Descending = $true }
)

if ($candidatePages.Count -eq 0) {
    throw "No annotation page matching '$Match' was found in $artifactDirectory"
}

$latestPage = $candidatePages[0]
Write-Output $latestPage.FullName

if (-not $PrintOnly) {
    Start-Process -FilePath $latestPage.FullName
}
