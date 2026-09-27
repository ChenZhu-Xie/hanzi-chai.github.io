[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Match = '*-annotation.html',

    [ValidateRange(1, 100)]
    [int]$Count = 1,

    [switch]$All,

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

$selectedPages = if ($All) {
    $candidatePages
} else {
    @($candidatePages | Select-Object -First $Count)
}

$selectedPages | ForEach-Object { Write-Output $_.FullName }

if (-not $PrintOnly) {
    $selectedPages | ForEach-Object {
        Start-Process -FilePath $_.FullName
    }
}
