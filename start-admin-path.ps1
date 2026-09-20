param(
    [switch]$NoBrowser,
    [ValidateRange(1, 65535)]
    [int]$Port = 5174,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
Set-Location -LiteralPath $ProjectDir

if (-not (Test-Path -LiteralPath (Join-Path $ProjectDir "node_modules"))) {
    throw "Dependencies are missing. Run 'bun install' once, then launch this script again."
}

$bun = Get-Command bun.exe -ErrorAction SilentlyContinue
if ($null -eq $bun) {
    $bun = Get-Command bun -ErrorAction SilentlyContinue
}
if ($null -eq $bun) {
    throw "bun was not found on PATH."
}

$BaseUrl = "http://${HostAddress}:$Port/"
$AdminUrl = "http://${HostAddress}:$Port/admin"

function Get-HanziChaiMode {
    try {
        $response = Invoke-WebRequest -Uri $BaseUrl -UseBasicParsing -TimeoutSec 2
        if ($response.Content -notlike "*汉字自动拆分系统*") {
            return $null
        }
        $module = Invoke-WebRequest -Uri "${BaseUrl}src/utils.ts" -UseBasicParsing -TimeoutSec 2
        if ($module.Content -match '"MODE"\s*:\s*"([^"]+)"') {
            return $Matches[1]
        }
        return "UNKNOWN"
    } catch {
        return $null
    }
}

$runningMode = Get-HanziChaiMode
if ($runningMode -eq "CF") {
    Write-Host "Reusing the running hanzi-chai server: $AdminUrl"
    if (-not $NoBrowser) {
        Start-Process $AdminUrl
    }
    exit 0
}
if ($null -ne $runningMode) {
    throw "Port $Port is running hanzi-chai in $runningMode mode. Stop it before starting CF mode."
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $processIds = $listener | Select-Object -ExpandProperty OwningProcess -Unique
    throw "Port $Port is already occupied by PID(s): $($processIds -join ', '). Stop that process or pass a different -Port."
}

if (-not $NoBrowser) {
    Start-Job -ArgumentList $BaseUrl, $AdminUrl -ScriptBlock {
        param($ReadyUrl, $OpenUrl)
        for ($attempt = 0; $attempt -lt 120; $attempt++) {
            try {
                $response = Invoke-WebRequest -Uri $ReadyUrl -UseBasicParsing -TimeoutSec 1
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                    Start-Process $OpenUrl
                    return
                }
            } catch {}
            Start-Sleep -Milliseconds 250
        }
    } | Out-Null
}

Write-Host "Starting CF mode (BrowserRouter): $AdminUrl"
& $bun.Source x vite --mode CF --host $HostAddress --port $Port --strictPort
if ($LASTEXITCODE -ne 0) {
    throw "Vite CF mode failed with exit code $LASTEXITCODE."
}
