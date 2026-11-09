# SPDX-License-Identifier: Apache-2.0
# Sync the workspace RWPP runtime (source of truth) into an ASCII execution
# path. Java's RWML self-attach (ASM agent) cannot open the runtime classpath
# when it contains the repository's Chinese path, so real-game runs must use
# an ASCII path. The workspace copy stays the editable source; this script
# mirrors the files the bridge/game actually reads.
#
#   pwsh -File scripts\sync_rwp_runtime.ps1
#   pwsh -File scripts\sync_rwp_runtime.ps1 -TargetRuntime "C:\IronZeroRuntime"
param(
    [string]$SourceRuntime = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path ".runtime\rwpp-runtime-1.15"),
    [string]$TargetRuntime = (Join-Path $env:TEMP "ironzero-rwpp-runtime-1.15")
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath (Join-Path $SourceRuntime "bin\rwpp-server.bat"))) {
    throw "source runtime not found: $SourceRuntime"
}
$ascii = (($TargetRuntime.ToCharArray() | Where-Object { [int]$_ -gt 127 }).Count -eq 0)
if (-not $ascii) {
    throw "TargetRuntime must be ASCII-only for the RWML ASM agent: $TargetRuntime"
}

New-Item -ItemType Directory -Force -Path $TargetRuntime | Out-Null
# First sync (or a repaired runtime) copies everything; later syncs only refresh
# the jars/assets that teammates rebuild.
if (-not (Test-Path -LiteralPath (Join-Path $TargetRuntime "bin\rwpp-server.bat"))) {
    Copy-Item -LiteralPath $SourceRuntime -Destination $TargetRuntime -Recurse -Force
} else {
    $extensionSource = Join-Path $SourceRuntime "extensions"
    $extensionTarget = Join-Path $TargetRuntime "extensions"
    New-Item -ItemType Directory -Force -Path $extensionTarget | Out-Null
    if (Test-Path -LiteralPath $extensionSource) {
        Copy-Item -Path (Join-Path $extensionSource "*") -Destination $extensionTarget -Recurse -Force
    }
    $mapsSource = Join-Path $SourceRuntime "assets\maps\ironzero"
    $mapsTarget = Join-Path $TargetRuntime "assets\maps\ironzero"
    if (Test-Path -LiteralPath $mapsSource) {
        New-Item -ItemType Directory -Force -Path $mapsTarget | Out-Null
        Copy-Item -Path (Join-Path $mapsSource "*") -Destination $mapsTarget -Recurse -Force
    }
}

Write-Host "RWPP runtime synced:"
Write-Host "  source: $SourceRuntime"
Write-Host "  target: $TargetRuntime"
$jar = Join-Path $TargetRuntime "extensions\ironzero-bridge.jar"
if (Test-Path -LiteralPath $jar) {
    $item = Get-Item -LiteralPath $jar
    Write-Host ("  bridge jar: {0} bytes  {1}" -f $item.Length, $item.LastWriteTime)
}
Get-ChildItem -LiteralPath (Join-Path $TargetRuntime "assets\maps\ironzero") -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host ("  map asset: {0} ({1} bytes)" -f $_.Name, $_.Length) }
