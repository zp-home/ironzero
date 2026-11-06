# SPDX-License-Identifier: Apache-2.0
# Build and start the platform container.
param(
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$compose = Join-Path $repo "web/docker-compose.yml"
Set-Location $repo

if ($NoBuild) {
    docker compose -f $compose up -d
} else {
    docker compose -f $compose up -d --build
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "docker compose failed with exit code $LASTEXITCODE"
}

Start-Sleep -Seconds 2
Write-Host "IronZero platform: http://127.0.0.1:8000"
Write-Host "Health:  http://127.0.0.1:8000/api/health"
