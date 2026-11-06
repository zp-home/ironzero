# SPDX-License-Identifier: Apache-2.0
# Start the platform in local dev mode (no Docker).
param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$env:PYTHONPATH = Join-Path $repo "src"
Set-Location $repo
Write-Host "IronZero platform dev server: http://${BindHost}:$Port"
python -m uvicorn web.backend.app:app --host $BindHost --port $Port --reload
