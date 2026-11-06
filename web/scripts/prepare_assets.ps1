# SPDX-License-Identifier: Apache-2.0
# Copy the local Rusted Warfare 1.15 assets needed by the sandbox renderer
# into a git-ignored .game_cache/ directory.
#
#   .\web\scripts\prepare_assets.ps1 -GameDir "E:\qwq\强化学习笔记\Rusted Warfare1.15PC"
#
# The cache is local-only: game assets are copyrighted and must never be
# committed or redistributed.
param(
    [Parameter(Mandatory = $true)]
    [string]$GameDir
)

$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$assets = Join-Path $GameDir "assets"
if (-not (Test-Path -LiteralPath $assets)) {
    Write-Error "assets/ not found under $GameDir"
}
$cache = Join-Path $repo ".game_cache"
New-Item -ItemType Directory -Force $cache | Out-Null

$tilesets = Join-Path $assets "tilesets"
$maps = Join-Path $assets "maps"
$tankSprites = Join-Path $assets "units\tanks"
foreach ($pair in @(
        @{ Source = $tilesets; Target = (Join-Path $cache "tilesets") },
        @{ Source = $maps; Target = (Join-Path $cache "maps") }
    )) {
    if (Test-Path -LiteralPath $pair.Source) {
        if (Test-Path -LiteralPath $pair.Target) { Remove-Item -LiteralPath $pair.Target -Recurse -Force }
        Copy-Item -LiteralPath $pair.Source -Destination $pair.Target -Recurse -Force
    }
}
$unitTarget = Join-Path $cache "units\tanks"
New-Item -ItemType Directory -Force $unitTarget | Out-Null
if (Test-Path -LiteralPath $tankSprites) {
    Copy-Item -Path (Join-Path $tankSprites "*.png") -Destination $unitTarget -Force
}

$mapCount = (Get-ChildItem -LiteralPath (Join-Path $cache "maps") -Recurse -Filter "*.tmx" -ErrorAction SilentlyContinue | Measure-Object).Count
$sizeMb = [math]::Round(((Get-ChildItem -LiteralPath $cache -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "game asset cache ready: $cache ($mapCount maps, $sizeMb MB)"
Write-Host "Docker: docker compose -f web/docker-compose.yml up -d --build"
