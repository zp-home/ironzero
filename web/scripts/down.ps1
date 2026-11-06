# SPDX-License-Identifier: Apache-2.0
# Stop the platform container.
$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$compose = Join-Path $repo "web/docker-compose.yml"
Set-Location $repo
docker compose -f $compose down
