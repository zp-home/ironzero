# SPDX-License-Identifier: Apache-2.0
# Build the IronZero RWPP server extension (stage 1 observation bridge).
#
#   .\scripts\build_rwpp_extension.ps1 `
#     -RuntimeDir "E:\IronZeroBuild\rwpp-runtime-1.15" `
#     -JavaHome "E:\jdk25copy"
#
# Requires an RWPP runtime built by scripts/build_rwpp_compat.ps1 (Java 25).
param(
    [string]$RuntimeDir = "E:\IronZeroBuild\rwpp-runtime-1.15",
    [string]$JavaHome = "E:\jdk25copy"
)

$ErrorActionPreference = "Continue"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $repo "src\ironzero_bridge\java"
$classes = Join-Path $repo ".runtime\bridge-classes"
$meta = Join-Path $repo ".runtime\bridge-meta\META-INF"

if (-not (Test-Path -LiteralPath (Join-Path $JavaHome "bin\javac.exe"))) {
    throw "javac not found under JavaHome=$JavaHome (Java 25 required)"
}
if (-not (Test-Path -LiteralPath (Join-Path $RuntimeDir "lib\game-lib-v3-1.15.jar"))) {
    throw "RWPP runtime not found under RuntimeDir=$RuntimeDir (run scripts\build_rwpp_compat.ps1 first)"
}

Remove-Item -LiteralPath $classes -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Split-Path $meta -Parent) -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $classes, $meta | Out-Null

@"
id=io.github.ironzero.bridge
name=IronZero RWPP Bridge
version=0.1.0
mainClass=ironzero.IronZeroBridge
dependsOn=io.github.rwpp.server
tags=server
"@ | Set-Content -LiteralPath (Join-Path $meta "extension.properties") -Encoding ascii

$sources = Get-ChildItem -LiteralPath $source -Recurse -Filter "*.java" | ForEach-Object { $_.FullName }
& (Join-Path $JavaHome "bin\javac.exe") -encoding UTF-8 -cp (Join-Path $RuntimeDir "lib\*") -d $classes @sources
if ($LASTEXITCODE -ne 0) { throw "extension compilation failed" }

$jarPath = Join-Path $RuntimeDir "extensions\ironzero-bridge.jar"
& (Join-Path $JavaHome "bin\jar.exe") --create --file $jarPath -C $classes . -C (Split-Path $meta -Parent) .
if ($LASTEXITCODE -ne 0) { throw "extension packaging failed" }
Write-Host "IronZero extension ready: $jarPath"
