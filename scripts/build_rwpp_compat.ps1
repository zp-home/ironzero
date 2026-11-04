# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GameDir,
    [string]$BuildRoot = (Join-Path $env:LOCALAPPDATA "RWRL"),
    [string]$JavaHome,
    [string]$ProxyUrl,
    [switch]$ReuseSources,
    [switch]$ReuseRuntime,
    [switch]$SkipSmoke,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $repoRoot "vendor\rwpp\pinned-versions.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json

function Write-Step { param([string]$Message); Write-Output "`n==> $Message" }

function Invoke-Download {
    param([string]$Uri, [string]$OutFile)
    Write-Output "Download: $Uri"
    $curlArgs = @("-L", "--fail", "--retry", "3", "--show-error", "-o", $OutFile)
    if ($ProxyUrl) { $curlArgs = @("-x", $ProxyUrl) + $curlArgs }
    $curlArgs += $Uri
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) { throw "Download failed with exit code $LASTEXITCODE`: $Uri" }
}

function Set-RequiredReplacement {
    param([string]$Path, [string]$OldValue, [string]$NewValue)
    $text = [IO.File]::ReadAllText($Path)
    if (-not $text.Contains($OldValue)) {
        if ($text.Contains($NewValue)) { Write-Output "Already patched: $Path"; return }
        throw "Expected text was not found in $Path`: $OldValue"
    }
    [IO.File]::WriteAllText($Path, $text.Replace($OldValue, $NewValue), [Text.UTF8Encoding]::new($false))
    Write-Output "Patched: $Path"
}

function Invoke-GradleBuild {
    param([string]$WorkingDirectory, [string[]]$Arguments, [string]$Label)
    Write-Output "Building $Label"
    Push-Location $WorkingDirectory
    try {
        & .\gradlew.bat @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Label build failed with exit code $LASTEXITCODE" }
    } finally { Pop-Location }
}

if ($PlanOnly) {
    Write-Output "Build root: $BuildRoot"
    Write-Output "RWPP commit: $($manifest.rwpp.commit)"
    Write-Output "RWML tag: $($manifest.rwml.tag)"
    Write-Output "game-lib: $($manifest.game_lib.version)"
    Write-Output "Game dir: $GameDir"
    Write-Output "Steps: download -> patch -> build RWML -> build RWPP -> assemble runtime -> smoke test"
    exit 0
}

$buildPathChars = $BuildRoot.ToCharArray() | Where-Object { [int]$_ -gt 127 }
if ($buildPathChars.Count -gt 0) { throw "BuildRoot must use an ASCII-only path: $BuildRoot" }
if (-not (Test-Path -LiteralPath $GameDir -PathType Container)) { throw "GameDir does not exist: $GameDir" }
$assetsSource = Join-Path $GameDir "assets"
if (-not (Test-Path -LiteralPath $assetsSource -PathType Container)) { throw "Game assets directory not found: $assetsSource" }

if (-not $JavaHome) {
    $javaExe = Get-ChildItem -LiteralPath (Join-Path $repoRoot ".runtime\jdk25") -Filter java.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($javaExe) { $JavaHome = Split-Path (Split-Path $javaExe.FullName -Parent) -Parent }
}
if (-not $JavaHome -or -not (Test-Path -LiteralPath (Join-Path $JavaHome "bin\java.exe"))) {
    throw "JavaHome must point to Java 25. Run .\scripts\bootstrap_java25.ps1 first."
}
$javaOutput = (& (Join-Path $JavaHome "bin\java.exe") -version 2>&1 | Out-String)
if ($javaOutput -notmatch 'version\s+"(?<major>\d+)' -or [int]$Matches.major -lt $manifest.java.major) {
    throw "Java $($manifest.java.major)+ is required. Found:`n$javaOutput"
}

$env:JAVA_HOME = $JavaHome
$env:PATH = (Join-Path $JavaHome "bin") + ";" + $env:PATH
if ($ProxyUrl) {
    $env:HTTP_PROXY = $ProxyUrl
    $env:HTTPS_PROXY = $ProxyUrl
    $proxyHost = ([Uri]$ProxyUrl).Host
    $proxyPort = ([Uri]$ProxyUrl).Port
    $env:JAVA_TOOL_OPTIONS = "-Dhttp.proxyHost=$proxyHost -Dhttp.proxyPort=$proxyPort -Dhttps.proxyHost=$proxyHost -Dhttps.proxyPort=$proxyPort"
    $env:GRADLE_OPTS = $env:JAVA_TOOL_OPTIONS
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
$downloads = Join-Path $BuildRoot "downloads"
New-Item -ItemType Directory -Force -Path $downloads | Out-Null
$rwppArchive = Join-Path $downloads "RWPP-$($manifest.rwpp.commit).tar.gz"
$rwmlArchive = Join-Path $downloads "RWML-$($manifest.rwml.commit).tar.gz"
$gameLibJar = Join-Path $downloads "game-lib-$($manifest.game_lib.version).jar"
$gameLibSources = Join-Path $downloads "game-lib-$($manifest.game_lib.version)-sources.jar"
$gameLibPom = Join-Path $downloads "game-lib-$($manifest.game_lib.version).pom"

Write-Step "Downloading pinned third-party sources and mapped game library"
if (-not (Test-Path -LiteralPath $rwppArchive)) { Invoke-Download -Uri "https://codeload.github.com/RWPP-Team/RWPP/tar.gz/$($manifest.rwpp.commit)" -OutFile $rwppArchive }
if (-not (Test-Path -LiteralPath $rwmlArchive)) { Invoke-Download -Uri "https://codeload.github.com/RWPP-Team/RWML/tar.gz/$($manifest.rwml.commit)" -OutFile $rwmlArchive }
$mavenBase = "https://repo1.maven.org/maven2/io/github/rwpp-team/game-lib/$($manifest.game_lib.version)"
if (-not (Test-Path -LiteralPath $gameLibJar)) { Invoke-Download -Uri "$mavenBase/game-lib-$($manifest.game_lib.version).jar" -OutFile $gameLibJar }
if (-not (Test-Path -LiteralPath $gameLibSources)) { Invoke-Download -Uri "$mavenBase/game-lib-$($manifest.game_lib.version)-sources.jar" -OutFile $gameLibSources }
if (-not (Test-Path -LiteralPath $gameLibPom)) { Invoke-Download -Uri "$mavenBase/game-lib-$($manifest.game_lib.version).pom" -OutFile $gameLibPom }

$actualGameLibHash = (Get-FileHash -LiteralPath $gameLibJar -Algorithm SHA256).Hash.ToUpperInvariant()
if ($actualGameLibHash -ne $manifest.game_lib.sha256.ToUpperInvariant()) { throw "game-lib checksum mismatch: $actualGameLibHash" }

$rwppSource = Join-Path $BuildRoot "RWPP-$($manifest.rwpp.commit.Substring(0, 10))"
$rwmlSource = Join-Path $BuildRoot "RWML-$($manifest.rwml.commit.Substring(0, 10))"
if (-not (Test-Path -LiteralPath $rwppSource)) {
    tar.exe -xzf $rwppArchive -C $BuildRoot
    $extracted = Get-ChildItem -LiteralPath $BuildRoot -Directory | Where-Object { $_.Name -like "RWPP-*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $extracted) { throw "RWPP source extraction failed" }
    Move-Item -LiteralPath $extracted.FullName -Destination $rwppSource
} elseif (-not $ReuseSources) { throw "RWPP source already exists. Pass -ReuseSources: $rwppSource" }

if (-not (Test-Path -LiteralPath $rwmlSource)) {
    tar.exe -xzf $rwmlArchive -C $BuildRoot
    $extracted = Get-ChildItem -LiteralPath $BuildRoot -Directory | Where-Object { $_.Name -like "RWML-*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $extracted) { throw "RWML source extraction failed" }
    Move-Item -LiteralPath $extracted.FullName -Destination $rwmlSource
} elseif (-not $ReuseSources) { throw "RWML source already exists. Pass -ReuseSources: $rwmlSource" }

Write-Step "Applying compatibility patches"
Set-RequiredReplacement -Path (Join-Path $rwppSource "gradle\libs.versions.toml") -OldValue 'game-lib = "v2-1.15"' -NewValue 'game-lib = "v3-1.15"'
Set-RequiredReplacement -Path (Join-Path $rwmlSource "gradle\libs.versions.toml") -OldValue 'game-lib = "v2-1.15"' -NewValue 'game-lib = "v3-1.15"'
Set-RequiredReplacement -Path (Join-Path $rwppSource "rwpp-api\src\commonMain\kotlin\io\github\rwpp\api\game\impl\GamePlatform.kt") -OldValue "LocaleEngine.b(plainName)" -NewValue "LocaleEngine.translateMapName(plainName)"
$daemonJvm = Join-Path $rwppSource "gradle\gradle-daemon-jvm.properties"
if (Test-Path -LiteralPath $daemonJvm) { Set-RequiredReplacement -Path $daemonJvm -OldValue "toolchainVendor=JETBRAINS" -NewValue "toolchainVendor=MICROSOFT" }

Write-Step "Preparing local Maven repository for RWML"
$localMaven = Join-Path $rwmlSource "build\local-maven\io\github\rwpp-team\game-lib\$($manifest.game_lib.version)"
New-Item -ItemType Directory -Force -Path $localMaven | Out-Null
Copy-Item -LiteralPath $gameLibJar -Destination (Join-Path $localMaven "game-lib-$($manifest.game_lib.version).jar") -Force
Copy-Item -LiteralPath $gameLibSources -Destination (Join-Path $localMaven "game-lib-$($manifest.game_lib.version)-sources.jar") -Force
Copy-Item -LiteralPath $gameLibPom -Destination (Join-Path $localMaven "game-lib-$($manifest.game_lib.version).pom") -Force

Write-Step "Building RWML desktop"
Invoke-GradleBuild -WorkingDirectory $rwmlSource -Arguments @(":desktop:publishToMavenLocal", "--no-daemon") -Label "RWML desktop"

Write-Step "Building RWPP server"
Invoke-GradleBuild -WorkingDirectory $rwppSource -Arguments @(":rwpp-server:installDist", "--no-daemon") -Label "RWPP server"

$runtimeDir = Join-Path $BuildRoot "rwpp-runtime-1.15"
if (Test-Path -LiteralPath $runtimeDir) {
    if (-not $ReuseRuntime) { throw "Runtime directory already exists. Pass -ReuseRuntime: $runtimeDir" }
} else { New-Item -ItemType Directory -Path $runtimeDir | Out-Null }

Write-Step "Assembling runtime"
$dist = Join-Path $rwppSource "rwpp-server\build\install\rwpp-server"
Copy-Item -Path (Join-Path $dist "*") -Destination $runtimeDir -Recurse -Force
Copy-Item -LiteralPath $assetsSource -Destination (Join-Path $runtimeDir "assets") -Recurse -Force
New-Item -ItemType Directory -Force -Path (Join-Path $runtimeDir "extensions") | Out-Null

if (-not $SkipSmoke) {
    Write-Step "Running environment check and smoke test"
    & (Join-Path $repoRoot "scripts\check_environment.ps1") -RuntimeDir $runtimeDir -JavaHome $JavaHome -Port 5123
    & (Join-Path $repoRoot "scripts\smoke_rwpp.ps1") -RuntimeDir $runtimeDir -JavaHome $JavaHome -Port 5123 -Ticks 180 -TimeoutSeconds 180
}

Write-Output "`nRWPP compatibility runtime is ready: $runtimeDir"
