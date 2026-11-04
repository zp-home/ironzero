[CmdletBinding()]
param(
    [string]$Destination,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Destination) {
    $Destination = Join-Path $root ".runtime\jdk25"
}

$javaCandidates = Get-ChildItem -LiteralPath $Destination -Filter java.exe -Recurse -ErrorAction SilentlyContinue
if ($javaCandidates -and -not $Force) {
    $existing = $javaCandidates | Select-Object -First 1
    Write-Output "Java 25 runtime already present: $($existing.FullName)"
    exit 0
}
if ($Force) {
    throw "Force refresh is intentionally not implemented. Remove the destination explicitly if you need to reinstall."
}

$url = "https://aka.ms/download-jdk/microsoft-jdk-25.0.5.0.2-windows-x64.zip"
$zip = Join-Path $root ".runtime\microsoft-jdk-25.zip"
New-Item -ItemType Directory -Force -Path (Split-Path $zip) | Out-Null
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

Write-Output "Downloading Microsoft OpenJDK 25..."
& curl.exe -L --fail --retry 3 --silent --show-error -o $zip $url
if ($LASTEXITCODE -ne 0) {
    throw "JDK download failed with exit code $LASTEXITCODE"
}

Write-Output "Extracting..."
Expand-Archive -LiteralPath $zip -DestinationPath $Destination -Force
$java = Get-ChildItem -LiteralPath $Destination -Filter java.exe -Recurse | Select-Object -First 1
if (-not $java) {
    throw "java.exe was not found after extraction"
}

$version = (& $java.FullName -version 2>&1 | Out-String)
if ($version -notmatch 'version\s+"(?<major>\d+)') {
    throw "Unable to parse Java version"
}
if ([int]$Matches.major -lt 25) {
    throw "Downloaded Java is older than 25"
}

Write-Output "Java 25 ready: $($java.FullName)"
