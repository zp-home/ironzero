[CmdletBinding()]
param(
    [string]$RuntimeDir = $env:RWPP_RUNTIME_DIR,
    [string]$JavaHome = $env:JAVA_HOME,
    [int]$Port = 5123,
    [int]$Ticks = 180,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $RuntimeDir) {
    throw "RuntimeDir is required. Pass -RuntimeDir or set RWPP_RUNTIME_DIR."
}
if (-not (Test-Path -LiteralPath $RuntimeDir -PathType Container)) {
    throw "RuntimeDir does not exist: $RuntimeDir"
}

$serverBat = Join-Path $RuntimeDir "bin\rwpp-server.bat"
$assetsDir = Join-Path $RuntimeDir "assets"
if (-not (Test-Path -LiteralPath $serverBat)) {
    throw "rwpp-server.bat not found: $serverBat"
}
if (-not (Test-Path -LiteralPath $assetsDir -PathType Container)) {
    throw "Game assets directory not found: $assetsDir"
}

if ($JavaHome) {
    $javaExe = Join-Path $JavaHome "bin\java.exe"
    if (-not (Test-Path -LiteralPath $javaExe)) {
        throw "java.exe not found under JavaHome: $JavaHome"
    }
    $env:JAVA_HOME = $JavaHome
    $env:PATH = (Join-Path $JavaHome "bin") + ";" + $env:PATH
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactDir = Join-Path $root "artifacts\smoke\$stamp"
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$stdoutPath = Join-Path $artifactDir "stdout.log"
$stderrPath = Join-Path $artifactDir "stderr.log"

function Stop-ProcessTree {
    param([int]$Id)

    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$Id" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -Id ([int]$child.ProcessId)
    }
    Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
}

$arguments = @("--no-console", "--port", "$Port", "--ticks", "$Ticks")
$process = Start-Process `
    -FilePath $serverBat `
    -ArgumentList $arguments `
    -WorkingDirectory $RuntimeDir `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath

$finished = $process.WaitForExit($TimeoutSeconds * 1000)
if (-not $finished) {
    Stop-ProcessTree -Id $process.Id
    throw "RWPP smoke test timed out after $TimeoutSeconds seconds. Artifacts: $artifactDir"
}

$exitCode = $process.ExitCode
$stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { "" }
$stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
$combined = $stdout + "`n" + $stderr
$severePattern = "Exception|SEVERE|FATAL|ERROR|FAILED|Could not|Unable to"
$severe = $combined -match $severePattern

$portAvailable = $false
$listener = $null
try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    $listener.Start()
    $portAvailable = $true
} catch {
    $portAvailable = $false
} finally {
    if ($listener) {
        $listener.Stop()
    }
}

$report = [pscustomobject]@{
    timestamp = $stamp
    runtime_dir = $RuntimeDir
    tick_limit = $Ticks
    exit_code = $exitCode
    severe_log_pattern = $severe
    port_available_after_exit = $portAvailable
    stdout = $stdoutPath
    stderr = $stderrPath
}
$reportPath = Join-Path $artifactDir "report.json"
$report | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $reportPath -Encoding utf8
$report | Format-List

if ($exitCode -ne 0) {
    throw "RWPP exited with code $exitCode. Report: $reportPath"
}
if ($severe) {
    throw "RWPP log contains error-like output. Report: $reportPath"
}
if (-not $portAvailable) {
    throw "Port $Port is still occupied after RWPP exit. Report: $reportPath"
}

Write-Output "RWPP smoke test passed. Report: $reportPath"
