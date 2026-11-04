[CmdletBinding()]
param(
    [string]$RuntimeDir = $env:RWPP_RUNTIME_DIR,
    [string]$JavaHome = $env:JAVA_HOME,
    [int]$Port = 5123
)

$ErrorActionPreference = "Stop"
$checks = [System.Collections.Generic.List[object]]::new()

function Add-Check {
    param(
        [string]$Name,
        [bool]$Ok,
        [string]$Detail
    )

    $checks.Add([pscustomobject]@{
        Check = $Name
        Ok = $Ok
        Detail = $Detail
    })
}

function Resolve-JavaExe {
    param([string]$JavaHomePath)

    if ($JavaHomePath) {
        $candidate = Join-Path $JavaHomePath "bin\java.exe"
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
        return $null
    }

    $command = Get-Command java -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    return $null
}

$javaExe = Resolve-JavaExe -JavaHomePath $JavaHome
if (-not $javaExe) {
    Add-Check "Java executable" $false "Java not found. Set JAVA_HOME or install JDK 25."
} else {
    $javaText = (& $javaExe -version 2>&1 | Out-String)
    $major = 0
    if ($javaText -match 'version\s+"(?<major>\d+)') {
        $major = [int]$Matches.major
    }
    Add-Check "Java executable" $true $javaExe
    Add-Check "Java major version" ($major -ge 25) ("found {0}; RWPP server documentation currently requires Java 25" -f $major)
}

if (-not $RuntimeDir) {
    Add-Check "RWPP runtime directory" $false "Set RWPP_RUNTIME_DIR or pass -RuntimeDir."
} else {
    $runtimeExists = Test-Path -LiteralPath $RuntimeDir -PathType Container
    Add-Check "RWPP runtime directory" $runtimeExists $RuntimeDir

    if ($runtimeExists) {
        $serverBat = Join-Path $RuntimeDir "bin\rwpp-server.bat"
        $assetsDir = Join-Path $RuntimeDir "assets"
        $extensionsDir = Join-Path $RuntimeDir "extensions"
        Add-Check "rwpp-server.bat" (Test-Path -LiteralPath $serverBat) $serverBat
        Add-Check "assets directory" (Test-Path -LiteralPath $assetsDir -PathType Container) $assetsDir
        Add-Check "extensions directory" (Test-Path -LiteralPath $extensionsDir -PathType Container) $extensionsDir
    }
}

$portListener = $null
try {
    $portListener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    $portListener.Start()
    Add-Check "loopback port $Port" $true "available"
} catch {
    Add-Check "loopback port $Port" $false $_.Exception.Message
} finally {
    if ($portListener) {
        $portListener.Stop()
    }
}

$checks | Format-Table -AutoSize
$blockers = @($checks | Where-Object { -not $_.Ok })
if ($blockers.Count -gt 0) {
    Write-Error ("Environment check failed with {0} blocker(s)." -f $blockers.Count) -ErrorAction Continue
    exit 1
}

Write-Output "Environment check passed."
