# ============================================================================
#  install_shortcuts.ps1 - OpenEar shortcuts in the Start menu and on the desktop
# ============================================================================
#
#  WHAT IT DOES
#    Creates an "OpenEar" folder holding four shortcuts, in the Start menu,
#    on the desktop, or both:
#      Start OpenEar     openear.bat start
#      Stop OpenEar      openear.bat stop
#      Restart OpenEar   openear.bat restart - also how OpenEar sees a microphone
#                        plugged in after it started
#      OpenEar Admin     the admin page, in the default browser
#    The first three open a console that shows what happened and waits for a
#    key, so a volunteer reads "OpenEar is RUNNING" (or why not) instead of
#    watching a window flash past.
#
#  FOR EVERY ACCOUNT ON THE PC: the all-users Start menu and the Public desktop.
#    Setup runs as an administrator, and on a presentation PC the person who
#    runs the service is often a different, ordinary account.
#
#  NEVER DELETES ANYTHING. Shortcuts of these four names are rewritten so they
#    point at this install; anything else in the folders is left alone.
#
#  CALLED BY: setup.bat, which decides whether to call it (first install, or
#    --shortcuts). Safe to run by hand as an administrator:
#      powershell -ExecutionPolicy Bypass -File install_shortcuts.ps1 -StartMenu -Desktop
#
#  PURE ASCII ON PURPOSE: PowerShell 5.1 reads a BOM-less UTF-8 script as ANSI.
# ============================================================================

param(
    [switch]$StartMenu,
    [switch]$Desktop,
    [string]$InstallDir = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'
$InstallDir = (Resolve-Path $InstallDir).Path
$bat  = Join-Path $InstallDir 'openear.bat'
$icon = Join-Path $InstallDir 'static\favicon.ico'
$cmd  = Join-Path $env:SystemRoot 'System32\cmd.exe'

if (-not (Test-Path $bat)) {
    Write-Host "  [FAIL] openear.bat not found in $InstallDir"
    exit 1
}

$places = @()
if ($StartMenu) { $places += @{ Name = 'Start menu'; Dir = [Environment]::GetFolderPath('CommonPrograms') } }
if ($Desktop)   { $places += @{ Name = 'desktop';    Dir = [Environment]::GetFolderPath('CommonDesktopDirectory') } }
if ($places.Count -eq 0) {
    Write-Host '  [SKIP] No shortcuts requested.'
    exit 0
}

$shell = New-Object -ComObject WScript.Shell
$failed = $false

foreach ($place in $places) {
    try {
        $folder = Join-Path $place.Dir 'OpenEar'
        New-Item -ItemType Directory -Path $folder -Force | Out-Null

        foreach ($verb in @('Start', 'Stop', 'Restart')) {
            $lnk = $shell.CreateShortcut((Join-Path $folder "$verb OpenEar.lnk"))
            $lnk.TargetPath = $cmd
            # cmd strips the outer quotes of /c "...", leaving: "<bat>" verb & pause
            $lnk.Arguments = '/c ""' + $bat + '" ' + $verb.ToLower() + ' & echo. & pause"'
            $lnk.WorkingDirectory = $InstallDir
            $lnk.Description = "$verb the OpenEar server"
            if (Test-Path $icon) { $lnk.IconLocation = "$icon,0" }
            $lnk.Save()
        }

        # A .url, not a .lnk: it opens in whatever browser the PC uses.
        $url = @('[InternetShortcut]', 'URL=http://localhost/admin')
        if (Test-Path $icon) { $url += @("IconFile=$icon", 'IconIndex=0') }
        Set-Content -Path (Join-Path $folder 'OpenEar Admin.url') -Value $url -Encoding ASCII

        Write-Host "  [OK] OpenEar folder added to the $($place.Name)"
    } catch {
        Write-Host "  [WARN] Could not add shortcuts to the $($place.Name): $($_.Exception.Message)"
        $failed = $true
    }
}

if ($failed) { exit 1 }
exit 0
