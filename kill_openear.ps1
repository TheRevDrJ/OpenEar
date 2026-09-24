# OpenEar — Real-time AI captioning and translation for churches
# Copyright (c) 2026 TheRevDrJ
# Licensed under AGPL-3.0 — see LICENSE file for details
#
# Stops ONLY OpenEar's own pythonw process — the one running server.py — and
# nothing else. Replaces an older wmic-based query in openear.bat; wmic is
# deprecated and absent from recent Windows builds, so we use Get-CimInstance.
#
# Safety (see BOB rules.md §12 — process-kills are rm -rf tier, kill narrow
# never broad): a process is a target ONLY if its command line contains
# "server.py". An empty match list kills nothing — there is no fall-through to
# a broad match. Each PID is validated > 0 before Stop-Process is called with
# that explicit -Id. Exits with the number of processes killed, so the caller
# (openear.bat stop) can tell whether anything was running.
#
#   -Quiet    suppress the per-process "Killing..." line (used by start cleanup)
#   -DryRun   report what WOULD be killed, kill nothing (verification)

param([switch]$Quiet, [switch]$DryRun)

$targets = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'pythonw.exe' -or $_.Name -eq 'pythonw3.13.exe') -and
    $_.CommandLine -and ($_.CommandLine -like '*server.py*')
}

$count = 0
foreach ($p in $targets) {
    if ($p.ProcessId -gt 0) {
        if (-not $Quiet) {
            $verb = if ($DryRun) { 'Would kill' } else { 'Killing' }
            Write-Host "  $verb stale OpenEar process (PID: $($p.ProcessId))..."
        }
        if (-not $DryRun) {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
        $count++
    }
}

exit $count
