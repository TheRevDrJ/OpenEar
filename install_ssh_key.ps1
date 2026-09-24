# ============================================================================
#  install_ssh_key.ps1 - authorize a public key for passwordless admin SSH
# ============================================================================
#
#  WHAT IT DOES
#    Appends the public key(s) in -KeyFile to Windows' admin authorized_keys
#    file, then hardens that file's ACL. Windows OpenSSH IGNORES the file
#    unless it is owned by SYSTEM/Administrators with no other entries, so the
#    ACL step is not optional decoration - without it SSH silently refuses
#    the key and you get a password prompt with no explanation.
#
#  CALLED BY: setup.bat, and ONLY when run as `setup.bat --remote`.
#    Standalone use is the repair path if the ACL got reset.
#
#  WHY THE KEY IS NOT IN THIS FILE, OR IN setup.bat
#    setup.bat used to carry a maintainer's public key inline. A public
#    installer that does that grants that maintainer administrator login on
#    every machine which runs it - the --remote flag makes it opt-in, but the
#    person opting in has no idea whose key they are authorizing. The key now
#    comes from ssh_authorized_key.txt, which is gitignored, so it is always
#    the operator's own key and never ships with a clone.
#    @decision:gold 2026-09-24
#
#  WHY IT APPENDS INSTEAD OF OVERWRITING
#    The old inline version used Set-Content, which silently destroyed any
#    admin keys already authorized on that machine - a church's own IT
#    support, for instance. This adds only keys that are not already present
#    and leaves everything else alone.
#    @decision:silver 2026-09-24
#
#  PURE ASCII ON PURPOSE: PowerShell 5.1 is the default SSH shell on these
#  boxes and reads a BOM-less UTF-8 script as ANSI, which turns a dash into a
#  string delimiter and dies far from the real cause.
# ============================================================================

param(
    [Parameter(Mandatory = $true)]
    [string]$KeyFile
)

$ErrorActionPreference = 'Stop'
$target = Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'

if (-not (Test-Path $KeyFile)) {
    Write-Host "  [ERROR] Key file not found: $KeyFile"
    exit 1
}

# Accept one key per line; ignore blanks and # comments.
$wanted = Get-Content $KeyFile |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith('#') }

if (-not $wanted) {
    Write-Host "  [ERROR] No keys found in $KeyFile"
    exit 1
}

$existing = @()
if (Test-Path $target) {
    $existing = Get-Content $target | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

# Compare on the key material itself, not the whole line: the trailing comment
# is free text and the same key can arrive with a different one.
function Get-KeyBody([string]$line) {
    $parts = $line -split '\s+'
    if ($parts.Count -ge 2) { return $parts[0] + ' ' + $parts[1] }
    return $line
}
$have = @($existing | ForEach-Object { Get-KeyBody $_ })
$toAdd = @($wanted | Where-Object { $have -notcontains (Get-KeyBody $_) })

if ($toAdd.Count -eq 0) {
    Write-Host "  [OK] Key(s) already authorized - nothing to add"
} else {
    $all = @($existing) + @($toAdd)
    Set-Content -Path $target -Value $all -Encoding UTF8
    Write-Host "  [OK] Authorized $($toAdd.Count) key(s); kept $($existing.Count) already present"
}

# Windows OpenSSH requires SYSTEM + Administrators and nothing else.
$acl = Get-Acl $target
$acl.SetAccessRuleProtection($true, $false)
foreach ($rule in @($acl.Access)) { [void]$acl.RemoveAccessRule($rule) }
foreach ($who in @('SYSTEM', 'Administrators')) {
    $acl.AddAccessRule(
        (New-Object System.Security.AccessControl.FileSystemAccessRule($who, 'FullControl', 'Allow'))
    )
}
Set-Acl -Path $target -AclObject $acl
Write-Host "  [OK] Permissions restricted to SYSTEM and Administrators"
