@echo off
:: OpenEar — Real-time AI captioning and translation for churches
:: Copyright (c) 2026 TheRevDrJ
:: Licensed under AGPL-3.0 — see LICENSE file for details
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"

:: ============================================================================
:: OpenEar Setup Script
:: Installs all dependencies and configures Windows for OpenEar.
:: Must be run as Administrator (for firewall rule and long paths).
::
:: It asks ONE question: captions only, or captions plus translation. The answer
:: is recorded in mode.json (openear_config.py owns that file) and read at every
:: server start. Run this script again at any time to change it.
::
::   setup.bat                   asks. Pressing Enter keeps this machine's
::                               current mode, or picks captions only on a
::                               first install.
::   setup.bat --captions-only   no question: captions only
::   setup.bat --translation     no question: captions plus translation
::   setup.bat --remote          also installs remote-management tools
::
:: Captions only is the default because it is the one that cannot hurt the PC it
:: runs on: it never touches the graphics card. Translation takes about 4.6 GB of
:: an NVIDIA GPU's memory, which is only safe on a machine that can spare it.
::
:: THIS FILE IS PARSED BY cmd.exe WITH DELAYED EXPANSION ON. An exclamation mark
:: in any echo is silently eaten, and a bare ( or ) inside an if-block ends the
:: block early. Escape parentheses as ^( ^) there; never use exclamation marks.
:: It must also stay CRLF (.gitattributes enforces it on checkout): with LF line
:: endings cmd mis-parses labels and falls through into the wrong branch.
:: ============================================================================

echo.
echo   ============================================
echo     OpenEar Setup
echo   ============================================
echo.

:: ----------------------------------------------------------------------------
:: Arguments. Unknown ones stop setup rather than being ignored, because a
:: mistyped --captions-only would otherwise run the whole install in the mode
:: nobody asked for.
:: ----------------------------------------------------------------------------
set "REMOTE=0"
set "MODE_ARG="
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--remote" goto arg_remote
if /i "%~1"=="--captions-only" goto arg_captions
if /i "%~1"=="--translation" goto arg_translation
echo   ERROR: Unknown option: %~1
echo   Options: --captions-only   --translation   --remote
echo.
exit /b 2
:arg_remote
set "REMOTE=1"
goto next_arg
:arg_captions
if "!MODE_ARG!"=="translation" goto arg_conflict
set "MODE_ARG=captions"
goto next_arg
:arg_translation
if "!MODE_ARG!"=="captions" goto arg_conflict
set "MODE_ARG=translation"
goto next_arg
:arg_conflict
echo   ERROR: --captions-only and --translation cannot be used together.
echo.
exit /b 2
:next_arg
shift
goto parse_args
:args_done

:: ----------------------------------------------------------------------------
:: Check for admin privileges
:: ----------------------------------------------------------------------------
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: This script must be run as Administrator.
    echo   Right-click and select "Run as administrator".
    echo.
    pause
    exit /b 1
)
echo   [OK] Running as Administrator

:: ----------------------------------------------------------------------------
:: Look for an NVIDIA graphics card. Only translation needs one - captions run on
:: the processor - so this only informs the question below. It never stops a
:: captions-only install, which is the point: "no graphics card needed" has to be
:: true of the installer too, not just of the server.
:: ----------------------------------------------------------------------------
::
:: A card counts as found only if nvidia-smi -L actually LISTS one ("GPU 0: ...").
:: Its exit code is not trusted: a crashed nvidia-smi returns a negative code,
:: which "if errorlevel 1" reads as success.
set "HAS_GPU=0"
set "GPUNAME="
for /f "tokens=2 delims=:(" %%g in ('nvidia-smi -L 2^>nul ^| findstr /b /c:"GPU "') do if not defined GPUNAME set "GPUNAME=%%g"
if defined GPUNAME set "HAS_GPU=1"

:: ----------------------------------------------------------------------------
:: Check Python is installed
:: ----------------------------------------------------------------------------
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   ERROR: Python is not installed or not on PATH.
    echo.
    echo   Install Python 3.11+ from the Microsoft Store or python.org
    echo   Make sure "Add to PATH" is checked during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   [OK] Python %PYVER% found

:: ----------------------------------------------------------------------------
:: Check pip is available
:: ----------------------------------------------------------------------------
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   ERROR: pip is not available. Reinstall Python with pip included.
    echo.
    pause
    exit /b 1
)
echo   [OK] pip available

:: ----------------------------------------------------------------------------
:: Captions only, or captions plus translation?
::
:: Asked HERE, before anything is installed, so nobody is asked a question
:: twenty minutes into an install they walked away from - and so cancelling
:: leaves the machine exactly as it was.
:: ----------------------------------------------------------------------------
set "CURRENT_MODE=none"
for /f "delims=" %%m in ('python "%SCRIPT_DIR%openear_config.py" --current 2^>nul') do set "CURRENT_MODE=%%m"

echo.
echo   ============================================
echo     Captions only, or captions plus translation?
echo   ============================================
echo.
echo   Captions only      English captions on every phone. Runs on the
echo                      processor - no graphics card needed - so it can
echo                      share this PC with streaming or video software.
echo.
echo   Translation        The same captions, plus live translation into
echo                      200+ languages. Needs an NVIDIA graphics card with
echo                      6 GB or more and uses about 4.6 GB of it, so it
echo                      wants a PC of its own.
echo.
if "!HAS_GPU!"=="1" (
    echo   Graphics card found:!GPUNAME!
) else (
    echo   Graphics card found: none from NVIDIA
)
if "!CURRENT_MODE!"=="translation" echo   This PC is currently set up for: captions plus translation
if "!CURRENT_MODE!"=="captions" echo   This PC is currently set up for: captions only
if "!CURRENT_MODE!"=="none" echo   This PC has not chosen yet.
echo.

if defined MODE_ARG (
    set "MODE=!MODE_ARG!"
    echo   Chosen on the command line: !MODE_ARG!
    goto mode_answered
)

set "DEFAULT_MODE=captions"
if "!CURRENT_MODE!"=="translation" set "DEFAULT_MODE=translation"
set "PROMPT_HINT=y/N"
if "!DEFAULT_MODE!"=="translation" set "PROMPT_HINT=Y/n"

:ask_mode
:: ANSWER is cleared first on purpose. set /p leaves a variable UNCHANGED when it
:: reads nothing - no keyboard, or a script piping in no input - so without this
:: it would keep a stale value, or one inherited from the environment.
set "ANSWER="
set /p ANSWER="   Add translation? [!PROMPT_HINT!]: "
set "MODE=!DEFAULT_MODE!"
if "!ANSWER!"=="" goto mode_answered
if /i "!ANSWER!"=="y" goto answer_translation
if /i "!ANSWER!"=="yes" goto answer_translation
if /i "!ANSWER!"=="n" goto answer_captions
if /i "!ANSWER!"=="no" goto answer_captions
echo   Please answer y or n.
goto ask_mode
:answer_translation
set "MODE=translation"
goto mode_answered
:answer_captions
set "MODE=captions"
:mode_answered

:: Translation without an NVIDIA card cannot work. Say so now, before installing
:: anything, rather than let the server discover it at its first start.
if not "!MODE!"=="translation" goto mode_final
if "!HAS_GPU!"=="1" goto mode_final
echo.
echo   Translation needs an NVIDIA graphics card, and none was found:
echo   nvidia-smi is missing, or it found no GPU.
echo.
echo   If this PC does have an NVIDIA card, install the full driver from
echo   https://www.nvidia.com/drivers - not the basic one Windows Update
echo   installs - then restart and run setup.bat again.
echo.
if defined MODE_ARG (
    echo   Setup stopped: --translation was requested. Nothing was changed.
    echo.
    exit /b 1
)
set "ANSWER="
set /p ANSWER="   Set up captions only for now instead? [Y/n]: "
if /i "!ANSWER!"=="n" goto cancel_setup
if /i "!ANSWER!"=="no" goto cancel_setup
set "MODE=captions"
goto mode_final
:cancel_setup
echo.
echo   Setup cancelled. Nothing was changed.
echo.
pause
exit /b 1
:mode_final

if "!MODE!"=="translation" (
    echo   [OK] Setting up: captions plus translation
) else (
    echo   [OK] Setting up: captions only
)

:: ----------------------------------------------------------------------------
:: Enable Windows Long Paths (the NVIDIA CUDA packages in requirements.txt have
:: paths longer than Windows allows by default)
:: ----------------------------------------------------------------------------
echo.
echo   Enabling Windows long path support...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Long paths enabled
) else (
    echo   [WARN] Could not enable long paths. Some packages may fail to install.
)

:: ----------------------------------------------------------------------------
:: Enable Git long paths (if git is installed)
:: ----------------------------------------------------------------------------
git --version >nul 2>&1
if %errorlevel% equ 0 (
    git config --global core.longpaths true >nul 2>&1
    echo   [OK] Git long paths enabled
) else (
    echo   [INFO] Git not found - skipping git config
)

:: ----------------------------------------------------------------------------
:: Add firewall rule for port 80 (HTTP)
:: ----------------------------------------------------------------------------
echo.
echo   Configuring Windows Firewall...

:: Check if rule already exists
netsh advfirewall firewall show rule name="OpenEar HTTP" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Firewall rule already exists
    goto firewall_done
)

netsh advfirewall firewall add rule name="OpenEar HTTP" dir=in action=allow protocol=TCP localport=80 >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Firewall rule added ^(port 80 inbound^)
) else (
    echo   [WARN] Could not add firewall rule. Clients on other devices may not connect.
)

:: Also allow pythonw.exe through firewall (needed for WebSocket connections)
for /f "delims=" %%i in ('where pythonw 2^>nul') do set "PYTHONW_PATH=%%i"
if defined PYTHONW_PATH (
    netsh advfirewall firewall add rule name="OpenEar Python" dir=in action=allow program="%PYTHONW_PATH%" enable=yes >nul 2>&1
    echo   [OK] Firewall rule added for pythonw
)
:firewall_done

:: ----------------------------------------------------------------------------
:: Install Visual C++ Runtime (required by onnxruntime and CTranslate2)
:: ----------------------------------------------------------------------------
echo.
echo   Checking Visual C++ Runtime...
python -c "import ctypes; ctypes.CDLL('msvcp140.dll')" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Visual C++ Runtime found
    goto vcpp_done
)
echo   Visual C++ Runtime not found. Downloading installer...
curl -L -o "%TEMP%\vc_redist.x64.exe" "https://aka.ms/vs/17/release/vc_redist.x64.exe" >nul 2>&1
if not exist "%TEMP%\vc_redist.x64.exe" (
    echo   [FAIL] Could not download Visual C++ Runtime.
    echo   Please install manually from: https://aka.ms/vs/17/release/vc_redist.x64.exe
    pause
    exit /b 1
)
echo   Installing Visual C++ Runtime...
"%TEMP%\vc_redist.x64.exe" /install /quiet /norestart
if %errorlevel% equ 0 (
    echo   [OK] Visual C++ Runtime installed
) else (
    echo   [WARN] Visual C++ Runtime install may have failed.
    echo   If OpenEar won't start, install manually from:
    echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
)
:vcpp_done

:: ----------------------------------------------------------------------------
:: Create virtual environment
:: ----------------------------------------------------------------------------
echo.
echo   Creating Python virtual environment...
if exist "%SCRIPT_DIR%venv" (
    echo   [OK] Virtual environment already exists
) else (
    python -m venv "%SCRIPT_DIR%venv"
    if !errorlevel! neq 0 (
        echo   [FAIL] Could not create virtual environment.
        pause
        exit /b 1
    )
    echo   [OK] Virtual environment created
)

:: ----------------------------------------------------------------------------
:: Install Python dependencies from pinned requirements.txt
:: ----------------------------------------------------------------------------
echo.
echo   Installing Python packages (this may take several minutes)...
echo.

"%SCRIPT_DIR%venv\Scripts\pip.exe" install -r "%SCRIPT_DIR%requirements.txt" --quiet
if %errorlevel% neq 0 (
    echo   [FAIL] Package installation failed.
    pause
    exit /b 1
)
echo   [OK] All packages installed

:: ----------------------------------------------------------------------------
:: Translation only: check the CUDA runtime really loads. nvidia-smi can find a
:: card while the CUDA libraries still fail - Windows Update's basic display
:: driver does exactly that - and translation would then fail at every start.
:: ----------------------------------------------------------------------------
if not "!MODE!"=="translation" goto cuda_done
echo.
"%SCRIPT_DIR%venv\Scripts\python.exe" -c "import nvidia.cublas, os, ctypes; p=os.path.join(os.path.dirname(nvidia.cublas.__path__[0]),'cublas','bin'); ctypes.CDLL(os.path.join(p,'cublas64_12.dll')); print('OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   WARNING: CUDA runtime DLLs are not accessible.
    echo.
    echo   nvidia-smi detected your GPU, but the CUDA runtime libraries
    echo   ^(cublas64_12.dll^) could not be loaded. This means translation
    echo   will fail at runtime even though the driver appears installed.
    echo.
    echo   Fix: Install the full NVIDIA Game Ready or Studio Driver from:
    echo     https://www.nvidia.com/drivers
    echo.
    echo   Windows Update installs a basic display driver only - it does
    echo   NOT include the CUDA runtime. You need the full driver package
    echo   from nvidia.com. After installing, restart and run setup again.
    echo.
    echo   Captions will still work. The admin page will say why translation
    echo   is off.
    echo.
    goto cuda_done
)
echo   [OK] NVIDIA GPU and CUDA runtime ready for translation
:cuda_done

:: ----------------------------------------------------------------------------
:: Record the choice. Written only now, after the packages installed, so a
:: failed install never leaves a mode recorded for software that is not there.
:: ----------------------------------------------------------------------------
echo.
"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%openear_config.py" --write !MODE!
if %errorlevel% neq 0 (
    echo   [FAIL] Could not record the mode in mode.json.
    pause
    exit /b 1
)

:: ----------------------------------------------------------------------------
:: Pre-download AI models. download_models.py reads the mode just recorded: the
:: speech model always, the translation model only for translation.
:: ----------------------------------------------------------------------------
echo.
if "!MODE!"=="translation" (
    echo   Downloading AI models - about 16 GB, one-time download...
) else (
    echo   Downloading the speech model - about 2.5 GB, one-time download...
)
echo   This will take several minutes depending on your internet speed.
echo.

"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%download_models.py" 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Models downloaded
) else (
    echo   [WARN] Model download failed. Run setup.bat again with an internet
    echo          connection before the first service - OpenEar will not download
    echo          the translation model itself, and fetching the speech model at
    echo          first start can take longer than OpenEar waits.
)

:: ----------------------------------------------------------------------------
:: Remote management tools (optional, --remote flag)
:: ----------------------------------------------------------------------------
if not "!REMOTE!"=="1" goto remote_done

echo.
echo   ============================================
echo     Remote Management Tools
echo   ============================================
echo.

:: Install Git
git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Git already installed
) else (
    echo   Installing Git...
    winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] Git installed
    ) else (
        echo   [WARN] Git install failed. Install manually from https://git-scm.com
    )
)

:: Install Tailscale
tailscale version >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Tailscale already installed
) else (
    echo   Installing Tailscale...
    curl -L -o "%TEMP%\tailscale.msi" "https://pkgs.tailscale.com/stable/tailscale-setup-latest-amd64.msi" >nul 2>&1
    msiexec /i "%TEMP%\tailscale.msi" /quiet /norestart >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] Tailscale installed
        echo   NOTE: Open Tailscale from the Start menu and sign in to activate.
        echo   NOTE: After signing in, from your Tailscale admin console:
        echo         1. Add this machine to your network node
        echo         2. Tag it with the 'openear' tag
        echo         3. Enable unattended/headless mode so it stays connected without a logged-in user
    ) else (
        echo   [WARN] Tailscale install failed. Install manually from https://tailscale.com/download
    )
)

:: Enable OpenSSH Server
sc query sshd >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] OpenSSH Server already installed
) else (
    echo   Enabling OpenSSH Server...
    powershell -Command "Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] OpenSSH Server installed
    ) else (
        echo   [WARN] OpenSSH Server install failed.
    )
)

:: Start and auto-start SSH
powershell -Command "Start-Service sshd; Set-Service -Name sshd -StartupType Automatic" >nul 2>&1
echo   [OK] SSH service started and set to auto-start

:: Set PowerShell as the default SSH shell
powershell -Command "New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell -Value 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -PropertyType String -Force" >nul 2>&1
echo   [OK] PowerShell set as default SSH shell

:: Install SSH public key for passwordless login
:: For admin accounts, Windows SSH uses C:\ProgramData\ssh\administrators_authorized_keys
:: and ignores the file unless its ACL is locked down, which install_ssh_key.ps1 does.
::
:: THE KEY IS NOT IN THIS FILE. It is read from ssh_authorized_key.txt beside this
:: script, which is gitignored. A public installer carrying a maintainer's key grants
:: that maintainer administrator login on every machine that runs it, and the person
:: opting in has no way to know whose key it is. Put your OWN public key there.
set "KEYFILE=%SCRIPT_DIR%ssh_authorized_key.txt"
if exist "%KEYFILE%" goto install_ssh_key
echo   [SKIP] No ssh_authorized_key.txt found, so no SSH key was authorized.
echo          For passwordless login, put your own public key in:
echo            %KEYFILE%
goto ssh_key_done
:install_ssh_key
echo   Authorizing SSH public key from ssh_authorized_key.txt...
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_ssh_key.ps1" -KeyFile "%KEYFILE%"
:ssh_key_done

:: SSH firewall rule
netsh advfirewall firewall show rule name="OpenSSH-Server" >nul 2>&1
if %errorlevel% neq 0 (
    netsh advfirewall firewall add rule name="OpenSSH-Server" dir=in action=allow protocol=TCP localport=22 >nul 2>&1
    echo   [OK] SSH firewall rule added
)

:: Enable Remote Desktop
echo   Enabling Remote Desktop...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server" /v fDenyTSConnections /t REG_DWORD /d 0 /f >nul 2>&1
powershell -Command "Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name 'fDenyTSConnections' -Value 0" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Remote Desktop enabled
) else (
    echo   [WARN] Could not enable Remote Desktop
)

:: RDP firewall rule
netsh advfirewall firewall show rule name="Remote Desktop" >nul 2>&1
if %errorlevel% neq 0 (
    netsh advfirewall firewall add rule name="Remote Desktop" dir=in action=allow protocol=TCP localport=3389 >nul 2>&1
    echo   [OK] RDP firewall rule added ^(port 3389^)
) else (
    echo   [OK] RDP firewall rule already exists
)

:remote_done

:: ----------------------------------------------------------------------------
:: Done
:: ----------------------------------------------------------------------------
echo.
echo   ============================================
echo     OpenEar setup complete
echo   ============================================
echo.
if "!MODE!"=="translation" (
    echo   This PC is set up for: captions plus translation
    echo   Choose languages on the admin page.
) else (
    echo   This PC is set up for: captions only
)
echo   To change that, run setup.bat again.
echo.
echo   If OpenEar is already running, restart it so the change takes
echo   effect:  openear.bat restart
echo.
echo   To start OpenEar:
echo     openear.bat start
echo.
echo   To start with visible logs:
echo     openear.bat verbose
echo.
echo   Admin page:  http://localhost/admin
echo   Client page: http://localhost
echo.
echo   NOTE: If this is a fresh Windows install, you may need to
echo   restart your computer for long path support to take effect.
echo.
pause
