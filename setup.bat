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
:: ============================================================================

echo.
echo   ============================================
echo     OpenEar Setup
echo   ============================================
echo.

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
:: Check NVIDIA drivers are installed (required for GPU stability)
:: ----------------------------------------------------------------------------
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   WARNING: NVIDIA drivers not detected!
    echo.
    echo   If this PC has an NVIDIA GPU, you MUST install the latest
    echo   drivers before running OpenEar. Without them, the system
    echo   will be very unstable.
    echo.
    echo   Download drivers from: https://www.nvidia.com/drivers
    echo.
    echo   Install drivers, restart your computer, then run this
    echo   setup script again.
    echo.
    set /p CONTINUE="   Continue anyway without GPU support? (y/N): "
    if /i not "!CONTINUE!"=="y" (
        echo   Setup cancelled. Install NVIDIA drivers first.
        pause
        exit /b 1
    )
    echo   [WARN] Continuing without NVIDIA GPU drivers
    goto gpu_done
)

for /f "tokens=2 delims=:" %%g in ('nvidia-smi -L 2^>nul ^| findstr /i "GPU"') do set GPUNAME=%%g
echo   [OK] NVIDIA drivers found -!GPUNAME!
:gpu_done

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
:: Enable Windows Long Paths (required for PyTorch/CUDA packages)
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
:: Install Visual C++ Runtime (required by CUDA/CTranslate2)
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
:: Install Python dependencies
:: ----------------------------------------------------------------------------
echo.
echo   Installing Python packages (this may take several minutes)...
echo.

:: Core server dependencies
echo   Installing FastAPI, uvicorn, and websockets...
python -m pip install fastapi uvicorn websockets --quiet
if %errorlevel% neq 0 (
    echo   [FAIL] FastAPI/uvicorn install failed
    pause
    exit /b 1
)
echo   [OK] FastAPI and uvicorn

echo   Installing sounddevice...
python -m pip install sounddevice --quiet
if %errorlevel% neq 0 (
    echo   [FAIL] sounddevice install failed
    pause
    exit /b 1
)
echo   [OK] sounddevice

echo   Installing onnx-asr (Parakeet speech recognition)...
python -m pip install "onnx-asr[gpu,hub]" --quiet
if %errorlevel% neq 0 (
    echo   [FAIL] onnx-asr install failed
    pause
    exit /b 1
)
echo   [OK] onnx-asr

echo   Installing sentencepiece and ctranslate2 (for NLLB translation)...
python -m pip install sentencepiece ctranslate2 --quiet
if %errorlevel% neq 0 (
    echo   [FAIL] sentencepiece/ctranslate2 install failed
    pause
    exit /b 1
)
echo   [OK] sentencepiece and ctranslate2

:: ----------------------------------------------------------------------------
:: CUDA GPU acceleration (for NLLB translation model)
:: ----------------------------------------------------------------------------
echo.
echo   Checking for NVIDIA GPU...

nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo   [INFO] No NVIDIA GPU detected. Translation will run on CPU.
    echo          This works but is slower. A GPU is recommended for translation.
    goto cuda_done
)

echo   [OK] NVIDIA GPU detected
echo   Installing CUDA libraries for translation acceleration...
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 --quiet
if %errorlevel% equ 0 (
    echo   [OK] CUDA libraries installed - GPU translation enabled
) else (
    echo   [WARN] CUDA install failed. Translation will run on CPU - slower but functional.
)
:cuda_done

:: ----------------------------------------------------------------------------
:: Pre-download AI models (Parakeet ASR + NLLB translation)
:: ----------------------------------------------------------------------------
echo.
echo   Downloading AI models (~5GB total, one-time download)...
echo   This will take several minutes depending on your internet speed.
echo.

python "%SCRIPT_DIR%download_models.py" 2>&1
if %errorlevel% equ 0 (
    echo   [OK] All models downloaded
) else (
    echo   [WARN] Model download failed. Models will download on first server start.
)

:: ----------------------------------------------------------------------------
:: Remote management tools (optional, --remote flag)
:: ----------------------------------------------------------------------------
if /i not "%1"=="--remote" goto remote_done

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
    winget install --id Git.Git -e --silent >nul 2>&1
    if %errorlevel% equ 0 (
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
    winget install --id tailscale.tailscale -e --silent >nul 2>&1
    if %errorlevel% equ 0 (
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
    if %errorlevel% equ 0 (
        echo   [OK] OpenSSH Server installed
    ) else (
        echo   [WARN] OpenSSH Server install failed.
    )
)

:: Start and auto-start SSH
powershell -Command "Start-Service sshd; Set-Service -Name sshd -StartupType Automatic" >nul 2>&1
echo   [OK] SSH service started and set to auto-start

:: TODO: Set PowerShell as the default SSH shell (currently defaults to cmd.exe)
::   powershell -Command "New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell -Value 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -PropertyType String -Force"

:: TODO: Install Jonathan's SSH public key for passwordless login
::   Key goes in: C:\Users\<username>\.ssh\authorized_keys
::   For admin users, Windows SSH uses: C:\ProgramData\ssh\administrators_authorized_keys
::   (that file needs restricted permissions — only SYSTEM and Administrators, not Users)

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
    echo   [OK] RDP firewall rule added (port 3389)
) else (
    echo   [OK] RDP firewall rule already exists
)

:remote_done

:: ----------------------------------------------------------------------------
:: Done
:: ----------------------------------------------------------------------------
echo.
echo   ============================================
echo     OpenEar setup complete!
echo   ============================================
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
