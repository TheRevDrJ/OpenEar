@echo off
:: OpenEar — Real-time AI captioning and translation for churches
:: Copyright (c) 2026 TheRevDrJ
:: Licensed under AGPL-3.0 — see LICENSE file for details
setlocal enabledelayedexpansion

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

echo   Installing faster-whisper...
python -m pip install faster-whisper --quiet
if %errorlevel% neq 0 (
    echo   [FAIL] faster-whisper install failed
    pause
    exit /b 1
)
echo   [OK] faster-whisper

echo   Installing argostranslate...
python -m pip install argostranslate --quiet
if %errorlevel% neq 0 (
    echo   [FAIL] argostranslate install failed
    pause
    exit /b 1
)
echo   [OK] argostranslate

echo   Installing SSL certificate support...
python -m pip install pip-system-certs --quiet
if %errorlevel% equ 0 (
    echo   [OK] SSL certificates - uses Windows cert store
) else (
    echo   [WARN] pip-system-certs failed. Language pack downloads may not work.
)

:: ----------------------------------------------------------------------------
:: CUDA GPU acceleration (optional but recommended)
:: ----------------------------------------------------------------------------
echo.
echo   Checking for NVIDIA GPU...

nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo   [INFO] No NVIDIA GPU detected. Whisper will run on CPU.
    echo          This works but is significantly slower. A GPU is recommended.
    goto cuda_done
)

echo   [OK] NVIDIA GPU detected
echo   Installing CUDA libraries for GPU acceleration...
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 --quiet
if %errorlevel% equ 0 (
    echo   [OK] CUDA libraries installed - GPU acceleration enabled
) else (
    echo   [WARN] CUDA install failed. Whisper will run on CPU - slower but functional.
)
:cuda_done

:: ----------------------------------------------------------------------------
:: Pre-download Whisper model
:: ----------------------------------------------------------------------------
echo.
echo   Downloading Whisper large-v3 model (~3GB, one-time download)...
echo   This will take a few minutes depending on your internet speed.
echo.

python "%SCRIPT_DIR%download_model.py" 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Whisper model downloaded
) else (
    echo   [WARN] Model download failed. It will download on first server start.
)

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
