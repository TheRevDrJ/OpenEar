@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: OpenEar Server Manager
:: Usage: openear.bat [command]
:: ============================================================================

set "SCRIPT_DIR=%~dp0"
set "SERVER_SCRIPT=%SCRIPT_DIR%server.py"
set "PID_FILE=%SCRIPT_DIR%openear.pid"
set "LOG_FILE=%SCRIPT_DIR%openear.log"

if "%~1"=="" goto help
if /i "%~1"=="start" goto start
if /i "%~1"=="stop" goto stop
if /i "%~1"=="restart" goto restart
if /i "%~1"=="status" goto status
if /i "%~1"=="verbose" goto verbose
if /i "%~1"=="log" goto log
if /i "%~1"=="devices" goto devices
if /i "%~1"=="version" goto version
if /i "%~1"=="help" goto help
if /i "%~1"=="--help" goto help
if /i "%~1"=="-h" goto help
goto help

:: ============================================================================
:start
::   Launches the server headless (no console window). Logs go to openear.log.
:: ============================================================================
call :find_pid
if defined RUNNING_PID (
    echo OpenEar is already running ^(PID: !RUNNING_PID!^).
    echo Use 'openear restart' to restart it.
    exit /b 0
)

echo Starting OpenEar...
start "" /b pythonw "%SERVER_SCRIPT%" > nul 2>&1

:: Give the model a moment to load and bind the port
ping 127.0.0.1 -n 6 > nul

call :find_pid
if defined RUNNING_PID (
    echo !RUNNING_PID! > "%PID_FILE%"
    echo.
    echo   OpenEar is running ^(PID: !RUNNING_PID!^)
    echo.
    echo   Admin:  http://localhost/admin.html
    echo   Client: http://localhost
    echo   Log:    %LOG_FILE%
    echo.
    echo   Use 'openear stop' to shut down.
    echo   Use 'openear log' to view live logs.
    echo.
) else (
    echo.
    echo   Failed to start OpenEar.
    echo   Try 'openear verbose' to see errors in the console.
    echo   Or check %LOG_FILE%
    echo.
    exit /b 1
)
exit /b 0

:: ============================================================================
:stop
::   Finds and kills the server process by port.
:: ============================================================================
call :find_pid
if not defined RUNNING_PID (
    echo OpenEar is not running.
    if exist "%PID_FILE%" del "%PID_FILE%"
    exit /b 0
)

echo Stopping OpenEar ^(PID: !RUNNING_PID!^)...
taskkill /F /PID !RUNNING_PID! > nul 2>&1
ping 127.0.0.1 -n 3 > nul

:: Verify it's actually dead
call :find_pid
if defined RUNNING_PID (
    echo Process still alive. Retrying...
    taskkill /F /PID !RUNNING_PID! > nul 2>&1
    ping 127.0.0.1 -n 3 > nul
)

if exist "%PID_FILE%" del "%PID_FILE%"
echo OpenEar stopped.
exit /b 0

:: ============================================================================
:restart
::   Stop then start.
:: ============================================================================
call :stop
echo.
call :start
exit /b 0

:: ============================================================================
:status
::   Shows whether the server is running and on what PID.
:: ============================================================================
call :find_pid
if defined RUNNING_PID (
    echo.
    echo   OpenEar is RUNNING ^(PID: !RUNNING_PID!^)
    echo.
    echo   Admin:  http://localhost/admin.html
    echo   Client: http://localhost
    echo   Log:    %LOG_FILE%
    echo.
) else (
    echo.
    echo   OpenEar is NOT RUNNING.
    echo   Use 'openear start' to launch it.
    echo.
)
exit /b 0

:: ============================================================================
:verbose
::   Starts the server in the foreground with logs visible in the console.
::   Ctrl+C to stop. Useful for troubleshooting.
:: ============================================================================
call :find_pid
if defined RUNNING_PID (
    echo OpenEar is already running headless ^(PID: !RUNNING_PID!^).
    echo Stop it first with 'openear stop' before starting in verbose mode.
    exit /b 1
)

echo.
echo   Starting OpenEar in verbose mode...
echo   Logs will appear below. Press Ctrl+C to stop.
echo   ================================================
echo.
python "%SERVER_SCRIPT%"
exit /b 0

:: ============================================================================
:log
::   Shows the last 40 lines of the log file, then follows new output.
::   Ctrl+C to stop watching.
:: ============================================================================
if not exist "%LOG_FILE%" (
    echo No log file found at %LOG_FILE%.
    echo Start the server first with 'openear start'.
    exit /b 1
)

echo.
echo   Showing log: %LOG_FILE%
echo   Press Ctrl+C to stop watching.
echo   ================================================
echo.

:: Show recent lines then follow
powershell -Command "Get-Content '%LOG_FILE%' -Tail 40 -Wait"
exit /b 0

:: ============================================================================
:devices
::   Lists available audio input devices (no server needed).
:: ============================================================================
echo.
echo   Available audio input devices:
echo   ==============================
echo.
python -c "import sounddevice as sd; devs = sd.query_devices(); [print(f'  [{i}] {d[\"name\"]}  ({sd.query_hostapis(d[\"hostapi\"])[\"name\"]}, {d[\"max_input_channels\"]}ch)') for i, d in enumerate(devs) if d['max_input_channels'] > 0]"
echo.
exit /b 0

:: ============================================================================
:version
::   Shows the OpenEar version.
:: ============================================================================
python -c "f=open(r'%SERVER_SCRIPT%'); [print(f'OpenEar v{l.split(chr(34))[1]}') or exit() for l in f if l.startswith('VERSION')]"
exit /b 0

:: ============================================================================
:find_pid
::   Finds the PID of whatever is listening on port 80.
::   Sets RUNNING_PID if found, clears it if not.
:: ============================================================================
set "RUNNING_PID="
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr "LISTENING" ^| findstr ":80 "') do (
    set "RUNNING_PID=%%a"
)
exit /b 0

:: ============================================================================
:help
:: ============================================================================
echo.
echo   OpenEar Server Manager
echo   ======================
echo.
echo   Usage: openear [command]
echo.
echo   Commands:
echo     start      Start the server in the background ^(headless^)
echo     stop       Stop the server
echo     restart    Stop and restart the server
echo     status     Check if the server is running
echo     verbose    Start with live logs in the console ^(Ctrl+C to stop^)
echo     log        Follow the log file in real time
echo     devices    List available audio input devices
echo     version    Show the OpenEar version
echo     help       Show this help message
echo.
echo   Examples:
echo     openear start       Launch headless, ready for clients
echo     openear verbose     Launch with visible output for troubleshooting
echo     openear devices     See which audio inputs are available
echo     openear log         Watch the log while running headless
echo.
exit /b 0
