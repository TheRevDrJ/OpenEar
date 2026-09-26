@echo off
:: OpenEar - Real-time AI captioning and translation for churches
:: Copyright (c) 2026 TheRevDrJ
:: Licensed under AGPL-3.0 - see LICENSE file for details
setlocal enabledelayedexpansion

:: ============================================================================
:: OpenEar Server Manager
:: Usage: openear.bat [command]
:: ============================================================================

set "SCRIPT_DIR=%~dp0"
set "SERVER_SCRIPT=%SCRIPT_DIR%server.py"
set "PID_FILE=%SCRIPT_DIR%openear.pid"
set "LOG_FILE=%SCRIPT_DIR%openear.log"
set "VENV_PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"
set "VENV_PYTHONW=%SCRIPT_DIR%venv\Scripts\pythonw.exe"

:: ----------------------------------------------------------------------------
:: The command, then any flags. Flags are collected into PASS_ARGS and handed to
:: server.py by start, restart and verbose.
::
:: Only KNOWN flags are accepted. The server ignores anything it does not
:: recognise, so a mistyped flag used to vanish silently and the server started
:: in whatever mode nobody asked for. Here it stops with a message instead.
::
:: Collected once, up front, because restart reaches :start through `call`, and a
:: called label sees the call's arguments, not the command line's: restart used
:: to drop every flag it was given.
::
:: Each flag is passed on in its CANONICAL spelling, not as typed. The match here
:: ignores case but the server's does not, so forwarding "--Captions-Only" as typed
:: got it past this check and then ignored downstream - the exact failure above.
:: ----------------------------------------------------------------------------
set "CMD=%~1"
set "PASS_ARGS="
set "MODE_CHECKED="
:collect_args
shift
if "%~1"=="" goto args_done
if /i "%~1"=="--log-text" goto arg_log_text
if /i "%~1"=="--captions-only" goto arg_captions
if /i "%~1"=="--translation" goto arg_translation
echo.
echo   Unknown option: %~1
echo   Run 'openear help' to see the options.
echo.
exit /b 2
:arg_log_text
set "PASS_ARGS=!PASS_ARGS! --log-text"
goto collect_args
:arg_captions
set "PASS_ARGS=!PASS_ARGS! --captions-only"
goto collect_args
:arg_translation
set "PASS_ARGS=!PASS_ARGS! --translation"
goto collect_args
:args_done

if "%CMD%"=="" goto help
if /i "%CMD%"=="start" goto start
if /i "%CMD%"=="stop" goto stop
if /i "%CMD%"=="restart" goto restart
if /i "%CMD%"=="status" goto status
if /i "%CMD%"=="verbose" goto verbose
if /i "%CMD%"=="log" goto log
if /i "%CMD%"=="devices" goto devices
if /i "%CMD%"=="version" goto version
if /i "%CMD%"=="help" goto help
if /i "%CMD%"=="--help" goto help
if /i "%CMD%"=="-h" goto help
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

call :check_mode
if errorlevel 1 exit /b 2

:: Clean up any stale OpenEar (server.py) pythonw before starting
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%kill_openear.ps1" -Quiet > nul 2>&1
ping 127.0.0.1 -n 2 > nul

echo Starting OpenEar...
start "" /b "%VENV_PYTHONW%" "%SERVER_SCRIPT%" !PASS_ARGS! > nul 2>&1

:: Poll for port 80 - check every 3 seconds, timeout after 180 seconds
set /a ELAPSED=0
echo   Loading models...
:start_wait
ping 127.0.0.1 -n 4 > nul
set /a ELAPSED+=3
call :find_pid
if defined RUNNING_PID goto start_success
if !ELAPSED! geq 180 goto start_timeout
if !ELAPSED! equ 60 echo   Still loading... ^(!ELAPSED!s^) - may be downloading models on first run
if !ELAPSED! neq 60 echo   Still loading... ^(!ELAPSED!s^)
goto start_wait

:start_success
echo !RUNNING_PID! > "%PID_FILE%"
echo.
echo   OpenEar is running ^(PID: !RUNNING_PID!^) - started in !ELAPSED!s
echo.
echo   Admin:  http://localhost/admin
echo   Client: http://localhost
echo   Log:    %LOG_FILE%
echo.
echo   Use 'openear stop' to shut down.
echo   Use 'openear log' to view live logs.
echo.
exit /b 0

:start_timeout
echo.
echo   Failed to start OpenEar after 3 minutes.
echo   Try 'openear verbose' to see errors in the console.
echo   Or check %LOG_FILE%
echo.
exit /b 1

:: ============================================================================
:stop
::   Kills any process on port 80, plus any pythonw running server.py.
::   This catches both the active server AND any stale processes that
::   failed to bind but are still holding GPU memory.
:: ============================================================================
set "FOUND_SOMETHING=0"

:: Kill whatever is on port 80
call :find_pid
if defined RUNNING_PID (
    echo Stopping OpenEar on port 80 ^(PID: !RUNNING_PID!^)...
    taskkill /F /PID !RUNNING_PID! > nul 2>&1
    set "FOUND_SOMETHING=1"
)

:: Kill any stale OpenEar (server.py) pythonw - the helper kills only
:: server.py processes and exits with the count, so we know if anything ran.
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%kill_openear.ps1"
if errorlevel 1 set "FOUND_SOMETHING=1"

if "!FOUND_SOMETHING!"=="0" (
    echo OpenEar is not running.
)

ping 127.0.0.1 -n 3 > nul
if exist "%PID_FILE%" del "%PID_FILE%"
if "!FOUND_SOMETHING!"=="1" echo OpenEar stopped.
exit /b 0

:: ============================================================================
:restart
::   Stop then start. The mode is checked FIRST: restart used to stop a running
::   server and only then find its flags refused, leaving nothing running and
::   reporting success. And :start's exit code is passed on - a failed start
::   used to come back as 0.
:: ============================================================================
call :check_mode
if errorlevel 1 exit /b 2
call :stop
echo.
call :start
exit /b !errorlevel!

:: ============================================================================
:status
::   Shows whether the server is running and on what PID.
:: ============================================================================
call :find_pid
if defined RUNNING_PID (
    echo.
    echo   OpenEar is RUNNING ^(PID: !RUNNING_PID!^)
    echo.
    echo   Admin:  http://localhost/admin
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
call :check_mode
if errorlevel 1 exit /b 2
"%VENV_PYTHON%" "%SERVER_SCRIPT%" !PASS_ARGS!
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
"%VENV_PYTHON%" -c "import sounddevice as sd; devs = sd.query_devices(); [print(f'  [{i}] {d[\"name\"]}  ({sd.query_hostapis(d[\"hostapi\"])[\"name\"]}, {d[\"max_input_channels\"]}ch)') for i, d in enumerate(devs) if d['max_input_channels'] > 0]"
echo.
exit /b 0

:: ============================================================================
:version
::   Shows the OpenEar version.
:: ============================================================================
"%VENV_PYTHON%" -c "f=open(r'%SERVER_SCRIPT%'); [print(f'OpenEar v{l.split(chr(34))[1]}') or exit() for l in f if l.startswith('VERSION')]"
exit /b 0

:: ============================================================================
:check_mode
::   Prints the mode a start with these flags will use, by the same rules the
::   server applies (openear_config.py), and fails on --captions-only together
::   with --translation. Checked once per command, before anything is stopped.
:: ============================================================================
if "!MODE_CHECKED!"=="1" exit /b 0
"%VENV_PYTHON%" "%SCRIPT_DIR%openear_config.py" !PASS_ARGS!
if errorlevel 1 exit /b 2
set "MODE_CHECKED=1"
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
echo   Usage: openear [command] [flags]
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
echo   Flags ^(after start, restart or verbose^):
echo     --log-text        Log transcription and translation text to text-logs/
echo     --captions-only   This run only: captions, no translation model loaded
echo     --translation     This run only: captions plus translation
echo.
echo   This machine's normal mode - captions only, or captions plus
echo   translation - is chosen by setup.bat. Run setup.bat again to change it.
echo.
echo   Examples:
echo     openear start       Launch headless, ready for clients
echo     openear verbose     Launch with visible output for troubleshooting
echo     openear devices     See which audio inputs are available
echo     openear log         Watch the log while running headless
echo.
exit /b 0
