@echo off
REM OnlyFunds 2 - start the long-running forward PAPER session (live Binance public data, no orders, no keys).
REM Usage: START_FORWARD_PAPER.bat [deployment]      (default deployment: binance-1h)
REM Starts the supervisor through the Windows Task Scheduler (task \OnlyFunds\OF2 Forward Paper <deployment>),
REM so it keeps running when this window, a terminal, an editor or Claude Code is closed.
REM The supervisor restarts the session after a crash. Stop gracefully with STOP_FORWARD_PAPER.bat [deployment];
REM check with STATUS_FORWARD_PAPER.bat [deployment]. Respects OF2_STATE_DIR (default: .\state).
cd /d "%~dp0"
set "DEP=%~1"
if "%DEP%"=="" set "DEP=binance-1h"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\forward_task.ps1" -Action start -Deployment %DEP%
