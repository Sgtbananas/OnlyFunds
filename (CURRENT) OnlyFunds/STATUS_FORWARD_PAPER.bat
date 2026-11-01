@echo off
REM Show whether the forward paper deployment is running (task, supervisor, session, heartbeat, last log lines).
REM Usage: STATUS_FORWARD_PAPER.bat [deployment]     (default deployment: binance-1h)
cd /d "%~dp0"
set "DEP=%~1"
if "%DEP%"=="" set "DEP=binance-1h"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\forward_task.ps1" -Action status -Deployment %DEP%
if "%~2"=="" pause
