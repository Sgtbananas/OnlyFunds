@echo off
REM Ask the forward paper session and its supervisor to stop gracefully (all records are preserved).
REM Usage: STOP_FORWARD_PAPER.bat [deployment]       (default deployment: binance-1h)
cd /d "%~dp0"
set "DEP=%~1"
if "%DEP%"=="" set "DEP=binance-1h"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\forward_task.ps1" -Action stop -Deployment %DEP%
