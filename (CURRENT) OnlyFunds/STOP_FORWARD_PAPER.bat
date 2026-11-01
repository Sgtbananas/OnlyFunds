@echo off
REM Ask the forward paper session and its supervisor to stop gracefully (all records are preserved).
REM Usage: STOP_FORWARD_PAPER.bat [deployment]       (default deployment: binance-1h)
cd /d "%~dp0"
set "DEP=%~1"
if "%DEP%"=="" set "DEP=binance-1h"
python scripts\forward_supervisor.py --deployment %DEP% --request-stop
echo Stop requested for %DEP%. The session finishes its current step, writes its end record and exits.
