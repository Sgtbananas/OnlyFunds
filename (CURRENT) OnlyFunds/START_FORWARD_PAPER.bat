@echo off
REM OnlyFunds 2 - start the long-running forward PAPER session (live Binance public data, no orders, no keys).
REM Usage: START_FORWARD_PAPER.bat [deployment]      (default deployment: binance-1h)
REM The supervisor restarts the session after a crash. To stop gracefully run STOP_FORWARD_PAPER.bat [deployment]
REM Respects the OF2_STATE_DIR environment variable (default: .\state).
cd /d "%~dp0"
set "DEP=%~1"
if "%DEP%"=="" set "DEP=binance-1h"
start "OF2 forward paper %DEP%" /min python scripts\forward_supervisor.py --deployment %DEP% --clear-stop -- --interval 1h --every 60 --websocket
echo Started supervisor for deployment %DEP% in a minimised window.
