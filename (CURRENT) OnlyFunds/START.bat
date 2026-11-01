@echo off
REM OnlyFunds 2 - start the Streamlit app (paper trading only)
cd /d "%~dp0"
python -m streamlit run main.py
pause
