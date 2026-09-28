@echo off
title FinClaw Demo Launcher
cd /d "%~dp0"

echo ===================================================
echo           FinClaw Streamlit Demo Launcher
echo ===================================================
echo [1/2] Checking Python Virtual Environment...
if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Could not find .venv\Scripts\streamlit.exe!
    echo Please ensure you are running this from D:\Documents\finclaw
    pause
    exit /b 1
)

echo [2/2] Launching FinClaw Web Dashboard on http://localhost:8501 ...
echo.
.venv\Scripts\streamlit.exe run finclaw_app\app.py

pause
