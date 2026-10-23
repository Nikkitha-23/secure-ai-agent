@echo off
title Secure AI Agent Chatbot Launcher
cd /d "%~dp0"

echo ===================================================
echo   Secure AI Agent - Single Click Chatbot Launcher
echo ===================================================
echo.

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" launcher.py
) else (
    python launcher.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Launcher exited with error code %ERRORLEVEL%.
    pause
)
