@echo off
setlocal enabledelayedexpansion
title SignalScope Forensic Engine Launcher
echo ===================================================
echo   SignalScope - AI Media Forensics & Verification
echo   SIH 2026 [Internal Hackathon] Team Syndicate
echo ===================================================
echo.

:: Detect real Python executable
set "PYTHON_EXE="

:: 1. Check Anaconda user profile
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
    goto :FoundPython
)

:: 2. Check Anaconda ProgramData
if exist "C:\ProgramData\anaconda3\python.exe" (
    set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
    goto :FoundPython
)

:: 3. Check Miniconda
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
    goto :FoundPython
)

:: 4. Check AppData Local Programs Python versions
for %%v in (313 312 311 310 39) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe"
        goto :FoundPython
    )
)

:: 5. Check py launcher
py -3 -c "import sys; exit(0)" >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_EXE=py -3"
    goto :FoundPython
)

:: 6. Check PATH python (excluding WindowsApps dummy stub)
for /f "delims=" %%i in ('where python 2^>nul') do (
    echo "%%i" | findstr /i "WindowsApps" >nul
    if errorlevel 1 (
        set "PYTHON_EXE=%%i"
        goto :FoundPython
    )
)

:FoundPython
if "%PYTHON_EXE%"=="" (
    echo [ERROR] No working Python installation found!
    echo Please install Python 3.10+ or Anaconda.
    pause
    exit /b 1
)

echo [*] Using Python: %PYTHON_EXE%

echo [*] Checking dependencies...
%PYTHON_EXE% -c "import fastapi, uvicorn, torch, torchvision, PIL, numpy" >nul 2>nul
if %errorlevel% neq 0 (
    echo [*] Installing missing dependencies from requirements.txt...
    %PYTHON_EXE% -m pip install -r requirements.txt
) else (
    echo [OK] All core dependencies verified!
)

echo.
echo [*] Launching SignalScope Forensic Engine on http://127.0.0.1:8000...
echo [*] Opening default web browser...

start http://127.0.0.1:8000

%PYTHON_EXE% -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
