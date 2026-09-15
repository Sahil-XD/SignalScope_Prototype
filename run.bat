@echo off
setlocal enabledelayedexpansion
title SignalScope Forensic Engine Launcher
echo ===================================================
echo   SignalScope - AI Media Forensics and Verification
echo   SIH 2026 [Internal Hackathon] Team Syndicate
echo ===================================================
echo.

:: Ensure working directory is always the project root
cd /d "%~dp0"

:: Prevent Intel OpenMP DLL initialization aborts and encoding crashes
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

set "PYTHON_EXE="

:: -----------------------------------------------------------------------------
:: GATHER ALL CANDIDATE PYTHON EXECUTABLES
:: -----------------------------------------------------------------------------
set CANDIDATES=
if defined VIRTUAL_ENV set CANDIDATES=!CANDIDATES! "%VIRTUAL_ENV%\Scripts\python.exe"
if defined CONDA_PREFIX set CANDIDATES=!CANDIDATES! "%CONDA_PREFIX%\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set CANDIDATES=!CANDIDATES! "%~dp0.venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\python.exe" set CANDIDATES=!CANDIDATES! "%~dp0venv\Scripts\python.exe"
if exist "%~dp0env\Scripts\python.exe" set CANDIDATES=!CANDIDATES! "%~dp0env\Scripts\python.exe"
if exist "%USERPROFILE%\anaconda3\python.exe" set CANDIDATES=!CANDIDATES! "%USERPROFILE%\anaconda3\python.exe"
if exist "C:\ProgramData\anaconda3\python.exe" set CANDIDATES=!CANDIDATES! "C:\ProgramData\anaconda3\python.exe"
if exist "%USERPROFILE%\miniconda3\python.exe" set CANDIDATES=!CANDIDATES! "%USERPROFILE%\miniconda3\python.exe"
if exist "C:\ProgramData\miniconda3\python.exe" set CANDIDATES=!CANDIDATES! "C:\ProgramData\miniconda3\python.exe"

for %%v in (314 313 312 311 310 39) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe" set CANDIDATES=!CANDIDATES! "%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe"
    if exist "C:\Program Files\Python%%v\python.exe" set CANDIDATES=!CANDIDATES! "C:\Program Files\Python%%v\python.exe"
    if exist "C:\Python%%v\python.exe" set CANDIDATES=!CANDIDATES! "C:\Python%%v\python.exe"
)

for /f "delims=" %%i in ('where.exe python 2^>nul') do (
    set CANDIDATES=!CANDIDATES! "%%i"
)

:: -----------------------------------------------------------------------------
:: PASS 1: Check if any candidate ALREADY has torch and fastapi installed
:: -----------------------------------------------------------------------------
for %%c in (!CANDIDATES!) do (
    if "%PYTHON_EXE%"=="" if exist %%c (
        %%c -c "import fastapi, torch, cv2" >nul 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=%%~c"
            goto :FoundPython
        )
    )
)

:: -----------------------------------------------------------------------------
:: PASS 2: Check for any working Python interpreter
:: -----------------------------------------------------------------------------
for %%c in (!CANDIDATES!) do (
    if "%PYTHON_EXE%"=="" if exist %%c (
        %%c -c "import sys; sys.exit(0)" >nul 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=%%~c"
            goto :FoundPython
        )
    )
)

:: PASS 3: Fallback check with py launcher
py -3 -c "import sys; sys.exit(0)" >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%p in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%p"
        goto :FoundPython
    )
)

:FoundPython
if "%PYTHON_EXE%"=="" (
    echo [ERROR] No working Python installation found on this machine!
    echo.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo IMPORTANT: Remember to check the box "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [*] Using Python: "!PYTHON_EXE!"
"!PYTHON_EXE!" --version

echo.
echo [*] Checking dependencies...
"!PYTHON_EXE!" -c "import fastapi, uvicorn, multipart, torch, torchvision, PIL, numpy, cv2" >nul 2>nul
if !errorlevel! neq 0 (
    echo [*] Some required packages are missing. Installing from requirements.txt...
    echo [*] Running pip install - this may take a few moments...
    "!PYTHON_EXE!" -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo.
        echo [*] Standard pip encountered an issue. Attempting CPU-optimized PyTorch fallback...
        "!PYTHON_EXE!" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        "!PYTHON_EXE!" -m pip install -r requirements.txt
    )
) else (
    echo [OK] All core dependencies verified.
)

:: Check if port 8000 is occupied
set "PORT=8000"
netstat -ano | findstr /r ":8000 .*LISTENING" >nul 2>nul
if !errorlevel! equ 0 (
    echo [*] Notice: Port 8000 is already in use by another process.
    echo [*] Switching to backup port 8001...
    set "PORT=8001"
)

echo.
echo [*] Launching SignalScope Forensic Engine on http://127.0.0.1:!PORT!
echo [*] Browser will open automatically in 3 seconds...
echo.

:: Open browser after 3 second delay to give Uvicorn time to bind
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:!PORT!"

:: Start Uvicorn ASGI server
"!PYTHON_EXE!" -m uvicorn app.main:app --host 127.0.0.1 --port !PORT!

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Server stopped with an error code.
    echo Check the messages above for details.
)
pause
