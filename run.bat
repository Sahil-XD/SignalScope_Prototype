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

:: Prevent Intel OpenMP DLL initialization aborts
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

set "PYTHON_EXE="

:: 1. Check for active virtualenv or conda environment
if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
        goto :FoundPython
    )
)
if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" (
        set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
        goto :FoundPython
    )
)

:: 2. Check for project-local virtual environment (.venv or venv)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    goto :FoundPython
)
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    goto :FoundPython
)
if exist "%~dp0env\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0env\Scripts\python.exe"
    goto :FoundPython
)

:: 3. Test if 'python' on system PATH is a genuine installation (not Microsoft Store dummy)
python -c "import sys; exit(0)" >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%p in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
        echo "%%p" | findstr /i "WindowsApps" >nul
        if errorlevel 1 (
            set "PYTHON_EXE=%%p"
            goto :FoundPython
        )
    )
)

:: 4. Check for py launcher
py -3 -c "import sys; exit(0)" >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%p in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%p"
        goto :FoundPython
    )
)

:: 5. Check Anaconda / Miniconda in user profile and common locations
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
    goto :FoundPython
)
if exist "C:\ProgramData\anaconda3\python.exe" (
    set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
    goto :FoundPython
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
    goto :FoundPython
)
if exist "C:\ProgramData\miniconda3\python.exe" (
    set "PYTHON_EXE=C:\ProgramData\miniconda3\python.exe"
    goto :FoundPython
)

:: 6. Check Local Programs Python versions (3.14 down to 3.9)
for %%v in (314 313 312 311 310 39) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe"
        goto :FoundPython
    )
    if exist "C:\Program Files\Python%%v\python.exe" (
        set "PYTHON_EXE=C:\Program Files\Python%%v\python.exe"
        goto :FoundPython
    )
    if exist "C:\Python%%v\python.exe" (
        set "PYTHON_EXE=C:\Python%%v\python.exe"
        goto :FoundPython
    )
)

:: 7. Search PATH for any non-WindowsApps python.exe
for /f "delims=" %%i in ('where.exe python 2^>nul') do (
    echo "%%i" | findstr /i "WindowsApps" >nul
    if errorlevel 1 (
        set "PYTHON_EXE=%%i"
        goto :FoundPython
    )
)

:FoundPython
if "%PYTHON_EXE%"=="" (
    echo [ERROR] No working Python installation found!
    echo.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo IMPORTANT: Make sure to check the box "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

echo [*] Python detected: "!PYTHON_EXE!"
"!PYTHON_EXE!" --version

echo.
echo [*] Checking dependencies...
"!PYTHON_EXE!" -c "import fastapi, uvicorn, multipart, torch, torchvision, PIL, numpy, cv2" >nul 2>nul
if %errorlevel% neq 0 (
    echo [*] Some dependencies are missing. Installing from requirements.txt...
    echo [*] Running: "!PYTHON_EXE!" -m pip install -r requirements.txt
    "!PYTHON_EXE!" -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [*] pip install encountered an issue. Attempting CPU-optimized PyTorch fallback...
        "!PYTHON_EXE!" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        "!PYTHON_EXE!" -m pip install -r requirements.txt
    )
) else (
    echo [OK] All core dependencies verified.
)

:: Check if port 8000 is occupied
set "PORT=8000"
netstat -ano | findstr /r ":8000 .*LISTENING" >nul 2>nul
if %errorlevel% equ 0 (
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

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server exited with an error code.
    echo Please check the error message above.
)
pause
