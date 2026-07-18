@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Sidon WebUI - Installer

echo.
echo ============================================================
echo   Sidon Speech Enhancement WebUI - CUDA installer
echo ============================================================
echo.

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No NVIDIA driver was found.
    echo Install the latest NVIDIA driver, restart Windows, then try again.
    pause
    exit /b 1
)

set "PYTHON_CMD="
py -3.11 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD (
    py -3.10 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.10"
)
if not defined PYTHON_CMD (
    python -c "import sys; assert (3,10) <= sys.version_info < (3,13)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo [ERROR] Python 3.10 or 3.11 was not found.
    echo Install 64-bit Python 3.11 from https://www.python.org/downloads/
    echo Enable "Add Python to PATH" in the installer.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating isolated Python environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :install_error
) else (
    echo [1/4] Reusing existing .venv...
)

echo [2/4] Updating installer tools...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :install_error

echo [3/4] Installing CUDA 12.8 PyTorch...
".venv\Scripts\python.exe" -m pip install --upgrade --no-cache-dir torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :install_error

echo [4/4] Installing the minimal WebUI dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade --no-cache-dir -r requirements.txt
if errorlevel 1 goto :install_error

echo.
echo Verifying CUDA...
".venv\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('CUDA OK:', torch.cuda.get_device_name(0), '| PyTorch', torch.__version__, '| CUDA', torch.version.cuda)"
if errorlevel 1 (
    echo.
    echo [ERROR] Packages installed, but CUDA validation failed.
    echo Update your NVIDIA driver and run install.bat again.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Installation complete. Start Sidon with run.bat
echo ============================================================
echo.
pause
exit /b 0

:install_error
echo.
echo [ERROR] Installation failed. Check the message above.
echo Confirm that your internet connection is working, then retry.
pause
exit /b 1
