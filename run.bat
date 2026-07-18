@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Sidon Speech Enhancement WebUI

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Sidon is not installed yet.
    echo Run install.bat first.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "GRADIO_ANALYTICS_ENABLED=False"
if not defined SIDON_MODEL_CACHE set "SIDON_MODEL_CACHE=%~dp0model_cache"
if not defined SIDON_INBROWSER set "SIDON_INBROWSER=1"

".venv\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'"
if errorlevel 1 (
    echo [ERROR] CUDA is unavailable in the Sidon environment.
    echo Run install.bat again and check your NVIDIA driver.
    pause
    exit /b 1
)

echo Starting Sidon at http://127.0.0.1:7860
echo The first start downloads about 1 GB, then loads Sidon on CUDA.
echo The browser opens only after the model is ready.
echo Press Ctrl+C to stop the WebUI.
echo.
".venv\Scripts\python.exe" app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Sidon stopped because of an error. Read the message above.
    pause
    exit /b 1
)

exit /b 0
