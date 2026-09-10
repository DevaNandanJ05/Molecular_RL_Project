@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo   Launching Windows HPC Molecular RL Training (PPO + MolGPT + Vina)
echo   Hardware: RTX 3090 (24 GB VRAM) ^| 64 GB RAM
echo ===============================================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please run setup_hpc_windows.bat first to initialize your environment.
    pause
    exit /b 1
)

:: Activate isolated virtual environment
call .venv\Scripts\activate.bat

:: Launch training with unbuffered output so logs appear in real-time
echo Starting Python training pipeline...
echo Logs will be written to logs\hpc_run\ and molecule_log.csv
echo Press Ctrl+C at any time to pause (emergency checkpoint will be saved).
echo.

python -u scripts\train_ppo_batched_in_hpc.py %*

echo.
echo ===============================================================================
echo   Training process completed or stopped.
echo ===============================================================================
pause
