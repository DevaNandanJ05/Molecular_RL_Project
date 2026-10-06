@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo   Launching Windows HPC MAX Molecular RL Training (PPO + MolGPT + Vina)
echo   Hardware Target: NVIDIA RTX 3090 (24 GB VRAM) ^| 16-32 CPU Cores ^| 64 GB RAM
echo   Mode: GPU Batch Generation (FP16 AMP) + SFT Prior-Agent KL Regularization
echo ===============================================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please run setup_hpc_windows.bat first to initialize your isolated environment.
    pause
    exit /b 1
)

:: Activate isolated virtual environment
call .venv\Scripts\activate.bat

:: Launch training with unbuffered output so logs appear in real-time
echo Starting HPC Max Python training pipeline...
echo Logs will be written to logs\hpc_max_run\ and molecule_log.csv
echo Checkpoints will be saved to checkpoints\hpc_max_run\
echo Press Ctrl+C at any time to pause (emergency checkpoint will be saved).
echo.

python -u scripts\train_ppo_hpc_max.py %*

echo.
echo ===============================================================================
echo   HPC Max Training process completed or stopped.
echo ===============================================================================
pause
