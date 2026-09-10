@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo   Windows HPC Virtual Environment Setup for Molecular RL
echo   Target: NVIDIA RTX 3090 (24 GB VRAM) ^| 64 GB RAM ^| Windows OS
echo ===============================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in system PATH.
    echo Please install Python 3.10 or 3.11 from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VER=%%i
echo [1/6] Found %PYTHON_VER%

:: 2. Create isolated virtual environment
if not exist ".venv" (
    echo [2/6] Creating isolated virtual environment in .venv ...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Virtual environment created at: %CD%\.venv
) else (
    echo [2/6] Existing .venv detected. Using %CD%\.venv
)

:: 3. Activate virtual environment
echo [3/6] Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 4. Upgrade Pip
echo [4/6] Upgrading pip tooling...
python -m pip install --upgrade pip setuptools wheel

:: 5. Install PyTorch with CUDA 12.1 for RTX 3090
echo [5/6] Installing PyTorch with CUDA 12.1 support...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: 6. Install project requirements (including NumPy <2 fix and openbabel-wheel)
echo [6/6] Installing dependencies from requirements_hpc_windows.txt...
python -m pip install -r requirements_hpc_windows.txt

:: 7. Create bin and temp folders
if not exist "bin" mkdir "bin"
if not exist "data\temp" mkdir "data\temp"
if not exist "checkpoints\hpc_run" mkdir "checkpoints\hpc_run"
if not exist "logs\hpc_run" mkdir "logs\hpc_run"

echo.
echo ===============================================================================
echo   VERIFICATION ^& SYSTEM CHECKS
echo ===============================================================================
python -c "import torch; print('CUDA Available   :', torch.cuda.is_available()); print('Active GPU       :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Fallback'); print('Torch Version    :', torch.__version__)"
python -c "import numpy; print('NumPy Version    :', numpy.__version__, '(Must be < 2.0.0)')"
python -c "import rdkit; print('RDKit Version    :', rdkit.__version__)"
python -c "from models.reward_oracle import resolve_obabel_path, resolve_vina_path; print('OpenBabel Path   :', resolve_obabel_path()); print('AutoDock Vina    :', resolve_vina_path())"

echo.
echo ===============================================================================
echo   SETUP FINISHED! Next steps:
echo ===============================================================================
echo   1. Ensure vina.exe is placed in the bin\ folder:
echo      %CD%\bin\vina.exe
echo.
echo   2. (Optional) If using AutoDock-GPU, place autodock_gpu_64wi.exe in:
echo      %CD%\bin\autodock_gpu_64wi.exe
echo.
echo   3. Launch training using:
echo      run_hpc_training.bat
echo ===============================================================================
echo.
pause
