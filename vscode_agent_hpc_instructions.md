# VS Code Agent Instructions: HPC Deployment & Vina-GPU Setup

This document provides explicit instructions for the AI Agent (e.g., VS Code Copilot/Agent) to set up the environment, compile the Vina-GPU engine natively, and execute the `scripts/train_ppo_vina_gpu.py` script on a Windows HPC workstation.

> [!IMPORTANT]
> **To the AI Agent reading this:** You are operating on a Windows-based High-Performance Computing (HPC) workstation. This project uses a local virtual environment to isolate dependencies from other users on this shared lab machine. **Do not install packages globally.**

---

## Phase 1: Compile Vina-GPU Natively on the HPC

To guarantee the compiled kernel perfectly matches the HPC's GPU architecture (e.g., RTX 3060/3090), you must compile Vina-GPU 2.1 natively.

### Step 1: Install Visual Studio, CUDA Toolkit & Boost

> [!CAUTION]
> **Do NOT use `DeltaGroupNJU/Vina-GPU.git`.** That is the stale original repo with no valid `HEAD` and corrupted kernel assets. Using it produces the fatal `0xC0000409 STATUS_STACK_BUFFER_OVERRUN` crash. Always use the correct `DeltaGroupNJUPT/Vina-GPU-2.1.git` repo.

1. **Visual Studio:** Ensure the HPC has **Visual Studio 2019** or **2022** with the **"Desktop development with C++"** workload installed.
2. **CUDA Toolkit:** Install the CUDA Toolkit version matching your driver (e.g., CUDA 12.x for RTX 3060). Download from:
   👉 **https://developer.nvidia.com/cuda-downloads**
   Verify installation: `nvcc --version`
3. **Boost 1.77.0 (REQUIRED — precompiled binaries for MSVC):**
   Download the precompiled Boost 1.77.0 binaries for MSVC 2019/2022 directly:
   👉 **https://sourceforge.net/projects/boost/files/boost-binaries/1.77.0/boost_1_77_0-msvc-14.2-64.exe/download**
   - Run the installer and extract to `C:\local\boost_1_77_0`
   - Verify the folder `C:\local\boost_1_77_0\lib64-msvc-14.2` exists and contains `.lib` files.
   - If the installer link is unavailable, use the source distribution:
     👉 **https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.zip**
     Extract it, then open a VS 2019 Developer Command Prompt and run:
     ```cmd
     cd C:\local\boost_1_77_0
     bootstrap.bat
     b2 address-model=64 link=static runtime-link=shared threading=multi --with-thread --with-filesystem --with-system
     ```

### Step 2: Clone the CORRECT Vina-GPU 2.1 Repository

> [!IMPORTANT]
> The correct upstream repository is `DeltaGroupNJUPT/Vina-GPU-2.1`. Clone it to a location **outside** your main project folder (e.g., `C:\tools\`).

Open PowerShell on the HPC:
```powershell
mkdir C:\tools
cd C:\tools
git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git
cd Vina-GPU-2.1
git log --oneline -5   # Verify: this must show real commits, NOT "fatal: bad default revision"
```

### Step 3: Configure the Visual Studio Project
1. Open **`C:\tools\Vina-GPU-2.1\Vina-GPU.sln`** in Visual Studio.
2. At the top of the window, change the build configuration dropdowns from `Debug` and `x86` to **`Release`** and **`x64`**.
3. Right-click the **Vina-GPU** project in Solution Explorer → click **Properties**.

### Step 4: Link Boost & CUDA to the Project
In the Project Properties window, configure the following:

1. **Include Directories** (`C/C++` → `General` → `Additional Include Directories`). Add:
   ```
   C:\local\boost_1_77_0
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include
   ```
2. **Library Directories** (`Linker` → `General` → `Additional Library Directories`). Add:
   ```
   C:\local\boost_1_77_0\lib64-msvc-14.2
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64
   ```
3. **Additional Dependencies** (`Linker` → `Input` → `Additional Dependencies`). Add:
   ```
   OpenCL.lib
   ```
4. Click **Apply** and **OK**.

### Step 5: Build the Executable
1. Go to **Build** → **Build Solution** (or press `Ctrl+Shift+B`).
2. If successful, you will see output like:
   ```
   ========== Build: 1 succeeded, 0 failed ==========
   ```
   The output file will be at: `C:\tools\Vina-GPU-2.1\x64\Release\Vina-GPU.exe`

### Step 6: Move ALL Required Files to the Project

> [!IMPORTANT]
> `Vina-GPU.exe` **cannot** run alone. It requires the OpenCL kernel source files in the same directory. Forgetting these causes the `0xC0000409` crash.

```powershell
# Create the bin folder in your project
New-Item -ItemType Directory -Force -Path "c:\Users\devan\OneDrive\Desktop\final project\bin"

# Copy the executable
Copy-Item "C:\tools\Vina-GPU-2.1\x64\Release\Vina-GPU.exe" "c:\Users\devan\OneDrive\Desktop\final project\bin\"

# Copy ALL kernel source files (.cl) - these are MANDATORY
Copy-Item "C:\tools\Vina-GPU-2.1\OpenCL\*.cl" "c:\Users\devan\OneDrive\Desktop\final project\bin\"

# Verify:
Get-ChildItem "c:\Users\devan\OneDrive\Desktop\final project\bin\"
# Expected: Vina-GPU.exe, Kernel1.cl, Kernel2.cl (at minimum)
```

### Step 7: First Run — Kernel Compilation (30-60 seconds, one time only)
On the very first run:
- Vina-GPU reads the `.cl` files and compiles them for your exact GPU.
- It generates `Kernel1_code.bin` and `Kernel2_code.bin` in the `bin/` folder.
- All subsequent runs will load the `.bin` files instantly.

**Test the executable directly before using it in training:**
```powershell
cd "c:\Users\devan\OneDrive\Desktop\final project"
.\bin\Vina-GPU.exe --help
# If it prints usage options (not a crash), the build is valid.
```

---

## Phase 2: Environment Verification & Activation

Before running the training script, activate the local virtual environment and verify dependencies.

1. **Activate the Virtual Environment (PowerShell):**
   ```powershell
   # From the project root directory
   .\.venv\Scripts\Activate.ps1
   ```
   *(If it fails due to execution policies, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

2. **Verify Dependencies:**
   Ensure `numpy<2` and `torch` (CUDA 12.1) are installed. 
   ```powershell
   pip list | findstr "numpy torch"
   ```

3. **Verify Other Binaries:**
   Ensure `obabel` is available (usually provided by the `openbabel-wheel` package in the `.venv\Lib\site-packages\openbabel\bin\obabel.exe` path).

---

## Phase 3: Testing the Setup (Smoke Test)

Before launching a multi-day production run, **always run a lightweight smoke test**. We have added specific CLI flags to `train_ppo_vina_gpu.py` to prevent disk thrashing (saving large checkpoints to OneDrive) and to complete a quick training loop.

Execute the following command to run a fast, non-saving test:

```powershell
python scripts\train_ppo_vina_gpu.py --no-save --n-steps 1024 --batch-size 16 --summary-freq 1
```

**Expected Output of Smoke Test:**
- **First Run Kernel Generation:** On the very first run, Vina-GPU will read the `.cl` files and compile them into `Kernel1_code.bin` and `Kernel2_code.bin`. This takes 30-60 seconds. On subsequent runs, it loads instantly.
- The script should detect the GPU.
- It should spawn multiple `SubprocVecEnv` workers.
- It will create isolated temp directories (`data/temp/worker_<pid>`).
- You should see the RL policy update at least once without crashing.
- It should gracefully exit and clean up the temp directories.

---

## Phase 4: Full Production Run

Once the smoke test passes, launch the full HPC training run. The script is configured to use the `VINA_GPU_CONFIG` defaults when run without the test overrides.

```powershell
python scripts\train_ppo_vina_gpu.py
```

> [!NOTE]
> By default, the production run will:
> - Save 1.6GB checkpoints to the `checkpoints/vina_gpu_run/` directory.
> - Run with `n_steps=4096` and `batch_size=128`.
> - Use up to 20 multiprocessing workers depending on hardware threads.

---

## Phase 5: Monitoring & Troubleshooting

### Monitoring
To monitor the training progress, open a separate terminal, activate the `.venv`, and launch TensorBoard:
```powershell
.\.venv\Scripts\Activate.ps1
python -m tensorboard.main --logdir logs\vina_gpu_run --port 6006
```

### Troubleshooting Edge Cases
- **`0xC0000409` / `STATUS_STACK_BUFFER_OVERRUN` crash:** The compiled `Vina-GPU.exe` binary or its OpenCL kernels are corrupt or mismatched with the GPU driver. This is NOT a Python or docking error. **Fix:** Delete `bin/Vina-GPU.exe` and all `.bin` / `.cl` files. Rebuild from the correct source (`DeltaGroupNJUPT/Vina-GPU-2.1.git`) following Phase 1 above. The Python script will now log a `[CRITICAL]` message and automatically fall back to CPU Vina if this is detected.
- **`0xC0000005` / ACCESS_VIOLATION crash:** Same cause as above — corrupt or mismatched kernels/build.
- **Disk Thrashing / OneDrive Sync Issues:** Use `--no-save` during debugging, or exclude `checkpoints/` from OneDrive sync.
- **Encoding Errors (`cp1252`):** The script enforces UTF-8 output via `sys.stdout.reconfigure(encoding='utf-8')`. If Unicode errors occur in child processes, check subprocess execution in `models/reward_oracle.py`.
- **Worker Hangs:** `train_ppo_vina_gpu.py` implements multiprocessing timeouts (120s max per docking). If a worker hangs, the environment resets the episode to prevent deadlocks. Look for `TimeoutExpired` warnings.
