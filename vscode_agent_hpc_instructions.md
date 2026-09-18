# VS Code Agent Instructions: HPC Deployment & Vina-GPU Setup

This document provides explicit instructions for the AI Agent (e.g., VS Code Copilot/Agent) to set up the environment, compile the Vina-GPU engine natively, and execute the `scripts/train_ppo_vina_gpu.py` script on a Windows HPC workstation.

> [!IMPORTANT]
> **To the AI Agent reading this:** You are operating on a Windows-based High-Performance Computing (HPC) workstation. This project uses a local virtual environment to isolate dependencies from other users on this shared lab machine. **Do not install packages globally.**

---

## Phase 1: Compile Vina-GPU Natively on the HPC

To guarantee the compiled kernel perfectly matches the HPC's GPU architecture (e.g., RTX 3060/3090), you must compile Vina-GPU 2.1 natively.

### Step 1: Install Visual Studio & Boost
1. **Visual Studio:** Ensure the HPC has **Visual Studio 2019** or **2022** installed with the **"Desktop development with C++"** workload.
2. **Download Boost:** Ensure **Boost 1.77.0** (or similar 1.7x version) source or precompiled Windows binaries are extracted (e.g., `C:\local\boost_1_77_0`).

### Step 2: Clone the Vina-GPU Repository
Open PowerShell on the HPC and clone the repository outside the main project (e.g., in a temp or tools folder):
```bash
git clone https://github.com/DeltaGroupNJU/Vina-GPU.git
cd Vina-GPU
```

### Step 3: Configure the Visual Studio Project
1. Open the **`Vina-GPU.sln`** solution file in Visual Studio.
2. At the top of the window, change the build configuration dropdowns from `Debug` and `x86` to **`Release`** and **`x64`**.
3. Right-click the **Vina-GPU** project in the Solution Explorer and click **Properties**.

### Step 4: Link Boost to the Project
In the Project Properties window:
1. **Include Directories:** Go to `C/C++` -> `General` -> `Additional Include Directories`. Add the path to the Boost extraction folder (e.g., `C:\local\boost_1_77_0`).
2. **Library Directories:** Go to `Linker` -> `General` -> `Additional Library Directories`. Add the path to the Boost compiled libraries folder (e.g., `C:\local\boost_1_77_0\stage\lib` or `C:\local\boost_1_77_0\lib64-msvc-14.2`).
3. Click **Apply** and **OK**.

### Step 5: Build the Executable
1. Go to the top menu bar: **Build** -> **Build Solution** (or press `Ctrl+Shift+B`).
2. If successful, it will generate **`Vina-GPU.exe`** in the `x64\Release` folder of the cloned repository.

### Step 6: Move Files to the RL Project
1. In the root of the `Molecular_RL_Project` directory, create a folder called `vina_gpu_bin`.
2. Copy the newly compiled **`Vina-GPU.exe`** into `vina_gpu_bin`.
3. Go to the original `Vina-GPU\OpenCL` folder you cloned, and copy **all the `.cl` files** (like `kernel1.cl`, `kernel2.cl`) and `lib/` files into your new `vina_gpu_bin` folder.
4. *(If dynamically linking Boost)*: Copy the necessary `boost_*.dll` files from your Boost folder into `vina_gpu_bin` as well.

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
- **Disk Thrashing / OneDrive Sync Issues:** If high disk I/O or OneDrive lagging occurs, use `--no-save` during debugging, or ensure the `checkpoints/` directory is excluded from OneDrive sync.
- **Encoding Errors (`cp1252`):** The script enforces UTF-8 output via `sys.stdout.reconfigure(encoding='utf-8')`. If Unicode errors occur in child processes, check subprocess execution in `models/reward_oracle.py`.
- **Worker Hangs:** `train_ppo_vina_gpu.py` implements multiprocessing timeouts (120s max per docking). If a worker hangs, the environment resets the episode to prevent deadlocks. Look for `TimeoutExpired` warnings.
