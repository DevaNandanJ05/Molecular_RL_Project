# Windows HPC Setup Guide: Isolated Molecular RL Environment

This guide provides complete instructions for setting up and running your MolGPT + PPO molecular reinforcement learning project on a **Windows-based High-Performance Computing (HPC) workstation** equipped with:
- **NVIDIA GeForce RTX 3090 (24 GB VRAM)**
- **64 GB DDR4 RAM**
- **Windows 10 / 11 / Windows Server OS**

---

## Table of Contents
1. [Protecting Your Project on a Shared College Lab Machine](#1-protecting-your-project-on-a-shared-college-lab-machine)
2. [Quickstart: Automated 1-Click Setup](#2-quickstart-automated-1-click-setup)
3. [Manual Installation Guide (Pip + Virtualenv)](#3-manual-installation-guide-pip--virtualenv)
4. [Alternative Setup (Conda / Mamba)](#4-alternative-setup-conda--mamba)
5. [Chemistry Binaries Setup (OpenBabel, Vina, AutoDock-GPU)](#5-chemistry-binaries-setup-openbabel-vina-autodock-gpu)
   - [OpenBabel (Zero Admin Needed)](#a-openbabel)
   - [AutoDock Vina (Primary Docking Engine)](#b-autodock-vina)
   - [AutoDock-GPU (GPU-Accelerated Docking)](#c-autodock-gpu)
6. [Launching and Managing Multi-Day Unattended Training](#6-launching-and-managing-multi-day-unattended-training)
7. [Monitoring & Visualizing Results](#7-monitoring--visualizing-results)
8. [Resuming Training from Checkpoints](#8-resuming-training-from-checkpoints)
9. [Troubleshooting & FAQs](#9-troubleshooting--faqs)

---

## 1. Protecting Your Project on a Shared College Lab Machine

In college computer labs, multiple students share the same workstation. If you install packages globally in system Python (`C:\Program Files\Python...` or `AppData\Roaming\Python\...`), other students running their projects can easily upgrade or uninstall packages, breaking your environment (e.g. accidentally upgrading NumPy to 2.x and breaking SciPy/SB3).

### Rule 1: Always Use a Project-Local Virtual Environment
By creating a `.venv` directly inside your project folder (`final project\.venv\`), your Python libraries, PyTorch CUDA build, and OpenBabel binaries are completely isolated. No actions taken in the global Python environment or by other students will affect your training run.

### Rule 2: Restrict Windows Folder Permissions (`icacls`)
If other students log in with different Windows student accounts, you can prevent them from viewing, modifying, or deleting your project files using Windows Access Control Lists (ACLs).

Open **PowerShell** and run:
```powershell
# Navigate to your project folder
cd "C:\Users\<YourUsername>\OneDrive\Desktop\final project"

# Remove inheritance and give Full Control ONLY to your user account
icacls . /inheritance:r /grant:r "%USERNAME%:(OI)(CI)F"
```
*(Only your Windows account can now access, edit, or execute files in this directory).*

---

## 2. Quickstart: Automated 1-Click Setup

We have created an automated setup script that configures everything for you.

1. Open your project folder in **File Explorer**:
   `C:\Users\<YourUsername>\OneDrive\Desktop\final project`
2. Double-click **`setup_hpc_windows.bat`** (or right-click -> "Run").
3. The script will automatically:
   - Create `.venv` inside the project folder.
   - Install **PyTorch with CUDA 12.1** for your RTX 3090.
   - Install compatible packages (with the mandatory `numpy<2` fix).
   - Install **`openbabel-wheel`** (which embeds `obabel.exe` without admin rights).
   - Create `bin\`, `data\temp\`, `checkpoints\`, and `logs\` folders.
   - Run system verification and print diagnostic checks.

Once complete, proceed to [Section 5](#5-chemistry-binaries-setup-openbabel-vina-autodock-gpu) to verify `vina.exe`.

---

## 3. Manual Installation Guide (Pip + Virtualenv)

If you prefer to run commands manually in PowerShell or Command Prompt:

### Step 1: Create and Activate Virtual Environment
```powershell
cd "C:\Users\<YourUsername>\OneDrive\Desktop\final project"

# Create virtual environment
python -m venv .venv

# Activate environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# (If PowerShell displays an execution policy error, run first:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
```

### Step 2: Install PyTorch with CUDA 12.1
The RTX 3090 is an Ampere-architecture GPU (Compute Capability 8.6). Installing PyTorch with CUDA 12.1 enables optimal tensor core utilization:
```powershell
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Project Dependencies
```powershell
pip install -r requirements_hpc_windows.txt
```

### List of Pip Packages Installed:
| Package | Version Range | Purpose |
| :--- | :--- | :--- |
| **`numpy`** | `>=1.24.0,<2.0.0` | **Crucial**: Prevents NumPy 2.x C-API incompatibility with SciPy / SB3 |
| **`scipy`** | `>=1.10.0,<1.13.0` | Sparse matrices and optimization |
| **`rdkit`** | `>=2023.3.1` | Chemical informatics, 3D conformers, SMILES parsing |
| **`openbabel-wheel`** | `>=3.1.1.20` | PDBQT generation + embeds isolated `obabel.exe` |
| **`torch`** | `>=2.1.0 (CUDA 12.1)` | Deep learning backbone with full RTX 3090 GPU acceleration |
| **`transformers`** | `>=4.30.0` | MolGPT autoregressive causal LM policy backbone |
| **`stable-baselines3`** | `>=2.1.0` | Proximal Policy Optimization (PPO) reinforcement learning |
| **`gymnasium`** | `>=0.28.1,<0.30.0` | Token-by-token RL environment definition |
| **`tensorboard`** | `>=2.13.0` | Real-time metric logging and loss visualization |

---

## 4. Alternative Setup (Conda / Mamba)

If your lab already has Anaconda or Miniconda installed:

```powershell
# Create conda environment with Python 3.10
conda create -n mol_rl python=3.10 -y
conda activate mol_rl

# Install OpenBabel and RDKit via conda-forge
conda install -c conda-forge openbabel rdkit -y

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install RL and language model dependencies
pip install "numpy<2" "scipy<1.13" transformers stable-baselines3 gymnasium tensorboard pandas tqdm
```

---

## 5. Chemistry Binaries Setup (OpenBabel, Vina, AutoDock-GPU)

To prevent conflicts with other students, all executables can be placed directly in your project's local `bin\` folder:
```
final project\
└── bin\
    ├── vina.exe               <-- AutoDock Vina 1.2.5
    ├── obabel.exe             <-- (Provided automatically by openbabel-wheel)
    └── autodock_gpu_64wi.exe  <-- (Optional AutoDock-GPU)
```
The script will automatically detect them!

---

### A. OpenBabel
- **Automatic Setup (Recommended)**: By installing `openbabel-wheel`, OpenBabel is pre-packaged inside your virtual environment at:
  `.venv\Lib\site-packages\openbabel\bin\obabel.exe`
  The project automatically locates this path. **No manual installation or admin rights required!**
- **Manual Alternative**: If you prefer the standalone installer, install OpenBabel to `C:\Program Files\OpenBabel-2.4.1\obabel.exe` or download the portable zip and copy `obabel.exe` into `final project\bin\`.

---

### B. AutoDock Vina (Primary Docking Engine)
AutoDock Vina performs CPU-based molecular docking directly from the DRD2 receptor `.pdbqt` file and bounding box coordinates.

1. Download the official Windows release from GitHub:
   👉 **[AutoDock Vina 1.2.5 Windows Release (GitHub)](https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_windows_x86_64.zip)**
2. Extract the archive.
3. Copy **`vina.exe`** into your project's `bin\` folder:
   `final project\bin\vina.exe`
4. Verify by running in PowerShell:
   ```powershell
   .\bin\vina.exe --version
   ```

---

### C. AutoDock-GPU (GPU-Accelerated Docking)
AutoDock-GPU uses OpenCL/CUDA to accelerate docking on your RTX 3090.

> [!NOTE]
> **Vina vs. AutoDock-GPU**:
> - **AutoDock Vina** docks directly into raw `.pdbqt` receptors (like `drd2_clean.pdbqt`) using bounding box coordinates. This is the **primary plug-and-play engine** for this project.
> - **AutoDock-GPU** is extraordinarily fast on GPU, but it strictly requires **AutoGrid4 pre-calculated grid maps** (`.maps.fld` and `.map` files).

To set up AutoDock-GPU:
1. Download the precompiled Windows 64-bit binary from Scripps Research:
   👉 **[AutoDock-GPU Releases (GitHub)](https://github.com/ccsb-scripps/AutoDock-GPU/releases)**
   File name: **`autodock_gpu_64wi.exe`**
2. Copy `autodock_gpu_64wi.exe` into `final project\bin\`.
3. Verify in PowerShell:
   ```powershell
   .\bin\autodock_gpu_64wi.exe --help
   ```

---

## 6. Launching and Managing Multi-Day Unattended Training

### Option A: 1-Click Launch (Recommended)
Simply double-click:
`run_hpc_training.bat`

### Option B: PowerShell Command Line
```powershell
# Activate your environment
.\.venv\Scripts\Activate.ps1

# Run training
python scripts\train_ppo_batched_in_hpc.py --n-envs 16 --timesteps 10000000
```

### CLI Command Options:
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--n-envs` | `16` | Number of parallel worker environments. |
| `--timesteps` | `10000000` | Total timesteps for the multi-day run (~200,000 molecules). |
| `--vec-env` | `subproc` | `subproc` (parallel multiprocessing) or `dummy` (single-process debug). |
| `--vina-path` | Auto-detect | Override path to `vina.exe`. |
| `--obabel-path` | Auto-detect | Override path to `obabel.exe`. |
| `--resume` | `None` | Path to a checkpoint `.zip` file to continue training. |

### Running in the Background (Surviving Remote Desktop Disconnect)
When connecting to your college lab machine via Remote Desktop (RDP), closing the RDP window can sometimes suspend interactive console sessions. To ensure training runs continuously:

1. Launch training using PowerShell `Start-Process`:
   ```powershell
   Start-Process -FilePath "cmd.exe" -ArgumentList "/k run_hpc_training.bat"
   ```
2. Or configure a **Windows Task Scheduler** task:
   - Action: Start a program -> `cmd.exe`
   - Arguments: `/c "C:\Users\<YourUser>\OneDrive\Desktop\final project\run_hpc_training.bat"`
   - Security options: "Run whether user is logged on or not".

---

## 7. Monitoring & Visualizing Results

### Real-Time Molecule CSV Log
Open `logs\hpc_run\molecule_log.csv` to inspect every molecule generated:
- Timestep & Episode
- SMILES string
- Chemical Validity (`True` / `False`)
- Total Reward
- QED (Drug-likeness)
- Vina Docking Affinity (kcal/mol)
- Molecular Weight

### Live TensorBoard Dashboard
In a separate terminal, run:
```powershell
.\.venv\Scripts\Activate.ps1
tensorboard --logdir logs\hpc_run --port 6006
```
Open your browser to: **`http://localhost:6006`**
You will see real-time curves for:
- `molecules/validity_rate`: % of generated SMILES that are valid molecules.
- `molecules/best_docking`: Strongest binding affinity discovered so far.
- `molecules/unique_count`: Chemical diversity tracking.
- `train/approx_kl`, `train/loss`, `train/value_loss`: PPO optimization stability.

---

## 8. Resuming Training from Checkpoints

The script automatically saves full model weights every 10,000 steps to:
`checkpoints\hpc_run\ppo_molgpt_hpc_<step>_steps.zip`

If the machine restarts or you stop training with `Ctrl+C`:
```powershell
python scripts\train_ppo_batched_in_hpc.py --resume checkpoints\hpc_run\ppo_molgpt_hpc_50000_steps.zip
```
The script will load the weights, restore the optimizer state, and continue training seamlessly.

---

## 9. Troubleshooting & FAQs

### Q1: "AttributeError: _ARRAY_API not found" or NumPy 2.x crash
**Cause**: Another student or a global installer updated NumPy to version 2.x.
**Fix**: Ensure your `.venv` is active and run:
```powershell
pip install "numpy<2.0.0" "scipy<1.13.0"
```

### Q2: "OpenBabel Binary: NOT FOUND"
**Fix**: Run `pip install openbabel-wheel` inside your active virtual environment. The script will automatically detect `site-packages\openbabel\bin\obabel.exe`.

### Q3: "AutoDock Vina: NOT FOUND"
**Fix**: Download `vina_1.2.5_windows_x86_64.zip` and place `vina.exe` into `final project\bin\vina.exe`.

### Q4: "CUDA out of memory"
**Cause**: MolGPT full unfreezing requires ~18-20 GB VRAM during backward passes.
**Fix**:
1. Check that other GPU tasks are closed using `nvidia-smi`.
2. PyTorch gradient checkpointing is enabled by default in `MolGPTExtractorHPC`, reducing activation memory to ~3 GB.
3. If memory is tight, reduce `batch_size` from 128 to 64 in `HPC_CONFIG`.
