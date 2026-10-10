# Running Molecule 1 on Google Colab ("Making it Rain")

This folder contains all clean, verified 3D input files ready for the **"Making it Rain" Protein-Ligand Molecular Dynamics pipeline** on Google Colab.

---

## 📁 Files Included
- **`protein.pdb`** (507 KB): Clean DRD2 receptor structure (all 20 standard amino acid chains, cofactors/waters removed).
- **`ligand.pdb`** (5.1 KB): Docked 3D pose of Molecule 1 (Vina pose -10.19 kcal/mol, binding Asp114) with CONECT connectivity records.
- **`ligand_H.pdb`** (7.7 KB): Molecule 1 with all 45 explicit hydrogens pre-added.
- **`ligand.sdf`** / **`ligand.mol2`**: Alternative formats in case you use different parameterization tools.
- **`drd2_mol1_colab.zip`** (107 KB): An all-in-one ZIP archive containing all the above files for quick upload.

---

## 🚀 Step-by-Step Instructions

### Step 1: Open the "Making it Rain" Colab Notebook
Click the official link below:
👉 **[Open Making it Rain: Protein-Ligand MD on Colab](https://colab.research.google.com/github/pablo-arantes/making-it-rain/blob/main/Protein_ligand.ipynb)**

---

### Step 2: Enable GPU Accelerator
1. In Colab, click **Runtime** in the top menu bar.
2. Select **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (Free on standard Colab).
4. Click **Save**.

---

### Step 3: Put Input Files in Google Drive
1. Go to your **Google Drive** ([drive.google.com](https://drive.google.com)).
2. Create a new folder named:
   ```
   drd2_mol1
   ```
3. Upload `protein.pdb` and `ligand.pdb` into that `drd2_mol1` folder.

---

### Step 4: Run the Notebook Cells in Order

#### Cell 1: Install Conda Colab
- Click the Run button (Play icon).
- Colab will install Conda and restart the kernel session automatically.

#### Cell 2: Install Dependencies
- Run this cell. It installs AmberTools, OpenMM, ProLIF, RDKit, and PyTraj (~3-5 minutes).

#### Cell 3: Mount Google Drive
- Run the cell. Click **Connect to Google Drive** and grant access.

#### Cell 4: Check GPU
- Run this cell. It will print your NVIDIA GPU info (`Tesla T4`).

#### Cell 5: Provide Input Files (Fill the Form)
Set the form parameters as follows:
- **`Protein_PDB_file_name`**: `protein.pdb`
- **`remove_waters`**: `yes`
- **`Ligand_PDB_file_name`**: `ligand.pdb`
- **`Add_ligand_hydrogens`**: `Yes`
- **`Charge`**: `0` *(Molecule 1 is neutral)*
- **`Google_Drive_Path`**: `/content/drive/MyDrive/drd2_mol1/` (or your chosen folder)

Click Run.

---

### Step 5: What to Run After Cell 5 (Complete Cell-by-Cell Guide)

#### 🔴 The MUST-RUN Core Simulation Cells:
1. **Cell 12 (`Parameters to generate the topology`)** — **[REQUIRED]**
   - Keep defaults (`ff14SB`, `TIP3P`, `Size_box: 14`, `GAFF2`).
   - Click Run. Builds the Amber topology and water box.
2. **Cell 17 (`Parameters for MD Equilibration protocol`)** — **[REQUIRED]**
   - Set `Temperature: 310` (or leave default 298).
   - Click Run.
3. **Cell 18 (`Runs an Equilibration MD simulation`)** — **[REQUIRED]**
   - Click Run. OpenMM runs energy minimization + equilibration on the GPU.
4. **Cell 20 (`Parameters for MD Production protocol`)** — **[REQUIRED]**
   - `Stride_Time: 10` (10 ns)
   - `Temperature: 310`
   - Click Run.
5. **Cell 21 (`Runs a Production MD simulation`)** — **[REQUIRED]**
   - Click Run. This is the main simulation (~20–25 mins on GPU).
6. **Cell 22 (`Concatenate and align the trajectory`)** — **[REQUIRED]**
   - Make sure `Google_Drive_Path` matches your folder (`/content/drive/MyDrive/drd2_mol1/`).
   - Click Run. Creates the aligned trajectory file `prot_lig_prod_all.dcd`.

---

#### 🟡 Optional 3D Viewers in Notebook (Can run or skip):
- **Cell 14 (`Show 3D structure`)**: Interactive 3D box preview (optional).
- **Cell 15 (`View LigPlot`)**: 2D interaction network before simulation (optional).
- **Cell 23 (`Load, view and check the trajectory`)**: 3D interactive viewer for the trajectory (optional).

---

#### 🟢 The Thesis Goldmine Analysis Cells (Run for your graphs):
- **Cell 30 (`Compute RMSD of protein's CA atoms`)** — **[MUST RUN FOR THESIS]**
  - Generates the standard **RMSD plot** (`rmsd_ca.png`) showing stability.
- **Cell 34 (`Compute RMSF of protein's CA atoms`)** — **[RECOMMENDED]**
  - Generates the **RMSF plot** (`rmsf_ca.png`) showing residue flexibility.
- **Cell 24 (`Ligand Interaction Network during MD`)** — **[RECOMMENDED]**
  - Uses ProLIF to prove the **ASP114** binding interaction was maintained.
- **Cell 26 (`MM-PBSA Binding Free Energy`)** — **[HIGH VALUE]**
  - Calculates the binding free energy ΔG.
- **Cells 31, 32, 33, 35, 36, 37, 38** — Extra graphs (Radius of Gyration, 2D RMSD, PCA) if you need extra pages in your thesis!

