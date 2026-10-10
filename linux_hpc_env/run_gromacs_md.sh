#!/bin/bash
# ==============================================================================
# AUTOMATED GROMACS MD PIPELINE FOR DRD2 PROTEIN-LIGAND COMPLEX (RTX 3090)
# ==============================================================================
# This script completely automates a 50-nanosecond Molecular Dynamics simulation.
# 
# PREREQUISITES (Run these on your Linux HPC before executing this script):
# 1. Install Conda (if not already installed)
# 2. Create an environment with GROMACS, ACPYPE, and OpenBabel:
#    $ conda create -n md_env -c conda-forge -c bioconda gromacs acpype openbabel python=3.10
#    $ conda activate md_env
#
# REQUIRED FILES (Place these in the same folder as this script):
# 1. mol1_out.pdbqt (The docked ligand from AutoDock Vina)
# 2. drd2_clean.pdb (The DRD2 receptor PDB file)
#
# TO RUN:
# $ bash run_gromacs_md.sh
# ==============================================================================

set -e # Exit immediately if a command fails

echo "========================================================================"
echo "  STARTING AUTOMATED GROMACS PROTEIN-LIGAND PIPELINE (50 NANOSECONDS)   "
echo "========================================================================"

# --- 1. Check Dependencies ---
command -v gmx >/dev/null 2>&1 || { echo >&2 "GROMACS (gmx) is required but not installed. Aborting."; exit 1; }
command -v acpype >/dev/null 2>&1 || { echo >&2 "acpype is required but not installed. Aborting."; exit 1; }
command -v obabel >/dev/null 2>&1 || { echo >&2 "OpenBabel (obabel) is required but not installed. Aborting."; exit 1; }

# --- 2. Generate Parameter Files (.mdp) ---
echo "[1/8] Generating GROMACS parameter files (.mdp)..."

cat << 'EOF' > ions.mdp
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 1
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = cutoff
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
EOF

cat << 'EOF' > minim.mdp
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 1
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
EOF

cat << 'EOF' > nvt.mdp
title       = NVT equilibration 
define      = -DPOSRES
integrator  = md
nsteps      = 50000     ; 100 ps
dt          = 0.002
nstxout     = 500
nstvout     = 500
nstenergy   = 500
nstlog      = 500
continuation = no
constraint_algorithm = lincs
constraints = h-bonds
lincs_iter  = 1
lincs_order = 4
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME
pme_order   = 4
fourierspacing = 0.16
tcoupl      = V-rescale
tc-grps     = Protein_Ligand Water_and_Ions
tau_t       = 0.1     0.1
ref_t       = 310     310  ; Human body temp
pcoupl      = no
pbc         = xyz
DispCorr    = EnerPres
gen_vel     = yes
gen_temp    = 310
gen_seed    = -1
EOF

cat << 'EOF' > npt.mdp
title       = NPT equilibration 
define      = -DPOSRES
integrator  = md
nsteps      = 50000     ; 100 ps
dt          = 0.002
nstxout     = 500
nstvout     = 500
nstenergy   = 500
nstlog      = 500
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
lincs_iter  = 1
lincs_order = 4
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME
pme_order   = 4
fourierspacing = 0.16
tcoupl      = V-rescale
tc-grps     = Protein_Ligand Water_and_Ions
tau_t       = 0.1     0.1
ref_t       = 310     310
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
pbc         = xyz
DispCorr    = EnerPres
gen_vel     = no
EOF

cat << 'EOF' > md.mdp
title       = OPLS MD Simulation 
integrator  = md
nsteps      = 25000000  ; 50 ns (25 million steps * 2fs)
dt          = 0.002
nstxout     = 0
nstvout     = 0
nstfout     = 0
nstenergy   = 5000
nstlog      = 5000
nstxout-compressed  = 5000
compressed-x-grps   = System
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
lincs_iter  = 1
lincs_order = 4
cutoff-scheme = Verlet
ns_type     = grid
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME
pme_order   = 4
fourierspacing = 0.16
tcoupl      = V-rescale
tc-grps     = Protein_Ligand Water_and_Ions
tau_t       = 0.1     0.1
ref_t       = 310     310
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
pbc         = xyz
DispCorr    = EnerPres
gen_vel     = no
EOF

# --- 3. Process Protein and Ligand ---
echo "[2/8] Processing Receptor and Ligand Topologies..."

# Convert PDBQT to PDB and assign unique residue name 'LIG'
obabel -ipdbqt mol1_out.pdbqt -opdb -O ligand.pdb -h
sed -i 's/UNL/LIG/g' ligand.pdb

# Generate Protein Topology (AMBER99SB-ILDN)
# Note: '8' selects amber99sb-ildn, '1' selects TIP3P water
echo -e "8\n1" | gmx pdb2gmx -f drd2_clean.pdb -o protein_processed.gro -water tip3p -ignh

# Generate Ligand Topology using ACPYPE (AMBER GAFF Forcefield with AM1-BCC charges)
acpype -i ligand.pdb -c bcc -n 0 -a amber -o gmx
cp ligand.acpype/ligand_GMX.gro .
cp ligand.acpype/ligand_GMX.itp .

# --- 4. Merge Protein and Ligand ---
echo "[3/8] Assembling the Protein-Ligand Complex..."

# Python script to safely merge GRO files and update the Topology
cat << 'EOF' > merge_complex.py
import sys

def merge_gro():
    with open('protein_processed.gro', 'r') as f:
        prot_lines = f.readlines()
    with open('ligand_GMX.gro', 'r') as f:
        lig_lines = f.readlines()
        
    prot_atoms = int(prot_lines[1].strip())
    lig_atoms = int(lig_lines[1].strip())
    total_atoms = prot_atoms + lig_atoms
    
    with open('complex.gro', 'w') as f:
        f.write("Protein-Ligand Complex\n")
        f.write(f"{total_atoms}\n")
        # Write protein atoms (excluding header, atom count, and box vectors)
        for line in prot_lines[2:-1]:
            f.write(line)
        # Write ligand atoms (excluding header, atom count, and box vectors)
        for line in lig_lines[2:-1]:
            f.write(line)
        # Write box vectors from protein
        f.write(prot_lines[-1])

def update_topology():
    with open('topol.top', 'r') as f:
        top_lines = f.readlines()
        
    with open('topol.top', 'w') as f:
        for line in top_lines:
            f.write(line)
            # Insert ligand .itp inclusion right after the forcefield inclusions
            if 'forcefield.itp"' in line:
                f.write('; Include ligand topology\n')
                f.write('#include "ligand_GMX.itp"\n\n')
                
        # Add ligand to the molecules list at the very end
        f.write("ligand              1\n")

merge_gro()
update_topology()
EOF

python3 merge_complex.py

# Create a custom index file combining Protein and Ligand into one group
cat << 'EOF' > make_ndx.txt
"Protein" | "ligand"
q
EOF
gmx make_ndx -f complex.gro -o index.ndx < make_ndx.txt

# --- 5. Solvation and Ions ---
echo "[4/8] Building Water Box and Neutralizing with 0.15M NaCl..."
gmx editconf -f complex.gro -o newbox.gro -c -d 1.0 -bt cubic
gmx solvate -cp newbox.gro -cs spc216.gro -o solv.gro -p topol.top
gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 1
echo "SOL" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15

# --- 6. Energy Minimization ---
echo "[5/8] Running Energy Minimization..."
gmx grompp -f minim.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 1
gmx mdrun -v -deffnm em

# --- 7. Equilibration (NVT and NPT) ---
echo "[6/8] Running Temperature (NVT) and Pressure (NPT) Equilibration..."
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt.tpr -maxwarn 1
# Assuming GPU is available, offload as much as possible
gmx mdrun -v -deffnm nvt -nb gpu -pme gpu -bonded gpu || gmx mdrun -v -deffnm nvt

gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -n index.ndx -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt -nb gpu -pme gpu -bonded gpu || gmx mdrun -v -deffnm npt

# --- 8. Production MD Simulation ---
echo "[7/8] RUNNING 50-NANOSECOND PRODUCTION MD (GPU ACCELERATED)..."
echo "      (This will take 1-3 hours on an RTX 3090. Grab a coffee!)"
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o md_0_1.tpr -maxwarn 1

# Fire up the RTX 3090
gmx mdrun -v -deffnm md_0_1 -nb gpu -pme gpu -bonded gpu -update gpu

# --- 9. Final Analysis ---
echo "[8/8] Calculating RMSD trajectory for validation graph..."
# Group 1 (Protein) for least squares fit, Group 13 (Ligand) for RMSD calculation
echo -e "1\n13" | gmx rms -s md_0_1.tpr -f md_0_1.xtc -o rmsd_ligand.xvg -tu ns -fit rot+trans

echo "========================================================================"
echo " SIMULATION COMPLETE!"
echo " Check 'rmsd_ligand.xvg' to see if your molecule stayed bound to DRD2."
echo " Use PyMOL or ChimeraX to view 'md_0_1.xtc' and watch the simulation video!"
echo "========================================================================"
