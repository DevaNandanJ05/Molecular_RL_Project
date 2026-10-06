"""
===============================================================================
AutoDock-GPU Grid Map Generator
===============================================================================
AutoDock-GPU requires pre-computed grid maps (.map) for the receptor,
generated using AutoGrid4. This script automates creating the Grid 
Parameter File (.gpf) and running AutoGrid4.

Usage on Linux HPC:
  sudo apt-get install autodock   # Installs autogrid4
  python scripts/prepare_adgpu_maps.py
===============================================================================
"""

import os
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Config
RECEPTOR_PDBQT = os.path.join(PROJECT_ROOT, "data", "raw", "drd2_clean.pdbqt")
MAPS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "drd2_maps")
AUTOGRID_BIN = "autogrid4" # Available via 'apt install autodock'

# Grid settings (Matching your Vina settings)
CENTER = (9.5, 5.2, -11.4)
SIZE_ANGSTROMS = (20.0, 20.0, 20.0)
GRID_SPACING = 0.375  # Standard AD4 spacing in Angstroms

# Calculate grid points (must be even numbers)
pts_x = int(SIZE_ANGSTROMS[0] / GRID_SPACING)
pts_y = int(SIZE_ANGSTROMS[1] / GRID_SPACING)
pts_z = int(SIZE_ANGSTROMS[2] / GRID_SPACING)

# Ensure even integers
pts_x += pts_x % 2
pts_y += pts_y % 2
pts_z += pts_z % 2

GPF_CONTENT = f"""npts {pts_x} {pts_y} {pts_z} # num.grid points in xyz
gridfld drd2_clean.maps.fld # grid_data_file
spacing {GRID_SPACING} # spacing(A)
receptor_types A C H HD N NA OA P SA S # receptor atom types
ligand_types A C F Cl Br I OA N NA HD H S SA P # standard ligand atom types
receptor {RECEPTOR_PDBQT} # macromolecule
gridcenter {CENTER[0]:.3f} {CENTER[1]:.3f} {CENTER[2]:.3f} # xyz-coordinates or auto
smooth 0.5 # store minimum energy w/in rad(A)
map drd2_clean.A.map # atom-specific affinity map
map drd2_clean.C.map # atom-specific affinity map
map drd2_clean.F.map # atom-specific affinity map
map drd2_clean.Cl.map # atom-specific affinity map
map drd2_clean.Br.map # atom-specific affinity map
map drd2_clean.I.map # atom-specific affinity map
map drd2_clean.OA.map # atom-specific affinity map
map drd2_clean.N.map # atom-specific affinity map
map drd2_clean.NA.map # atom-specific affinity map
map drd2_clean.HD.map # atom-specific affinity map
map drd2_clean.H.map # atom-specific affinity map
map drd2_clean.S.map # atom-specific affinity map
map drd2_clean.SA.map # atom-specific affinity map
map drd2_clean.P.map # atom-specific affinity map
elecmap drd2_clean.e.map # electrostatic potential map
dsolvmap drd2_clean.d.map # desolvation potential map
dielectric -0.1465 # <0, AD4 distance-dep.diel;>0, constant
"""

def main():
    if not os.path.exists(RECEPTOR_PDBQT):
        print(f"[Error] Receptor not found at {RECEPTOR_PDBQT}")
        return

    os.makedirs(MAPS_DIR, exist_ok=True)
    gpf_path = os.path.join(MAPS_DIR, "drd2_clean.gpf")

    with open(gpf_path, "w") as f:
        f.write(GPF_CONTENT)
    print(f"[INFO] Created GPF file at {gpf_path}")

    # Change to MAPS_DIR so autogrid4 outputs files there
    os.chdir(MAPS_DIR)

    print(f"[INFO] Running {AUTOGRID_BIN}...")
    try:
        # Check if autogrid4 is installed
        subprocess.run([AUTOGRID_BIN, "-v"], capture_output=True, check=True)
        
        # Run AutoGrid
        subprocess.run([AUTOGRID_BIN, "-p", "drd2_clean.gpf", "-l", "drd2_clean.glg"], check=True)
        print(f"[SUCCESS] Grid maps generated successfully in {MAPS_DIR}")
        print(f"          The file 'drd2_clean.maps.fld' is ready for AutoDock-GPU.")
    except FileNotFoundError:
        print(f"[ERROR] '{AUTOGRID_BIN}' not found.")
        print(f"        Please install it first: sudo apt-get install autodock")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] AutoGrid4 execution failed. See drd2_clean.glg in {MAPS_DIR} for details.")

if __name__ == "__main__":
    main()
