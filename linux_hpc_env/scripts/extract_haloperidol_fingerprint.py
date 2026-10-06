"""
Extract Haloperidol Reference Interaction Fingerprint for IF-ARS
=================================================================
Task 1 of the IF-ARS Pipeline (Phase 3)

This script:
  1. Takes the canonical SMILES of Haloperidol (the control DRD2 antagonist)
  2. Generates a 3D conformer using RDKit (ETKDG + UFF optimization)
  3. Converts it to PDBQT format via OpenBabel
  4. Docks it into the DRD2 receptor (drd2_clean.pdbqt) using AutoDock Vina
     with HIGH exhaustiveness (32) for an accurate reference pose
  5. Merges the receptor + docked ligand into a single PDB complex
  6. Runs PLIP (Protein-Ligand Interaction Profiler) to detect all
     non-covalent interactions at the atom level
  7. Saves the reference interaction fingerprint as JSON

Output: data/reference/haloperidol_fingerprint.json

This JSON file is loaded by the IF-ARS reward shaper during Phase 4
training to compare AI-generated molecules against Haloperidol's
proven binding pattern.

Usage:
  python scripts/extract_haloperidol_fingerprint.py
  python scripts/extract_haloperidol_fingerprint.py --exhaustiveness 64
  python scripts/extract_haloperidol_fingerprint.py --vina-path bin/vina.exe
"""

import os
import sys
import json
import uuid
import subprocess
import argparse
from datetime import datetime

# ----------------------------------------------------------------------
# Project root setup
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.reward_oracle import resolve_vina_path, resolve_obabel_path

# ----------------------------------------------------------------------
# Haloperidol -- The Gold Standard DRD2 Antagonist
# ----------------------------------------------------------------------
# Canonical SMILES from PubChem CID: 3559
# Haloperidol is the prototypical first-generation antipsychotic.
# It binds DRD2 with high affinity (~1 nM Ki) via a conserved
# salt bridge with Asp114(3.32) and hydrophobic contacts with
# Phe389, Phe390, Trp386, and Val115.
HALOPERIDOL_SMILES = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
HALOPERIDOL_NAME = "Haloperidol"

# ----------------------------------------------------------------------
# DRD2 Binding Pocket Parameters (from 6CM4 crystal structure)
# These match the exact coordinates used in reward_oracle.py
# ----------------------------------------------------------------------
DOCKING_BOX = {
    "center_x": 9.5,
    "center_y": 5.2,
    "center_z": -11.4,
    "size_x": 20.0,
    "size_y": 20.0,
    "size_z": 20.0,
}

# Known pharmacologically critical DRD2 residues (from literature)
# These are used for annotation and validation, not filtering
CRITICAL_RESIDUES = {
    "ASP114": "Conserved salt bridge / H-bond anchor (TM3, Ballesteros-Weinstein 3.32)",
    "VAL115": "Hydrophobic contact in orthosteric pocket (TM3)",
    "CYS118": "Secondary pocket lining residue (TM3)",
    "THR119": "Polar contact in sub-pocket (TM3)",
    "ILE184": "Hydrophobic packing in extracellular loop (ECL2)",
    "SER193": "Serine motif for agonist/antagonist selectivity (TM5, 5.42)",
    "SER197": "Serine motif for agonist/antagonist selectivity (TM5, 5.46)",
    "PHE389": "Aromatic cage for pi-stacking (TM6)",
    "PHE390": "Aromatic cage for pi-stacking (TM6)",
    "TRP386": "Rotamer toggle switch (TM6, 6.48)",
    "HIS393": "Upper pocket polar contact (TM6)",
    "TYR416": "Tyrosine lid in TM7 (7.43)",
}


def generate_3d_conformer(smiles: str):
    """
    Generate an optimized 3D conformer from a SMILES string using RDKit.
    Returns the RDKit Mol object with 3D coordinates, or None on failure.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[ERROR] Failed to parse SMILES: {smiles}")
        return None

    mol = Chem.AddHs(mol)

    # Use ETKDG for high-quality initial embedding
    params = AllChem.ETKDGv3()
    params.randomSeed = 42  # Reproducible conformer
    params.numThreads = 0   # Use all available cores

    if AllChem.EmbedMolecule(mol, params) != 0:
        print("[WARNING] ETKDGv3 failed, trying ETKDGv2...")
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            print("[ERROR] Could not generate 3D conformer for Haloperidol.")
            return None

    # UFF force field optimization for realistic bond lengths/angles
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    except Exception as e:
        print(f"[WARNING] UFF optimization failed: {e}. Using unoptimized conformer.")

    return mol


def convert_sdf_to_pdbqt(sdf_path: str, pdbqt_path: str, obabel_path: str) -> bool:
    """
    Convert SDF to PDBQT using OpenBabel.
    Returns True on success, False on failure.
    """
    env = os.environ.copy()
    obabel_dir = os.path.dirname(obabel_path)
    candidate_data_dirs = [
        os.path.join(obabel_dir, "data"),
        os.path.join(obabel_dir, "bin", "data"),
        os.path.join(os.path.dirname(obabel_dir), "share", "openbabel"),
    ]
    for candidate in candidate_data_dirs:
        if os.path.exists(candidate):
            env["BABEL_DATADIR"] = candidate
            break

    cmd = [obabel_path, sdf_path, "-O", pdbqt_path, "-h"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0 or not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
        print(f"[ERROR] OpenBabel conversion failed:")
        print(f"  stdout: {result.stdout.strip()}")
        print(f"  stderr: {result.stderr.strip()}")
        return False

    print(f"[OK] PDBQT generated: {pdbqt_path} ({os.path.getsize(pdbqt_path)} bytes)")
    return True


def run_vina_docking(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    output_pdbqt: str,
    vina_path: str,
    exhaustiveness: int = 32,
    num_modes: int = 9,
    cpu: int = 0,  # 0 = auto-detect
) -> tuple:
    """
    Dock a ligand into a receptor using AutoDock Vina with high exhaustiveness.
    Returns (best_score, output_pdbqt_path) or (None, None) on failure.
    """
    cmd = [
        vina_path,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", output_pdbqt,
        "--center_x", str(DOCKING_BOX["center_x"]),
        "--center_y", str(DOCKING_BOX["center_y"]),
        "--center_z", str(DOCKING_BOX["center_z"]),
        "--size_x", str(DOCKING_BOX["size_x"]),
        "--size_y", str(DOCKING_BOX["size_y"]),
        "--size_z", str(DOCKING_BOX["size_z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ]

    if cpu > 0:
        cmd.extend(["--cpu", str(cpu)])

    print(f"\n[Vina] Docking with exhaustiveness={exhaustiveness} (this may take a few minutes)...")
    print(f"[Vina] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Vina docking failed:")
        print(f"  stderr: {result.stderr.strip()}")
        return None, None

    # Parse all modes from Vina output
    print("\n[Vina] Docking Results:")
    print("-" * 50)
    best_score = None
    for line in result.stdout.split("\n"):
        # Vina output format:  "   1       -8.5      0.000      0.000"
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    mode = int(parts[0])
                    score = float(parts[1])
                    if mode <= num_modes:
                        print(f"  Mode {mode}: {score:>7.2f} kcal/mol")
                        if best_score is None or score < best_score:
                            best_score = score
                except ValueError:
                    continue
    print("-" * 50)

    if best_score is not None:
        print(f"[Vina] Best binding affinity: {best_score:.2f} kcal/mol")
    else:
        print("[ERROR] Could not parse docking score from Vina output.")

    return best_score, output_pdbqt


def pdbqt_to_pdb(pdbqt_path: str, pdb_path: str, obabel_path: str) -> bool:
    """
    Convert PDBQT to PDB format using OpenBabel.
    Required because PLIP works with PDB files.
    """
    env = os.environ.copy()
    obabel_dir = os.path.dirname(obabel_path)
    candidate_data_dirs = [
        os.path.join(obabel_dir, "data"),
        os.path.join(obabel_dir, "bin", "data"),
        os.path.join(os.path.dirname(obabel_dir), "share", "openbabel"),
    ]
    for candidate in candidate_data_dirs:
        if os.path.exists(candidate):
            env["BABEL_DATADIR"] = candidate
            break

    cmd = [obabel_path, pdbqt_path, "-O", pdb_path, "-h"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0 or not os.path.exists(pdb_path):
        print(f"[ERROR] PDBQT->PDB conversion failed: {result.stderr.strip()}")
        return False

    return True


def merge_receptor_ligand_pdb(receptor_pdb: str, ligand_pdb: str, complex_pdb: str):
    """
    Merge receptor and docked ligand PDB files into a single complex PDB.
    The ligand is assigned chain 'Z' and residue name 'LIG' so PLIP
    can detect it as a heteroatom ligand.
    """
    with open(complex_pdb, "w") as out:
        # Write receptor atoms
        with open(receptor_pdb, "r") as rec:
            for line in rec:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    out.write(line)
            out.write("TER\n")

        # Write ligand atoms as HETATM with chain Z and residue LIG
        atom_serial = 9001  # Start ligand atom numbering high to avoid clashes
        with open(ligand_pdb, "r") as lig:
            for line in lig:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # PDB format: columns are fixed-width
                    # Rewrite as HETATM with chain Z, residue LIG, resnum 1
                    record = "HETATM"
                    serial = f"{atom_serial:>5d}"
                    atom_name = line[12:16]
                    alt_loc = line[16:17]
                    res_name = "LIG"
                    chain = "Z"
                    res_seq = "   1"
                    icode = line[26:27]
                    coords = line[30:54]
                    occupancy = line[54:60] if len(line) > 60 else " 1.00 "
                    bfactor = line[60:66] if len(line) > 66 else "  0.00"
                    element = line[76:78].strip() if len(line) > 78 else atom_name.strip()[0]

                    new_line = (
                        f"{record}{serial} {atom_name}{alt_loc}{res_name:>3s} "
                        f"{chain}{res_seq}{icode}   {coords}{occupancy}{bfactor}"
                        f"          {element:>2s}  \n"
                    )
                    out.write(new_line)
                    atom_serial += 1
            out.write("TER\n")

        out.write("END\n")

    print(f"[OK] Complex PDB written: {complex_pdb}")


def extract_plip_interactions(complex_pdb_path: str) -> dict:
    """
    Run PLIP on the merged receptor-ligand complex PDB file.
    Returns a structured dictionary of all detected non-covalent interactions.
    """
    try:
        from plip.structure.preparation import PDBComplex
    except ImportError:
        print("\n[ERROR] PLIP is not installed!")
        print("  Install it with: pip install plip")
        print("  If that fails on Windows, try: pip install plip --no-deps")
        sys.exit(1)

    print(f"\n[PLIP] Analyzing complex: {complex_pdb_path}")

    # ---------------------------------------------------------------
    # Monkey-patch OpenBabel's pybel.Molecule.write to handle missing
    # InChIKey format on Windows (openbabel-wheel doesn't ship it).
    # PLIP internally calls molecule.write(format='inchikey') which
    # crashes with: ValueError: inchikey is not a recognised format
    # This patch intercepts that specific call and returns "N/A".
    # ---------------------------------------------------------------
    try:
        import openbabel.pybel as pybel
        _original_mol_write = pybel.Molecule.write
        def _patched_mol_write(self, format='smi', filename=None, overwrite=False, opt=None):
            if format == 'inchikey':
                return "N/A"
            if opt is None:
                opt = {}
            return _original_mol_write(self, format=format, filename=filename, overwrite=overwrite, opt=opt)
        pybel.Molecule.write = _patched_mol_write
        print("[PLIP] Patched OpenBabel for Windows InChIKey compatibility.")
    except ImportError:
        pass  # openbabel not installed via pybel, PLIP may use its own

    mol = PDBComplex()
    mol.load_pdb(complex_pdb_path)

    # Find all ligands detected by PLIP
    lig_names = []
    for l in mol.ligands:
        lig_names.append(f"{l.hetid}:{l.chain}:{l.position}")
    print(f"[PLIP] Detected ligands: {lig_names}")

    interactions = {
        "hydrogen_bonds": [],
        "hydrophobic_contacts": [],
        "pi_stacking": [],
        "pi_cation": [],
        "salt_bridges": [],
        "water_bridges": [],
        "halogen_bonds": [],
        "metal_complexes": [],
    }

    ligand_found = False

    for ligand in mol.ligands:
        mol.characterize_complex(ligand)

    for bsite_key, binding_site in mol.interaction_sets.items():
        print(f"\n[PLIP] Processing binding site: {bsite_key}")
        ligand_found = True

        # -- Hydrogen Bonds (Protein as Donor) --
        # When protein is donor: d = protein donor atom, a = ligand acceptor atom
        # a_orig_idx = PDB serial of the ligand atom involved
        for hb in binding_site.hbonds_pdon:
            interactions["hydrogen_bonds"].append({
                "type": "protein_donor",
                "residue": hb.restype + str(hb.resnr),
                "residue_name": hb.restype,
                "residue_number": hb.resnr,
                "residue_chain": hb.reschain,
                "ligand_atom_pdb_idx": hb.a_orig_idx if hasattr(hb, 'a_orig_idx') else None,
                "protein_atom_pdb_idx": hb.d_orig_idx if hasattr(hb, 'd_orig_idx') else None,
                "distance_ah": round(hb.distance_ah, 2) if hasattr(hb, 'distance_ah') else None,
                "distance_ad": round(hb.distance_ad, 2) if hasattr(hb, 'distance_ad') else None,
                "angle": round(hb.angle, 1) if hasattr(hb, 'angle') else None,
                "donor_type": hb.dtype if hasattr(hb, 'dtype') else None,
                "acceptor_type": hb.atype if hasattr(hb, 'atype') else None,
                "is_critical": (hb.restype + str(hb.resnr)) in CRITICAL_RESIDUES,
            })

        # -- Hydrogen Bonds (Ligand as Donor) --
        # When ligand is donor: d = ligand donor atom, a = protein acceptor atom
        # d_orig_idx = PDB serial of the ligand atom involved
        for hb in binding_site.hbonds_ldon:
            interactions["hydrogen_bonds"].append({
                "type": "ligand_donor",
                "residue": hb.restype + str(hb.resnr),
                "residue_name": hb.restype,
                "residue_number": hb.resnr,
                "residue_chain": hb.reschain,
                "ligand_atom_pdb_idx": hb.d_orig_idx if hasattr(hb, 'd_orig_idx') else None,
                "protein_atom_pdb_idx": hb.a_orig_idx if hasattr(hb, 'a_orig_idx') else None,
                "distance_ah": round(hb.distance_ah, 2) if hasattr(hb, 'distance_ah') else None,
                "distance_ad": round(hb.distance_ad, 2) if hasattr(hb, 'distance_ad') else None,
                "angle": round(hb.angle, 1) if hasattr(hb, 'angle') else None,
                "donor_type": hb.dtype if hasattr(hb, 'dtype') else None,
                "acceptor_type": hb.atype if hasattr(hb, 'atype') else None,
                "is_critical": (hb.restype + str(hb.resnr)) in CRITICAL_RESIDUES,
            })

        # -- Hydrophobic Contacts --
        # ligatom_orig_idx = PDB serial of the ligand atom, bsatom_orig_idx = protein atom
        for hp in binding_site.hydrophobic_contacts:
            interactions["hydrophobic_contacts"].append({
                "residue": hp.restype + str(hp.resnr),
                "residue_name": hp.restype,
                "residue_number": hp.resnr,
                "residue_chain": hp.reschain,
                "ligand_atom_pdb_idx": hp.ligatom_orig_idx if hasattr(hp, 'ligatom_orig_idx') else None,
                "protein_atom_pdb_idx": hp.bsatom_orig_idx if hasattr(hp, 'bsatom_orig_idx') else None,
                "distance": round(hp.distance, 2) if hasattr(hp, 'distance') else None,
                "is_critical": (hp.restype + str(hp.resnr)) in CRITICAL_RESIDUES,
            })

        # -- Pi-Stacking --
        for ps in binding_site.pistacking:
            interactions["pi_stacking"].append({
                "residue": ps.restype + str(ps.resnr),
                "residue_name": ps.restype,
                "residue_number": ps.resnr,
                "residue_chain": ps.reschain,
                "type": ps.type if hasattr(ps, 'type') else None,  # T-shaped or parallel
                "distance": round(ps.distance, 2) if hasattr(ps, 'distance') else None,
                "angle": round(ps.angle, 1) if hasattr(ps, 'angle') else None,
                "offset": round(ps.offset, 2) if hasattr(ps, 'offset') else None,
                "is_critical": (ps.restype + str(ps.resnr)) in CRITICAL_RESIDUES,
            })

        # -- Pi-Cation Interactions --
        for pc in binding_site.pication_paro if hasattr(binding_site, 'pication_paro') else []:
            interactions["pi_cation"].append({
                "residue": pc.restype + str(pc.resnr),
                "residue_name": pc.restype,
                "residue_number": pc.resnr,
                "residue_chain": pc.reschain,
                "distance": round(pc.distance, 2) if hasattr(pc, 'distance') else None,
                "is_critical": (pc.restype + str(pc.resnr)) in CRITICAL_RESIDUES,
            })
        for pc in binding_site.pication_laro if hasattr(binding_site, 'pication_laro') else []:
            interactions["pi_cation"].append({
                "residue": pc.restype + str(pc.resnr),
                "residue_name": pc.restype,
                "residue_number": pc.resnr,
                "residue_chain": pc.reschain,
                "distance": round(pc.distance, 2) if hasattr(pc, 'distance') else None,
                "is_critical": (pc.restype + str(pc.resnr)) in CRITICAL_RESIDUES,
            })

        # -- Salt Bridges --
        for sb in binding_site.saltbridge_pneg if hasattr(binding_site, 'saltbridge_pneg') else []:
            interactions["salt_bridges"].append({
                "type": "protein_negative",
                "residue": sb.restype + str(sb.resnr),
                "residue_name": sb.restype,
                "residue_number": sb.resnr,
                "residue_chain": sb.reschain,

                "distance": round(sb.distance, 2) if hasattr(sb, 'distance') else None,
                "is_critical": (sb.restype + str(sb.resnr)) in CRITICAL_RESIDUES,
            })
        for sb in binding_site.saltbridge_lneg if hasattr(binding_site, 'saltbridge_lneg') else []:
            interactions["salt_bridges"].append({
                "type": "ligand_negative",
                "residue": sb.restype + str(sb.resnr),
                "residue_name": sb.restype,
                "residue_number": sb.resnr,
                "residue_chain": sb.reschain,

                "distance": round(sb.distance, 2) if hasattr(sb, 'distance') else None,
                "is_critical": (sb.restype + str(sb.resnr)) in CRITICAL_RESIDUES,
            })

        # -- Water Bridges --
        for wb in binding_site.water_bridges:
            interactions["water_bridges"].append({
                "residue": wb.restype + str(wb.resnr),
                "residue_name": wb.restype,
                "residue_number": wb.resnr,
                "residue_chain": wb.reschain,
                "distance_aw": round(wb.distance_aw, 2) if hasattr(wb, 'distance_aw') else None,
                "distance_dw": round(wb.distance_dw, 2) if hasattr(wb, 'distance_dw') else None,
                "is_critical": (wb.restype + str(wb.resnr)) in CRITICAL_RESIDUES,
            })

        # -- Halogen Bonds --
        for xb in binding_site.halogen_bonds if hasattr(binding_site, 'halogen_bonds') else []:
            interactions["halogen_bonds"].append({
                "residue": xb.restype + str(xb.resnr),
                "residue_name": xb.restype,
                "residue_number": xb.resnr,
                "residue_chain": xb.reschain,
                "distance": round(xb.distance, 2) if hasattr(xb, 'distance') else None,
                "don_angle": round(xb.don_angle, 1) if hasattr(xb, 'don_angle') else None,
                "acc_angle": round(xb.acc_angle, 1) if hasattr(xb, 'acc_angle') else None,
                "is_critical": (xb.restype + str(xb.resnr)) in CRITICAL_RESIDUES,
            })

        # -- Metal Complexes --
        for mc in binding_site.metal_complexes if hasattr(binding_site, 'metal_complexes') else []:
            interactions["metal_complexes"].append({
                "residue": mc.restype + str(mc.resnr),
                "residue_name": mc.restype,
                "residue_number": mc.resnr,
                "metal_type": mc.metal_type if hasattr(mc, 'metal_type') else None,
                "distance": round(mc.distance, 2) if hasattr(mc, 'distance') else None,
                "is_critical": (mc.restype + str(mc.resnr)) in CRITICAL_RESIDUES,
            })

    if not ligand_found:
        print("[WARNING] PLIP did not detect any ligands in the complex!")
        print("  This usually means the ligand HETATM records were not recognized.")
        print("  Check that the complex PDB has the ligand as HETATM with residue name 'LIG'.")

    return interactions


def build_interaction_fingerprint(interactions: dict) -> dict:
    """
    Convert raw PLIP interactions into a multi-hot encoded fingerprint.
    
    The fingerprint is a dictionary mapping each unique
    (residue, interaction_type) pair to its count and metadata.
    This is the core data structure that IF-ARS uses for comparison.
    """
    fingerprint = {}

    for hb in interactions["hydrogen_bonds"]:
        key = f"HBond_{hb['residue']}"
        if key not in fingerprint:
            fingerprint[key] = {
                "interaction_type": "hydrogen_bond",
                "residue": hb["residue"],
                "residue_name": hb["residue_name"],
                "residue_number": hb["residue_number"],
                "count": 0,
                "is_critical": hb["is_critical"],
                "details": [],
            }
        fingerprint[key]["count"] += 1
        fingerprint[key]["details"].append({
            "hb_type": hb["type"],
            "distance_ad": hb.get("distance_ad"),
            "angle": hb.get("angle"),
        })

    for hp in interactions["hydrophobic_contacts"]:
        key = f"Hydrophobic_{hp['residue']}"
        if key not in fingerprint:
            fingerprint[key] = {
                "interaction_type": "hydrophobic",
                "residue": hp["residue"],
                "residue_name": hp["residue_name"],
                "residue_number": hp["residue_number"],
                "count": 0,
                "is_critical": hp["is_critical"],
                "details": [],
            }
        fingerprint[key]["count"] += 1
        fingerprint[key]["details"].append({
            "distance": hp.get("distance"),
        })

    for ps in interactions["pi_stacking"]:
        key = f"PiStack_{ps['residue']}"
        if key not in fingerprint:
            fingerprint[key] = {
                "interaction_type": "pi_stacking",
                "residue": ps["residue"],
                "residue_name": ps["residue_name"],
                "residue_number": ps["residue_number"],
                "count": 0,
                "is_critical": ps["is_critical"],
                "details": [],
            }
        fingerprint[key]["count"] += 1
        fingerprint[key]["details"].append({
            "stack_type": ps.get("type"),
            "distance": ps.get("distance"),
            "angle": ps.get("angle"),
        })

    for pc in interactions["pi_cation"]:
        key = f"PiCat_{pc['residue']}"
        if key not in fingerprint:
            fingerprint[key] = {
                "interaction_type": "pi_cation",
                "residue": pc["residue"],
                "residue_name": pc["residue_name"],
                "residue_number": pc["residue_number"],
                "count": 0,
                "is_critical": pc["is_critical"],
                "details": [],
            }
        fingerprint[key]["count"] += 1

    for sb in interactions["salt_bridges"]:
        key = f"SaltBridge_{sb['residue']}"
        if key not in fingerprint:
            fingerprint[key] = {
                "interaction_type": "salt_bridge",
                "residue": sb["residue"],
                "residue_name": sb["residue_name"],
                "residue_number": sb["residue_number"],
                "count": 0,
                "is_critical": sb["is_critical"],
                "details": [],
            }
        fingerprint[key]["count"] += 1
        fingerprint[key]["details"].append({
            "sb_type": sb.get("type"),
            "distance": sb.get("distance"),
        })

    for wb in interactions["water_bridges"]:
        key = f"WaterBridge_{wb['residue']}"
        if key not in fingerprint:
            fingerprint[key] = {
                "interaction_type": "water_bridge",
                "residue": wb["residue"],
                "residue_name": wb["residue_name"],
                "residue_number": wb["residue_number"],
                "count": 0,
                "is_critical": wb["is_critical"],
                "details": [],
            }
        fingerprint[key]["count"] += 1

    for xb in interactions["halogen_bonds"]:
        key = f"HalogenBond_{xb['residue']}"
        if key not in fingerprint:
            fingerprint[key] = {
                "interaction_type": "halogen_bond",
                "residue": xb["residue"],
                "residue_name": xb["residue_name"],
                "residue_number": xb["residue_number"],
                "count": 0,
                "is_critical": xb["is_critical"],
                "details": [],
            }
        fingerprint[key]["count"] += 1

    return fingerprint


def build_multihot_vector(fingerprint: dict, all_possible_keys: list = None) -> list:
    """
    Convert the fingerprint dictionary into a binary multi-hot vector.
    Each position corresponds to a unique (residue, interaction_type) pair.
    1 = interaction present, 0 = interaction absent.
    
    If all_possible_keys is None, the vector is built only from detected
    interactions (used for the reference). During training, the same key
    order will be used to encode generated molecules.
    """
    if all_possible_keys is None:
        all_possible_keys = sorted(fingerprint.keys())

    vector = []
    for key in all_possible_keys:
        vector.append(1 if key in fingerprint else 0)

    return vector, all_possible_keys


def print_interaction_summary(interactions: dict, fingerprint: dict):
    """
    Print a beautiful summary of all detected interactions.
    """
    print("\n" + "=" * 70)
    print("  HALOPERIDOL x DRD2 INTERACTION FINGERPRINT SUMMARY")
    print("=" * 70)

    total_interactions = sum(
        len(v) for v in interactions.values()
    )
    critical_count = sum(
        1 for fp in fingerprint.values() if fp["is_critical"]
    )

    print(f"\n  Total Interactions Detected: {total_interactions}")
    print(f"  Unique Fingerprint Keys:    {len(fingerprint)}")
    print(f"  Critical Residue Contacts:  {critical_count}")
    print()

    # Group by interaction type
    type_counts = {}
    for fp_key, fp_val in fingerprint.items():
        itype = fp_val["interaction_type"]
        type_counts[itype] = type_counts.get(itype, 0) + fp_val["count"]

    print("  Interaction Breakdown:")
    print("  " + "-" * 40)
    type_labels = {
        "hydrogen_bond": "Hydrogen Bonds",
        "hydrophobic": "Hydrophobic Contacts",
        "pi_stacking": "Pi-Stacking",
        "pi_cation": "Pi-Cation",
        "salt_bridge": "Salt Bridges",
        "water_bridge": "Water Bridges",
        "halogen_bond": "Halogen Bonds",
    }
    for itype, label in type_labels.items():
        count = type_counts.get(itype, 0)
        if count > 0:
            print(f"    {label:<25s}: {count}")

    # Critical residue analysis
    print("\n  Critical Pharmacological Contacts:")
    print("  " + "-" * 40)
    for fp_key, fp_val in sorted(fingerprint.items()):
        if fp_val["is_critical"]:
            res = fp_val["residue"]
            itype = fp_val["interaction_type"].replace("_", " ").title()
            note = CRITICAL_RESIDUES.get(fp_val["residue_name"] + str(fp_val["residue_number"]), "")
            print(f"    [OK] {res:<10s} | {itype:<20s} | {note}")

    # Non-critical contacts
    non_critical = [fp for fp in fingerprint.values() if not fp["is_critical"]]
    if non_critical:
        print(f"\n  Additional Contacts ({len(non_critical)}):")
        print("  " + "-" * 40)
        for fp_val in non_critical:
            res = fp_val["residue"]
            itype = fp_val["interaction_type"].replace("_", " ").title()
            print(f"    * {res:<10s} | {itype}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Haloperidol reference interaction fingerprint for IF-ARS"
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=32,
        help="Vina exhaustiveness for reference docking (default: 32, higher = more accurate)"
    )
    parser.add_argument(
        "--vina-path", type=str, default=None,
        help="Custom path to vina.exe"
    )
    parser.add_argument(
        "--obabel-path", type=str, default=None,
        help="Custom path to obabel.exe"
    )
    parser.add_argument(
        "--receptor-path", type=str, default=None,
        help="Custom path to receptor PDBQT (default: data/raw/drd2_clean.pdbqt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/reference/)"
    )

    args = parser.parse_args()

    # -- Resolve paths --
    vina_path = resolve_vina_path(args.vina_path)
    obabel_path = resolve_obabel_path(args.obabel_path)
    receptor_path = args.receptor_path or os.path.join(PROJECT_ROOT, "data", "raw", "drd2_clean.pdbqt")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "data", "reference")

    # -- Pre-flight checks --
    print("=" * 70)
    print("  HALOPERIDOL REFERENCE FINGERPRINT EXTRACTION")
    print("  IF-ARS Task 1 | Phase 3 of MOLECULEX Pipeline")
    print("=" * 70)
    print(f"\n  Drug:           {HALOPERIDOL_NAME}")
    print(f"  SMILES:         {HALOPERIDOL_SMILES}")
    print(f"  Receptor:       {receptor_path}")
    print(f"  Vina:           {vina_path}")
    print(f"  OpenBabel:      {obabel_path}")
    print(f"  Exhaustiveness: {args.exhaustiveness}")
    print(f"  Output:         {output_dir}")

    # Validate binaries exist
    errors = []
    if not os.path.isfile(vina_path):
        errors.append(f"  [X] Vina not found: {vina_path}")
    if not os.path.isfile(obabel_path):
        errors.append(f"  [X] OpenBabel not found: {obabel_path}")
    if not os.path.isfile(receptor_path):
        errors.append(f"  [X] Receptor not found: {receptor_path}")

    if errors:
        print("\n[PREFLIGHT FAILED]")
        for e in errors:
            print(e)
        print("\nPlease ensure all binaries are installed. See HPC_WINDOWS_SETUP.md.")
        sys.exit(1)
    else:
        print("\n  [OK] All binaries verified\n")

    # -- Create output directories --
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(PROJECT_ROOT, "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate unique temp file names
    run_id = uuid.uuid4().hex[:8]
    temp_sdf = os.path.join(temp_dir, f"haloperidol_{run_id}.sdf")
    temp_lig_pdbqt = os.path.join(temp_dir, f"haloperidol_{run_id}.pdbqt")
    temp_docked_pdbqt = os.path.join(temp_dir, f"haloperidol_{run_id}_docked.pdbqt")
    temp_docked_pdb = os.path.join(temp_dir, f"haloperidol_{run_id}_docked.pdb")
    temp_receptor_pdb = os.path.join(temp_dir, f"receptor_{run_id}.pdb")
    temp_complex_pdb = os.path.join(temp_dir, f"complex_{run_id}.pdb")

    # Keep the final docked pose and complex for reference
    final_docked_pdb = os.path.join(output_dir, "haloperidol_docked_pose.pdb")
    final_complex_pdb = os.path.join(output_dir, "haloperidol_drd2_complex.pdb")

    try:
        # -- Step 1: Generate 3D Conformer --
        print("\n[Step 1/6] Generating 3D conformer for Haloperidol...")
        from rdkit import Chem
        from rdkit.Chem import QED, Descriptors

        mol = generate_3d_conformer(HALOPERIDOL_SMILES)
        if mol is None:
            print("[FATAL] Could not generate 3D conformer. Aborting.")
            sys.exit(1)

        # Calculate basic properties
        mol_2d = Chem.MolFromSmiles(HALOPERIDOL_SMILES)
        qed_score = QED.qed(mol_2d)
        mw = Descriptors.MolWt(mol_2d)
        print(f"  [OK] QED: {qed_score:.3f}")
        print(f"  [OK] MW:  {mw:.1f} Da")
        print(f"  [OK] Atoms: {mol.GetNumAtoms()} (with H)")

        # Write SDF
        writer = Chem.SDWriter(temp_sdf)
        writer.write(mol)
        writer.close()
        print(f"  [OK] SDF written: {temp_sdf}")

        # -- Step 2: Convert to PDBQT --
        print("\n[Step 2/6] Converting SDF -> PDBQT via OpenBabel...")
        if not convert_sdf_to_pdbqt(temp_sdf, temp_lig_pdbqt, obabel_path):
            print("[FATAL] SDF to PDBQT conversion failed. Aborting.")
            sys.exit(1)

        # -- Step 3: Dock with High Exhaustiveness --
        print(f"\n[Step 3/6] Docking Haloperidol into DRD2 (exhaustiveness={args.exhaustiveness})...")
        best_score, _ = run_vina_docking(
            ligand_pdbqt=temp_lig_pdbqt,
            receptor_pdbqt=receptor_path,
            output_pdbqt=temp_docked_pdbqt,
            vina_path=vina_path,
            exhaustiveness=args.exhaustiveness,
        )
        if best_score is None:
            print("[FATAL] Docking failed. Aborting.")
            sys.exit(1)

        # -- Step 4: Convert docked pose & receptor to PDB --
        print("\n[Step 4/6] Converting docked pose to PDB for PLIP...")
        if not pdbqt_to_pdb(temp_docked_pdbqt, temp_docked_pdb, obabel_path):
            print("[FATAL] Docked PDBQT to PDB conversion failed. Aborting.")
            sys.exit(1)
        print(f"  [OK] Docked ligand PDB: {temp_docked_pdb}")

        if not pdbqt_to_pdb(receptor_path, temp_receptor_pdb, obabel_path):
            print("[FATAL] Receptor PDBQT to PDB conversion failed. Aborting.")
            sys.exit(1)
        print(f"  [OK] Receptor PDB: {temp_receptor_pdb}")

        # -- Step 5: Merge into complex & run PLIP --
        print("\n[Step 5/6] Merging receptor + ligand and running PLIP...")
        merge_receptor_ligand_pdb(temp_receptor_pdb, temp_docked_pdb, temp_complex_pdb)

        # Copy final files for reference
        import shutil
        shutil.copy2(temp_docked_pdb, final_docked_pdb)
        shutil.copy2(temp_complex_pdb, final_complex_pdb)

        interactions = extract_plip_interactions(temp_complex_pdb)

        # -- Step 6: Build & Save Fingerprint --
        print("\n[Step 6/6] Building interaction fingerprint...")
        fingerprint = build_interaction_fingerprint(interactions)
        multihot_vector, key_order = build_multihot_vector(fingerprint)

        # Print summary
        print_interaction_summary(interactions, fingerprint)

        # -- Save JSON --
        output_json = os.path.join(output_dir, "haloperidol_fingerprint.json")
        output_data = {
            "metadata": {
                "drug_name": HALOPERIDOL_NAME,
                "drug_smiles": HALOPERIDOL_SMILES,
                "receptor_pdb_id": "6CM4",
                "receptor_path": receptor_path,
                "docking_score_kcal_mol": best_score,
                "docking_exhaustiveness": args.exhaustiveness,
                "qed": qed_score,
                "molecular_weight": mw,
                "extraction_date": datetime.now().isoformat(),
                "vina_version": vina_path,
                "docking_box": DOCKING_BOX,
            },
            "critical_residues": CRITICAL_RESIDUES,
            "raw_interactions": interactions,
            "fingerprint": fingerprint,
            "multihot_key_order": key_order,
            "multihot_vector": multihot_vector,
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n[SUCCESS] Reference fingerprint saved to:")
        print(f"  -> {output_json}")
        print(f"  -> {final_docked_pdb}")
        print(f"  -> {final_complex_pdb}")
        print(f"\n  Fingerprint has {len(fingerprint)} unique interaction keys.")
        print(f"  Multi-hot vector length: {len(multihot_vector)}")
        print(f"  Haloperidol best docking score: {best_score:.2f} kcal/mol")

    finally:
        # -- Cleanup temp files --
        for f in [temp_sdf, temp_lig_pdbqt, temp_docked_pdbqt,
                  temp_docked_pdb, temp_receptor_pdb, temp_complex_pdb]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
