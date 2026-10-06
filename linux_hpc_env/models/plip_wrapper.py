"""
PLIP Wrapper Module for IF-ARS Pipeline
=========================================
Task 2 of the IF-ARS Pipeline (Phase 3)

A reusable Python module that wraps the Protein-Ligand Interaction Profiler
(PLIP) for use during RL training. This module provides:

  1. plip_analyze_pose()  - Extracts all non-covalent interactions from a
                            docked ligand-receptor complex
  2. compare_fingerprints() - Computes the overlap score between a generated
                              molecule's interactions and the Haloperidol
                              reference fingerprint
  3. PLIPAnalyzer class    - High-level interface that loads the reference
                             fingerprint once and efficiently scores new
                             molecules during training

This module is designed to be called thousands of times during PPO training.
It handles:
  - Windows InChIKey compatibility (openbabel-wheel patch)
  - Temporary file management with zero disk leaks
  - Graceful error handling (returns empty fingerprint on failure)
  - Efficient caching of the reference fingerprint

Usage during training:
    from models.plip_wrapper import PLIPAnalyzer

    analyzer = PLIPAnalyzer(
        receptor_pdbqt="data/raw/drd2_clean.pdbqt",
        reference_json="data/reference/haloperidol_fingerprint.json",
        obabel_path=None,  # auto-detect
    )

    # Score a docked pose
    result = analyzer.score_docked_pose(docked_pdbqt_path)
    # result = {
    #     "overlap_score": 0.72,       # 72% of Haloperidol's interactions matched
    #     "matched_keys": [...],       # Which interactions were reproduced
    #     "missed_keys": [...],        # Which critical interactions were missed
    #     "novel_keys": [...],         # New interactions not in Haloperidol
    #     "critical_overlap": 0.875,   # 7/8 critical residues matched
    #     "fingerprint": {...},        # Full fingerprint dictionary
    # }
"""

import os
import sys
import json
import uuid
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Import the binary resolution functions from reward_oracle
from models.reward_oracle import resolve_obabel_path


# -----------------------------------------------------------------------
# Windows InChIKey Compatibility Patch
# Must be applied BEFORE importing PLIP
# -----------------------------------------------------------------------
_plip_patched = False

def _apply_plip_windows_patch():
    """
    Patch OpenBabel's pybel.Molecule.write to handle missing InChIKey
    format on Windows. The openbabel-wheel package does not include
    InChIKey support, causing PLIP to crash with:
        ValueError: inchikey is not a recognised Open Babel format

    This patch intercepts that specific call and returns "N/A".
    Safe to call multiple times (applies only once).
    """
    global _plip_patched
    if _plip_patched:
        return

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
        _plip_patched = True
    except ImportError:
        pass


# -----------------------------------------------------------------------
# Low-Level PLIP Functions
# -----------------------------------------------------------------------

def _get_obabel_env(obabel_path: str) -> dict:
    """Build an environment dict with BABEL_DATADIR set correctly."""
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
    return env


def _pdbqt_to_pdb(pdbqt_path: str, pdb_path: str, obabel_path: str) -> bool:
    """Convert PDBQT to PDB format using OpenBabel. Returns True on success."""
    env = _get_obabel_env(obabel_path)
    cmd = [obabel_path, pdbqt_path, "-O", pdb_path, "-h"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.returncode == 0 and os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0


def _merge_receptor_ligand_pdb(receptor_pdb: str, ligand_pdb: str, complex_pdb: str):
    """
    Merge receptor and docked ligand PDB files into a single complex PDB.
    The ligand is written as HETATM records with chain 'Z' and residue
    name 'LIG' so that PLIP can detect it as a small-molecule ligand.
    """
    with open(complex_pdb, "w") as out:
        # Write receptor ATOM records
        with open(receptor_pdb, "r") as rec:
            for line in rec:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    out.write(line)
            out.write("TER\n")

        # Write ligand atoms as HETATM with chain Z, residue LIG
        atom_serial = 9001
        with open(ligand_pdb, "r") as lig:
            for line in lig:
                if line.startswith("ATOM") or line.startswith("HETATM"):
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


def plip_analyze_pose(complex_pdb_path: str) -> dict:
    """
    Run PLIP on a merged receptor-ligand complex PDB file.

    Args:
        complex_pdb_path: Path to a PDB file containing both the receptor
                          and the docked ligand (ligand as HETATM with
                          residue name 'LIG', chain 'Z').

    Returns:
        A dictionary with keys:
            hydrogen_bonds, hydrophobic_contacts, pi_stacking,
            pi_cation, salt_bridges, water_bridges, halogen_bonds,
            metal_complexes
        Each value is a list of interaction dictionaries.
        Returns an empty interaction dict on any failure.
    """
    # Apply the Windows compatibility patch before importing PLIP
    _apply_plip_windows_patch()

    empty_result = {
        "hydrogen_bonds": [],
        "hydrophobic_contacts": [],
        "pi_stacking": [],
        "pi_cation": [],
        "salt_bridges": [],
        "water_bridges": [],
        "halogen_bonds": [],
        "metal_complexes": [],
    }

    try:
        from plip.structure.preparation import PDBComplex
    except ImportError:
        return empty_result

    try:
        mol = PDBComplex()
        mol.load_pdb(complex_pdb_path)

        for ligand in mol.ligands:
            mol.characterize_complex(ligand)

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

        for bsite_key, bs in mol.interaction_sets.items():
            # Hydrogen Bonds (Protein as Donor)
            # When protein is donor: a = ligand acceptor, d = protein donor
            for hb in bs.hbonds_pdon:
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
                })

            # Hydrogen Bonds (Ligand as Donor)
            # When ligand is donor: d = ligand donor, a = protein acceptor
            for hb in bs.hbonds_ldon:
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
                })

            # Hydrophobic Contacts
            # ligatom_orig_idx = ligand atom PDB serial, bsatom_orig_idx = protein atom
            for hp in bs.hydrophobic_contacts:
                interactions["hydrophobic_contacts"].append({
                    "residue": hp.restype + str(hp.resnr),
                    "residue_name": hp.restype,
                    "residue_number": hp.resnr,
                    "residue_chain": hp.reschain,
                    "ligand_atom_pdb_idx": hp.ligatom_orig_idx if hasattr(hp, 'ligatom_orig_idx') else None,
                    "protein_atom_pdb_idx": hp.bsatom_orig_idx if hasattr(hp, 'bsatom_orig_idx') else None,
                    "distance": round(hp.distance, 2) if hasattr(hp, 'distance') else None,
                })

            # Pi-Stacking
            for ps in bs.pistacking:
                interactions["pi_stacking"].append({
                    "residue": ps.restype + str(ps.resnr),
                    "residue_name": ps.restype,
                    "residue_number": ps.resnr,
                    "residue_chain": ps.reschain,
                    "stack_type": ps.type if hasattr(ps, 'type') else None,
                    "distance": round(ps.distance, 2) if hasattr(ps, 'distance') else None,
                    "angle": round(ps.angle, 1) if hasattr(ps, 'angle') else None,
                })

            # Pi-Cation
            for pc in (bs.pication_paro if hasattr(bs, 'pication_paro') else []):
                interactions["pi_cation"].append({
                    "residue": pc.restype + str(pc.resnr),
                    "residue_name": pc.restype,
                    "residue_number": pc.resnr,
                    "distance": round(pc.distance, 2) if hasattr(pc, 'distance') else None,
                })
            for pc in (bs.pication_laro if hasattr(bs, 'pication_laro') else []):
                interactions["pi_cation"].append({
                    "residue": pc.restype + str(pc.resnr),
                    "residue_name": pc.restype,
                    "residue_number": pc.resnr,
                    "distance": round(pc.distance, 2) if hasattr(pc, 'distance') else None,
                })

            # Salt Bridges
            for sb in (bs.saltbridge_pneg if hasattr(bs, 'saltbridge_pneg') else []):
                interactions["salt_bridges"].append({
                    "type": "protein_negative",
                    "residue": sb.restype + str(sb.resnr),
                    "residue_name": sb.restype,
                    "residue_number": sb.resnr,
                    "distance": round(sb.distance, 2) if hasattr(sb, 'distance') else None,
                })
            for sb in (bs.saltbridge_lneg if hasattr(bs, 'saltbridge_lneg') else []):
                interactions["salt_bridges"].append({
                    "type": "ligand_negative",
                    "residue": sb.restype + str(sb.resnr),
                    "residue_name": sb.restype,
                    "residue_number": sb.resnr,
                    "distance": round(sb.distance, 2) if hasattr(sb, 'distance') else None,
                })

            # Water Bridges
            for wb in bs.water_bridges:
                interactions["water_bridges"].append({
                    "residue": wb.restype + str(wb.resnr),
                    "residue_name": wb.restype,
                    "residue_number": wb.resnr,
                    "distance_aw": round(wb.distance_aw, 2) if hasattr(wb, 'distance_aw') else None,
                    "distance_dw": round(wb.distance_dw, 2) if hasattr(wb, 'distance_dw') else None,
                })

            # Halogen Bonds
            for xb in (bs.halogen_bonds if hasattr(bs, 'halogen_bonds') else []):
                interactions["halogen_bonds"].append({
                    "residue": xb.restype + str(xb.resnr),
                    "residue_name": xb.restype,
                    "residue_number": xb.resnr,
                    "distance": round(xb.distance, 2) if hasattr(xb, 'distance') else None,
                })

            # Metal Complexes
            for mc in (bs.metal_complexes if hasattr(bs, 'metal_complexes') else []):
                interactions["metal_complexes"].append({
                    "residue": mc.restype + str(mc.resnr),
                    "residue_name": mc.restype,
                    "residue_number": mc.resnr,
                    "distance": round(mc.distance, 2) if hasattr(mc, 'distance') else None,
                })

        return interactions

    except Exception as e:
        # During training, we cannot let PLIP errors crash the entire run.
        # Return empty interactions and let the reward shaper handle it.
        print(f"[PLIP WARNING] Analysis failed: {e}")
        return empty_result


def build_fingerprint(interactions: dict) -> dict:
    """
    Convert raw PLIP interactions into a fingerprint dictionary.

    Each unique (residue, interaction_type) pair becomes a key in the
    fingerprint. The value contains the count and interaction metadata.

    This is the core data structure used by IF-ARS for comparison.
    """
    fingerprint = {}

    type_map = {
        "hydrogen_bonds": "hydrogen_bond",
        "hydrophobic_contacts": "hydrophobic",
        "pi_stacking": "pi_stacking",
        "pi_cation": "pi_cation",
        "salt_bridges": "salt_bridge",
        "water_bridges": "water_bridge",
        "halogen_bonds": "halogen_bond",
        "metal_complexes": "metal_complex",
    }

    prefix_map = {
        "hydrogen_bonds": "HBond",
        "hydrophobic_contacts": "Hydrophobic",
        "pi_stacking": "PiStack",
        "pi_cation": "PiCat",
        "salt_bridges": "SaltBridge",
        "water_bridges": "WaterBridge",
        "halogen_bonds": "HalogenBond",
        "metal_complexes": "MetalComplex",
    }

    for interaction_category, interaction_list in interactions.items():
        prefix = prefix_map.get(interaction_category, interaction_category)
        itype = type_map.get(interaction_category, interaction_category)

        for interaction in interaction_list:
            residue = interaction.get("residue", "UNK0")
            key = f"{prefix}_{residue}"

            if key not in fingerprint:
                fingerprint[key] = {
                    "interaction_type": itype,
                    "residue": residue,
                    "residue_name": interaction.get("residue_name", "UNK"),
                    "residue_number": interaction.get("residue_number", 0),
                    "count": 0,
                }
            fingerprint[key]["count"] += 1

    return fingerprint


def compare_fingerprints(
    generated_fingerprint: dict,
    reference_fingerprint: dict,
    critical_residues: dict = None,
) -> dict:
    """
    Compare a generated molecule's interaction fingerprint against
    the Haloperidol reference fingerprint.

    Args:
        generated_fingerprint: Fingerprint dict from build_fingerprint()
        reference_fingerprint: Fingerprint dict from haloperidol_fingerprint.json
        critical_residues: Optional dict of {residue: description} for
                           pharmacologically critical residues

    Returns:
        {
            "overlap_score": float,       # Fraction of reference keys matched (0.0 - 1.0)
            "critical_overlap": float,    # Fraction of critical residue keys matched
            "matched_keys": list,         # Reference keys present in generated
            "missed_keys": list,          # Reference keys absent in generated
            "novel_keys": list,           # Generated keys not in reference
            "total_reference": int,       # Number of reference fingerprint keys
            "total_generated": int,       # Number of generated fingerprint keys
            "critical_matched": list,     # Critical residue keys that matched
            "critical_missed": list,      # Critical residue keys that were missed
        }
    """
    if critical_residues is None:
        critical_residues = {}

    ref_keys = set(reference_fingerprint.keys())
    gen_keys = set(generated_fingerprint.keys())

    matched = ref_keys & gen_keys
    missed = ref_keys - gen_keys
    novel = gen_keys - ref_keys

    # Overall overlap
    overlap_score = len(matched) / len(ref_keys) if ref_keys else 0.0

    # Critical residue overlap
    critical_ref_keys = set()
    for key, val in reference_fingerprint.items():
        res_label = val.get("residue_name", "") + str(val.get("residue_number", ""))
        if res_label in critical_residues:
            critical_ref_keys.add(key)

    critical_matched = critical_ref_keys & gen_keys
    critical_missed = critical_ref_keys - gen_keys
    critical_overlap = len(critical_matched) / len(critical_ref_keys) if critical_ref_keys else 0.0

    return {
        "overlap_score": round(overlap_score, 4),
        "critical_overlap": round(critical_overlap, 4),
        "matched_keys": sorted(list(matched)),
        "missed_keys": sorted(list(missed)),
        "novel_keys": sorted(list(novel)),
        "total_reference": len(ref_keys),
        "total_generated": len(gen_keys),
        "critical_matched": sorted(list(critical_matched)),
        "critical_missed": sorted(list(critical_missed)),
    }


# -----------------------------------------------------------------------
# High-Level PLIPAnalyzer Class (Used During Training)
# -----------------------------------------------------------------------

class PLIPAnalyzer:
    """
    High-level interface for running PLIP analysis during PPO training.

    Loads the Haloperidol reference fingerprint once on initialization,
    then provides efficient methods to score new docked poses.

    Usage:
        analyzer = PLIPAnalyzer(
            receptor_pdbqt="data/raw/drd2_clean.pdbqt",
            reference_json="data/reference/haloperidol_fingerprint.json",
        )

        # Option A: Score from a docked PDBQT file
        result = analyzer.score_docked_pose("path/to/docked_out.pdbqt")

        # Option B: Score from raw interaction dict (if you already ran PLIP)
        result = analyzer.score_interactions(interactions_dict)
    """

    def __init__(
        self,
        receptor_pdbqt: str = "data/raw/drd2_clean.pdbqt",
        reference_json: str = "data/reference/haloperidol_fingerprint.json",
        obabel_path: str = None,
    ):
        # Resolve paths relative to project root
        if not os.path.isabs(receptor_pdbqt):
            receptor_pdbqt = os.path.join(PROJECT_ROOT, receptor_pdbqt)
        if not os.path.isabs(reference_json):
            reference_json = os.path.join(PROJECT_ROOT, reference_json)

        self.receptor_pdbqt = receptor_pdbqt
        self.obabel_path = resolve_obabel_path(obabel_path)
        self.temp_dir = os.path.join(PROJECT_ROOT, "data", "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Pre-convert receptor PDBQT to PDB once (cached for all calls)
        self._receptor_pdb_cache = os.path.join(
            self.temp_dir, "receptor_plip_cache.pdb"
        )
        if not os.path.exists(self._receptor_pdb_cache):
            success = _pdbqt_to_pdb(
                self.receptor_pdbqt, self._receptor_pdb_cache, self.obabel_path
            )
            if not success:
                print(f"[PLIPAnalyzer ERROR] Failed to convert receptor to PDB.")
                print(f"  Receptor: {self.receptor_pdbqt}")
                print(f"  OpenBabel: {self.obabel_path}")

        # Load reference fingerprint
        self.reference_fingerprint = {}
        self.critical_residues = {}
        self.reference_metadata = {}

        if os.path.exists(reference_json):
            with open(reference_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.reference_fingerprint = data.get("fingerprint", {})
            self.critical_residues = data.get("critical_residues", {})
            self.reference_metadata = data.get("metadata", {})
            print(f"[PLIPAnalyzer] Loaded reference fingerprint: "
                  f"{len(self.reference_fingerprint)} keys, "
                  f"{len(self.critical_residues)} critical residues")
        else:
            print(f"[PLIPAnalyzer WARNING] Reference JSON not found: {reference_json}")
            print(f"  Run scripts/extract_haloperidol_fingerprint.py first!")

        # Apply Windows compatibility patch
        _apply_plip_windows_patch()

    def score_docked_pose(self, docked_pdbqt_path: str) -> dict:
        """
        Analyze a docked ligand PDBQT and compare against the reference.

        Args:
            docked_pdbqt_path: Path to a Vina output PDBQT file
                               (contains only the best docked pose)

        Returns:
            Comparison result dict from compare_fingerprints(), plus
            the raw "interactions" and "fingerprint" data.
            Returns a zeroed result on any failure.
        """
        empty_result = {
            "overlap_score": 0.0,
            "critical_overlap": 0.0,
            "matched_keys": [],
            "missed_keys": list(self.reference_fingerprint.keys()),
            "novel_keys": [],
            "total_reference": len(self.reference_fingerprint),
            "total_generated": 0,
            "critical_matched": [],
            "critical_missed": [],
            "interactions": {},
            "fingerprint": {},
        }

        if not os.path.exists(docked_pdbqt_path):
            return empty_result

        # Generate unique, PID-safe temp file names for this call
        call_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        temp_mode1_pdbqt = os.path.join(self.temp_dir, f"plip_mode1_{call_id}.pdbqt")
        temp_lig_pdb = os.path.join(self.temp_dir, f"plip_lig_{call_id}.pdb")
        temp_complex_pdb = os.path.join(self.temp_dir, f"plip_cmplx_{call_id}.pdb")

        try:
            # Step 0: Extract Mode 1 (lowest energy binding pose) from multi-model Vina output
            with open(docked_pdbqt_path, "r", encoding="utf-8", errors="ignore") as f_in:
                lines = f_in.readlines()

            mode1_lines = []
            in_mode1 = False
            has_models = any(line.startswith("MODEL") for line in lines)
            if has_models:
                for line in lines:
                    if line.startswith("MODEL 1") or (not in_mode1 and line.startswith("MODEL")):
                        in_mode1 = True
                    elif line.startswith("ENDMDL"):
                        if in_mode1:
                            mode1_lines.append(line)
                            break
                    if in_mode1:
                        mode1_lines.append(line)
            else:
                mode1_lines = lines

            if not mode1_lines:
                return empty_result

            with open(temp_mode1_pdbqt, "w", encoding="utf-8") as f_out:
                f_out.writelines(mode1_lines)

            # Step 1: Convert isolated Mode 1 ligand PDBQT to PDB
            if not _pdbqt_to_pdb(temp_mode1_pdbqt, temp_lig_pdb, self.obabel_path):
                return empty_result

            # Step 2: Merge receptor (cached PDB) + ligand into complex
            if not os.path.exists(self._receptor_pdb_cache):
                return empty_result
            _merge_receptor_ligand_pdb(
                self._receptor_pdb_cache, temp_lig_pdb, temp_complex_pdb
            )

            # Step 3: Run PLIP
            interactions = plip_analyze_pose(temp_complex_pdb)

            # Step 4: Build fingerprint and compare
            fingerprint = build_fingerprint(interactions)
            comparison = compare_fingerprints(
                fingerprint, self.reference_fingerprint, self.critical_residues
            )

            # Attach raw data to the result
            comparison["interactions"] = interactions
            comparison["fingerprint"] = fingerprint

            return comparison

        except Exception as e:
            print(f"[PLIPAnalyzer WARNING] score_docked_pose failed: {e}")
            return empty_result

        finally:
            # Clean up temp files
            for f in [temp_mode1_pdbqt, temp_lig_pdb, temp_complex_pdb]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    def score_interactions(self, interactions: dict) -> dict:
        """
        Score a pre-computed interactions dict against the reference.
        Use this if you've already run PLIP yourself.

        Args:
            interactions: Raw interaction dict from plip_analyze_pose()

        Returns:
            Comparison result dict.
        """
        fingerprint = build_fingerprint(interactions)
        comparison = compare_fingerprints(
            fingerprint, self.reference_fingerprint, self.critical_residues
        )
        comparison["interactions"] = interactions
        comparison["fingerprint"] = fingerprint
        return comparison

    def get_reference_info(self) -> dict:
        """Return a summary of the loaded reference fingerprint."""
        return {
            "drug_name": self.reference_metadata.get("drug_name", "Unknown"),
            "drug_smiles": self.reference_metadata.get("drug_smiles", ""),
            "docking_score": self.reference_metadata.get("docking_score_kcal_mol", None),
            "num_fingerprint_keys": len(self.reference_fingerprint),
            "num_critical_residues": len(self.critical_residues),
            "critical_residues": list(self.critical_residues.keys()),
            "fingerprint_keys": sorted(list(self.reference_fingerprint.keys())),
        }


# -----------------------------------------------------------------------
# Standalone Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  PLIP Wrapper Module - Self Test")
    print("=" * 60)

    # Test 1: Load the reference fingerprint
    analyzer = PLIPAnalyzer()
    info = analyzer.get_reference_info()

    print(f"\n  Reference Drug: {info['drug_name']}")
    print(f"  Docking Score:  {info['docking_score']} kcal/mol")
    print(f"  Fingerprint Keys: {info['num_fingerprint_keys']}")
    print(f"  Critical Residues: {info['num_critical_residues']}")
    print(f"  Critical List: {info['critical_residues']}")
    print(f"\n  All fingerprint keys:")
    for key in info['fingerprint_keys']:
        fp_entry = analyzer.reference_fingerprint[key]
        print(f"    {key:<30s} | count={fp_entry['count']}")

    # Test 2: If the docked pose exists, re-analyze it
    docked_pose = os.path.join(PROJECT_ROOT, "data", "reference", "haloperidol_docked_pose.pdb")
    if os.path.exists(docked_pose):
        print(f"\n  Re-analyzing Haloperidol docked pose for validation...")
        # Convert PDB back to PDBQT for testing (or use the PDB directly)
        # For this self-test, we'll use the complex PDB directly
        complex_pdb = os.path.join(PROJECT_ROOT, "data", "reference", "haloperidol_drd2_complex.pdb")
        if os.path.exists(complex_pdb):
            interactions = plip_analyze_pose(complex_pdb)
            result = analyzer.score_interactions(interactions)
            print(f"\n  Self-comparison results (should be ~1.0):")
            print(f"    Overlap Score:    {result['overlap_score']}")
            print(f"    Critical Overlap: {result['critical_overlap']}")
            print(f"    Matched Keys:     {len(result['matched_keys'])}/{result['total_reference']}")
            print(f"    Missed Keys:      {result['missed_keys']}")
            print(f"    Novel Keys:       {result['novel_keys']}")

    print(f"\n{'=' * 60}")
    print("  Self-test complete!")
    print(f"{'=' * 60}")
