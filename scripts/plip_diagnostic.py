"""
Diagnostic script to inspect PLIP interaction object attributes.
This will reveal the exact field names for ligand atom indices.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Apply the Windows InChIKey patch
try:
    import openbabel.pybel as pybel
    _orig = pybel.Molecule.write
    def _patched(self, format='smi', filename=None, overwrite=False, opt=None):
        if format == 'inchikey': return "N/A"
        if opt is None: opt = {}
        return _orig(self, format=format, filename=filename, overwrite=overwrite, opt=opt)
    pybel.Molecule.write = _patched
except ImportError:
    pass

from plip.structure.preparation import PDBComplex

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
complex_pdb = os.path.join(PROJECT_ROOT, "data", "reference", "haloperidol_drd2_complex.pdb")

mol = PDBComplex()
mol.load_pdb(complex_pdb)

for ligand in mol.ligands:
    mol.characterize_complex(ligand)

for bsite_key, bs in mol.interaction_sets.items():
    print(f"=== Binding Site: {bsite_key} ===\n")

    # Inspect H-bonds
    if bs.hbonds_pdon:
        hb = bs.hbonds_pdon[0]
        print("--- H-Bond (protein donor) attributes ---")
        for attr in sorted(dir(hb)):
            if not attr.startswith('_'):
                try:
                    val = getattr(hb, attr)
                    if not callable(val):
                        print(f"  {attr:30s} = {val}")
                except:
                    print(f"  {attr:30s} = <error>")
        print()

    if bs.hbonds_ldon:
        hb = bs.hbonds_ldon[0]
        print("--- H-Bond (ligand donor) attributes ---")
        for attr in sorted(dir(hb)):
            if not attr.startswith('_'):
                try:
                    val = getattr(hb, attr)
                    if not callable(val):
                        print(f"  {attr:30s} = {val}")
                except:
                    print(f"  {attr:30s} = <error>")
        print()

    # Inspect hydrophobic
    if bs.hydrophobic_contacts:
        hp = bs.hydrophobic_contacts[0]
        print("--- Hydrophobic Contact attributes ---")
        for attr in sorted(dir(hp)):
            if not attr.startswith('_'):
                try:
                    val = getattr(hp, attr)
                    if not callable(val):
                        print(f"  {attr:30s} = {val}")
                except:
                    print(f"  {attr:30s} = <error>")
        print()

    # Inspect pi-stacking
    if bs.pistacking:
        ps = bs.pistacking[0]
        print("--- Pi-Stacking attributes ---")
        for attr in sorted(dir(ps)):
            if not attr.startswith('_'):
                try:
                    val = getattr(ps, attr)
                    if not callable(val):
                        print(f"  {attr:30s} = {val}")
                except:
                    print(f"  {attr:30s} = <error>")
        print()

    # Salt bridges
    sbs = list(bs.saltbridge_pneg if hasattr(bs, 'saltbridge_pneg') else [])
    sbs += list(bs.saltbridge_lneg if hasattr(bs, 'saltbridge_lneg') else [])
    if sbs:
        sb = sbs[0]
        print("--- Salt Bridge attributes ---")
        for attr in sorted(dir(sb)):
            if not attr.startswith('_'):
                try:
                    val = getattr(sb, attr)
                    if not callable(val):
                        print(f"  {attr:30s} = {val}")
                except:
                    print(f"  {attr:30s} = <error>")
        print()

    break  # Only inspect first binding site
