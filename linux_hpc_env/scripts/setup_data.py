import os
import urllib.request
from rdkit import Chem

def download_drd2_data():
    # Dynamically find the project root folder (one level above this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Download DRD2 Receptor (PDB ID: 6LUQ)
    pdb_id = "6luq"
    pdb_url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    receptor_path = os.path.join(raw_dir, f"{pdb_id}.pdb")

    if not os.path.exists(receptor_path):
        print(f"Downloading DRD2 receptor ({pdb_id})...")
        try:
            # Set a User-Agent header so RCSB doesn't block the Python request
            req = urllib.request.Request(pdb_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(receptor_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Saved receptor to: {receptor_path}")
        except Exception as e:
            print(f"Error downloading PDB file: {e}")
            return
    else:
        print(f"Receptor file already exists at: {receptor_path}")

    # 2. Reference SMILES for Haloperidol
    haloperidol_smiles = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c3ccc(F)cc3"
    mol = Chem.MolFromSmiles(haloperidol_smiles)
    if mol:
        print(f"Haloperidol SMILES loaded successfully. Heavy atoms: {mol.GetNumAtoms()}")
    else:
        print("Failed to parse Haloperidol SMILES.")

if __name__ == "__main__":
    download_drd2_data()