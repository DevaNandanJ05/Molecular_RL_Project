import os
import subprocess
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, AllChem, rdMolDescriptors

class RewardOracle:
    def __init__(self, receptor_pdbqt_path=None, vina_executable="vina"):
        """
        Oracle to compute rewards for generated SMILES.
        Combines RDKit metrics (QED, Validity, Complexity) with AutoDock Vina docking scores.
        """
        self.receptor_path = receptor_pdbqt_path
        self.vina_executable = vina_executable

    def evaluate_smiles(self, smiles: str, run_docking: bool = False) -> dict:
        """
        Evaluates a single SMILES string.
        """
        mol = Chem.MolFromSmiles(smiles)
        
        # 1. Validity Check
        if mol is None:
            return {"valid": False, "qed": 0.0, "docking_score": 0.0, "total_reward": -5.0}
        
        # 2. Strict sanitization check (prevents impossible chemistry)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return {"valid": False, "qed": 0.0, "docking_score": 0.0, "total_reward": -5.0}

        # 3. Prevent empty string cheat
        if len(smiles.strip()) < 3:
            return {"valid": False, "qed": 0.0, "docking_score": 0.0, "total_reward": -5.0}
            
        qed_score = QED.qed(mol)
        mw = Descriptors.MolWt(mol)
        
        # --- NEW: STRUCTURAL COMPLEXITY CHECKS ---
        penalty = 0.0
        
        # Prevent extreme sizes
        if mw < 150.0 or mw > 500.0:
            penalty += 2.0
            
        # Penalize linear chains (force it to generate drug-like rings)
        ring_count = rdMolDescriptors.CalcNumRings(mol)
        if ring_count == 0:
            penalty += 1.5
            
        # Penalize excessive rotatable bonds (floppy molecules are bad drugs)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if rot_bonds > 10:
            penalty += 1.0

        # 5. Simple Mock/Quick Reward (QED-based for fast loop testing)
        if not run_docking or self.receptor_path is None or not os.path.exists(self.receptor_path):
            total_reward = (qed_score * 3.0) - penalty
            return {
                "valid": True,
                "qed": qed_score,
                "mw": mw,
                "docking_score": None,
                "total_reward": total_reward
            }

        # 6. AutoDock Vina Docking Evaluation (Stage 2)
        docking_score = self._run_vina_docking(mol)
        
        # Combine inverted docking score with QED and structural penalties
        total_reward = (-docking_score if docking_score is not None else -2.0) + (qed_score * 1.5) - penalty

        return {
            "valid": True,
            "qed": qed_score,
            "mw": mw,
            "docking_score": docking_score,
            "total_reward": total_reward
        }

    def _run_vina_docking(self, mol) -> float:
        try:
            mol_3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) != 0:
                return 0.0  # Failed 3D embedding
            
            AllChem.UFFOptimizeMolecule(mol_3d)

            temp_sdf = "temp_ligand.sdf"
            temp_pdbqt = "temp_ligand.pdbqt"
            
            writer = Chem.SDWriter(temp_sdf)
            writer.write(mol_3d)
            writer.close()

            babel_cmd = ["obabel", temp_sdf, "-O", temp_pdbqt, "-h"]
            subprocess.run(babel_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            vina_cmd = [
                self.vina_executable,
                "--receptor", self.receptor_path,
                "--ligand", temp_pdbqt,
                "--center_x", "9.5", "--center_y", "5.2", "--center_z", "-11.4",
                "--size_x", "20.0", "--size_y", "20.0", "--size_z", "20.0",
                "--exhaustiveness", "4" 
            ]

            result = subprocess.run(vina_cmd, capture_output=True, text=True)

            for line in result.stdout.split('\n'):
                if "   1 " in line:
                    parts = line.split()
                    return float(parts[1])

        except Exception as e:
            return None
        finally:
            for f in ["temp_ligand.sdf", "temp_ligand.pdbqt"]:
                if os.path.exists(f):
                    os.remove(f)
        return None