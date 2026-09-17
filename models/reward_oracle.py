import os
import sys
import shutil
import subprocess
import uuid
from collections import deque
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, AllChem, rdMolDescriptors


def resolve_vina_path(custom_path=None) -> str:
    """
    Locates the AutoDock Vina executable on Windows / Linux across:
      1. Explicit argument or VINA_PATH environment variable
      2. Project local bin/ folder (<project_root>/bin/vina.exe)
      3. Active virtualenv / conda environment (Scripts/vina.exe or Library/bin/vina.exe)
      4. System PATH
      5. Standard Windows installation directories
    """
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    if "VINA_PATH" in os.environ:
        candidates.append(os.environ["VINA_PATH"])

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.append(os.path.join(project_root, "bin", "vina.exe"))
    candidates.append(os.path.join(project_root, "bin", "vina"))

    # Active Python environment
    candidates.append(os.path.join(sys.prefix, "Scripts", "vina.exe"))
    candidates.append(os.path.join(sys.prefix, "Library", "bin", "vina.exe"))
    candidates.append(os.path.join(sys.prefix, "bin", "vina"))

    # System PATH
    which_vina = shutil.which("vina") or shutil.which("vina.exe")
    if which_vina:
        candidates.append(which_vina)

    # Standard Windows install locations
    candidates.extend([
        r"C:\AutoDockVina\vina.exe",
        r"C:\Program Files\AutoDockVina\vina.exe",
        r"C:\Tools\vina.exe",
    ])

    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)

    return custom_path or r"C:\AutoDockVina\vina.exe"


def resolve_obabel_path(custom_path=None) -> str:
    """
    Locates OpenBabel executable on Windows / Linux across:
      1. Explicit argument or OBABEL_PATH environment variable
      2. Project local bin/ folder (<project_root>/bin/obabel.exe)
      3. Active virtualenv (pip install openbabel-wheel -> site-packages/openbabel/bin/obabel.exe)
      4. Active conda environment (Library/bin/obabel.exe)
      5. System PATH
      6. Standard Windows installation directories
    """
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    if "OBABEL_PATH" in os.environ:
        candidates.append(os.environ["OBABEL_PATH"])

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.append(os.path.join(project_root, "bin", "obabel.exe"))
    candidates.append(os.path.join(project_root, "bin", "obabel"))

    # Check site-packages from openbabel-wheel (isolated in virtualenv!)
    try:
        import openbabel
        ob_pkg_dir = os.path.dirname(openbabel.__file__)
        candidates.append(os.path.join(ob_pkg_dir, "bin", "obabel.exe"))
        candidates.append(os.path.join(ob_pkg_dir, "obabel.exe"))
    except ImportError:
        pass

    # Active Python environment
    candidates.append(os.path.join(sys.prefix, "Library", "bin", "obabel.exe"))
    candidates.append(os.path.join(sys.prefix, "Scripts", "obabel.exe"))
    candidates.append(os.path.join(sys.prefix, "bin", "obabel"))

    # System PATH
    which_ob = shutil.which("obabel") or shutil.which("obabel.exe")
    if which_ob:
        candidates.append(which_ob)

    # Standard Windows install locations
    candidates.extend([
        r"C:\Program Files\OpenBabel-2.4.1\obabel.exe",
        r"C:\Program Files\OpenBabel-3.1.1\obabel.exe",
        r"C:\OpenBabel-2.4.1\obabel.exe",
        r"C:\OpenBabel-3.1.1\obabel.exe",
    ])

    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)

    return custom_path or r"C:\Program Files\OpenBabel-2.4.1\obabel.exe"


class RewardOracle:
    def __init__(
        self, 
        receptor_pdbqt_path="data/raw/drd2_clean.pdbqt", 
        vina_executable=None,
        exhaustiveness=4,
        obabel_path=None,
        cpu=1,
        similarity_threshold=0.75,
        diversity_penalty=3.0,
        recent_buffer_size=100
    ):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if receptor_pdbqt_path and not os.path.isabs(receptor_pdbqt_path):
            self.receptor_path = os.path.abspath(os.path.join(project_root, receptor_pdbqt_path))
        else:
            self.receptor_path = receptor_pdbqt_path

        self.vina_executable = resolve_vina_path(vina_executable)
        self.obabel_path = resolve_obabel_path(obabel_path)
        self.exhaustiveness = exhaustiveness
        self.cpu = max(int(cpu), 1)

        # Internal Tanimoto diversity tracking against recent candidates
        self.similarity_threshold = float(similarity_threshold)
        self.diversity_penalty = float(diversity_penalty)
        self.recent_buffer_size = int(recent_buffer_size)
        self.recent_candidates = deque(maxlen=self.recent_buffer_size)

    def reset_diversity_buffer(self):
        """Clears the internal recent candidates memory buffer."""
        self.recent_candidates.clear()

    def evaluate_smiles(self, smiles: str, run_docking: bool = False) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return {
                "valid": False, "qed": 0.0, "docking_score": 0.0, "total_reward": -5.0,
                "max_tanimoto": 0.0, "diversity_penalty": 0.0
            }
        
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return {
                "valid": False, "qed": 0.0, "docking_score": 0.0, "total_reward": -5.0,
                "max_tanimoto": 0.0, "diversity_penalty": 0.0
            }

        if len(smiles.strip()) < 3:
            return {
                "valid": False, "qed": 0.0, "docking_score": 0.0, "total_reward": -5.0,
                "max_tanimoto": 0.0, "diversity_penalty": 0.0
            }
            
        qed_score = QED.qed(mol)
        mw = Descriptors.MolWt(mol)
        
        # Structural Complexity Penalties
        penalty = 0.0
        if mw < 150.0 or mw > 500.0:
            penalty += 2.0
            
        ring_count = rdMolDescriptors.CalcNumRings(mol)
        if ring_count == 0:
            penalty += 1.5
            
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if rot_bonds > 10:
            penalty += 1.0

        # Internal Tanimoto Similarity Diversity Check against Recent Candidates
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        max_sim = 0.0
        diversity_penalty_applied = 0.0
        if len(self.recent_candidates) > 0:
            sims = DataStructs.BulkTanimotoSimilarity(fp, list(self.recent_candidates))
            max_sim = float(max(sims))
            if max_sim > self.similarity_threshold:
                diversity_penalty_applied = self.diversity_penalty
                penalty += diversity_penalty_applied

        # Register current candidate fingerprint into recent buffer
        self.recent_candidates.append(fp)

        if not run_docking or self.receptor_path is None or not os.path.exists(self.receptor_path):
            if run_docking:
                print(f"[Oracle Warning] Receptor not found at {self.receptor_path}. Falling back to 2D.")
            total_reward = (qed_score * 3.0) - penalty
            return {
                "valid": True, "qed": qed_score, "mw": mw, 
                "docking_score": None, "total_reward": total_reward,
                "max_tanimoto": max_sim, "diversity_penalty": diversity_penalty_applied
            }

        # 3D Docking Evaluation
        docking_score = self._run_vina_docking(mol)
        
        # Combine inverted docking score with QED and structural penalties
        # A successful pose yields a negative score (e.g., -9.5), inverted here to +9.5 for RL maximization
        if docking_score is not None and docking_score < 0:
            total_reward = abs(docking_score) + (qed_score * 1.5) - penalty
        else:
            total_reward = -2.0 + (qed_score * 1.5) - penalty

        return {
            "valid": True, "qed": qed_score, "mw": mw,
            "docking_score": docking_score, "total_reward": total_reward,
            "max_tanimoto": max_sim, "diversity_penalty": diversity_penalty_applied
        }

    def _run_vina_docking(self, mol) -> float:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        temp_dir = os.path.join(project_root, "data", "temp")
        os.makedirs(temp_dir, exist_ok=True)

        thread_id = uuid.uuid4().hex
        temp_sdf = os.path.join(temp_dir, f"temp_{thread_id}.sdf")
        temp_pdbqt = os.path.join(temp_dir, f"temp_{thread_id}.pdbqt")
        temp_out = os.path.join(temp_dir, f"temp_{thread_id}_out.pdbqt")
        temp_log = os.path.join(temp_dir, f"temp_{thread_id}_log.txt")
        
        try:
            mol_3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) != 0:
                return 0.0 
            
            AllChem.UFFOptimizeMolecule(mol_3d)

            writer = Chem.SDWriter(temp_sdf)
            writer.write(mol_3d)
            writer.close()

            obabel_path = self.obabel_path
            
            # Set BABEL_DATADIR automatically so OpenBabel finds its internal tables
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

            babel_cmd = [obabel_path, temp_sdf, "-O", temp_pdbqt, "-h"]
            babel_res = subprocess.run(babel_cmd, capture_output=True, text=True, env=env)
            
            # Ensure PDBQT was created and is not empty
            if babel_res.returncode != 0 or not os.path.exists(temp_pdbqt) or os.path.getsize(temp_pdbqt) == 0:
                return 0.0

            vina_cmd = [
                self.vina_executable,
                "--receptor", self.receptor_path,
                "--ligand", temp_pdbqt,
                "--out", temp_out,
                "--center_x", "9.5", "--center_y", "5.2", "--center_z", "-11.4",
                "--size_x", "20.0", "--size_y", "20.0", "--size_z", "20.0",
                "--exhaustiveness", str(self.exhaustiveness),
                "--cpu", str(self.cpu)
            ]

            result = subprocess.run(vina_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                err_msg = result.stderr.strip() or result.stdout.strip()
                if "WARNING" not in err_msg:
                    print(f"[Oracle Error] Docking Failed: {err_msg}")
                return 0.0

            for line in result.stdout.split('\n'):
                if "   1 " in line:
                    parts = line.split()
                    return float(parts[1])

        except Exception as e:
            import traceback
            print(f"[CRITICAL ORACLE EXCEPTION]: {str(e)}")
            traceback.print_exc()
            return 0.0
        finally:
            for f in [temp_sdf, temp_pdbqt, temp_out, temp_log]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        return 0.0