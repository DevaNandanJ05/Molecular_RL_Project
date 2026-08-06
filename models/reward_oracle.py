'''import os
import subprocess
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, AllChem

class RewardOracle:
    def __init__(self, receptor_pdbqt_path=None, vina_executable="vina"):
        """
        Oracle to compute rewards for generated SMILES.
        Combines RDKit metrics (QED, Validity) with AutoDock Vina docking scores.
        """
        self.receptor_path = receptor_pdbqt_path
        self.vina_executable = vina_executable

    def evaluate_smiles(self, smiles: str, run_docking: bool = False) -> dict:
        """
        Evaluates a single SMILES string.
        """
        # 0. Prevent the empty string cheat
        if len(smiles.strip()) < 3:
            return {
                "valid": False, "qed": 0.0, "docking_score": 0.0, 
                "total_reward": -5.0 # Penalty for being lazy
            }

        mol = Chem.MolFromSmiles(smiles)
        
        # 1. Validity Check
        if mol is None:
            return {
                "valid": False, "qed": 0.0, "docking_score": 0.0,
                "total_reward": -5.0  # Penalty for invalid SMILES syntax
            }
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return {"valid": False, "qed": 0.0, "docking_score": 0.0, "total_reward": -5.0}

        if len(smiles.strip()) < 3:
            return {"valid": False, "qed": 0.0, "docking_score": 0.0, "total_reward": -5.0}
            
        qed_score = QED.qed(mol)
        mw = Descriptors.MolWt(mol)
        
        if mw < 30.0 or mw > 500.0:
            return {"valid": False, "qed": 0.0, "docking_score": 0.0, "total_reward": -2.0}

        return {
            "valid": True,
            "qed": qed_score,
            "mw": mw,
            "docking_score": None,
            "total_reward": qed_score * 2.0
        }
        # 2. Prevent the "NNNNNN" continuous chain cheat (Must have at least some weight)
        mw = Descriptors.MolWt(mol)
        if mw < 50.0 or mw > 800.0:
            return {
                "valid": False, "qed": 0.0, "docking_score": 0.0,
                "total_reward": -2.0  # Penalty for unrealistic size
            }
            
        # 3. Basic Properties via RDKit
        qed_score = QED.qed(mol)
        
        # 4. Simple Mock/Quick Reward
        if not run_docking or self.receptor_path is None or not os.path.exists(self.receptor_path):
            return {
                "valid": True, "qed": qed_score, "mw": mw, "docking_score": None,
                "total_reward": qed_score * 2.0
            }

        # ... (keep the rest of the Vina docking code below this) ...

    def _run_vina_docking(self, mol) -> float:
        """
        Converts RDKit Mol -> 3D conformer -> SDF -> PDBQT -> Runs AutoDock Vina.
        """
        try:
            # Add hydrogens & Embed 3D
            mol_3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) != 0:
                return 0.0  # Failed 3D embedding (steric strain penalty)
            
            AllChem.UFFOptimizeMolecule(mol_3d)

            # Write temporary files for conversion
            temp_sdf = "temp_ligand.sdf"
            temp_pdbqt = "temp_ligand.pdbqt"
            
            writer = Chem.SDWriter(temp_sdf)
            writer.write(mol_3d)
            writer.close()

            # Convert SDF to PDBQT via OpenBabel CLI
            babel_cmd = ["obabel", temp_sdf, "-O", temp_pdbqt, "-h"]
            subprocess.run(babel_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Define DRD2 Pocket Center (PDB: 6LUQ approximate pocket center)
            # Adjust center_x/y/z based on your prepared pocket grid box
            vina_cmd = [
                self.vina_executable,
                "--receptor", self.receptor_path,
                "--ligand", temp_pdbqt,
                "--center_x", "9.5", "--center_y", "5.2", "--center_z", "-11.4",
                "--size_x", "20.0", "--size_y", "20.0", "--size_z", "20.0",
                "--exhaustiveness", "4"  # Kept low for 4GB VRAM / Laptop CPU loop speed
            ]

            result = subprocess.run(vina_cmd, capture_output=True, text=True)

            # Parse best binding pose score (kcal/mol)
            for line in result.stdout.split('\n'):
                if "   1 " in line:
                    parts = line.split()
                    return float(parts[1])

        except Exception as e:
            # If Vina or OpenBabel fails/is not configured yet
            return None
        finally:
            # Cleanup temp files
            for f in ["temp_ligand.sdf", "temp_ligand.pdbqt"]:
                if os.path.exists(f):
                    os.remove(f)

        return None


# Quick local test if run directly
if __name__ == "__main__":
    oracle = RewardOracle()
    sample_smiles = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c3ccc(F)cc3"  # Haloperidol
    res = oracle.evaluate_smiles(sample_smiles, run_docking=False)
    print("Haloperidol Baseline Evaluation:", res)'''

import os
import sys
import torch
import torch.optim as optim
from torch.distributions import Categorical

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.policy_network import RNNSmilesGenerator, SmilesTokenizer
from models.reward_oracle import RewardOracle

def pretrain_tokenizer_baseline(model, tokenizer, optimizer, device):
    """
    Supervised pretraining phase: Teach the network basic valid SMILES syntax 
    so it stops generating garbage strings like '======' or 'Nn)Nc'.
    """
    print("--- Starting Supervised Pretraining (Behavioral Cloning) ---")
    # A tiny seed library of known valid drug-like molecules
    seed_smiles = [
        "c1ccccc1",                 # Benzene ring
        "CCO",                      # Ethanol
        "CCN",                      # Ethylamine
        "CC(=O)O",                  # Acetic acid
        "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c3ccc(F)cc3" # Haloperidol
    ]
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    
    for epoch in range(200):
        total_loss = 0
        for smi in seed_smiles:
            encoded = tokenizer.encode(smi)
            inputs = torch.tensor([encoded[:-1]], dtype=torch.long).to(device)
            targets = torch.tensor([encoded[1:]], dtype=torch.long).to(device)
            
            hidden = model.init_hidden(batch_size=1, device=device)
            logits, _ = model(inputs, hidden) # logits shape: (seq_len, vocab_size)
            
            loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print("--- Pretraining Complete. Network now understands basic SMILES syntax. ---\n")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training started on: {device} ---")

    tokenizer = SmilesTokenizer()
    model = RNNSmilesGenerator(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # 1. Run Pretraining so the model starts with valid grammar
    pretrain_tokenizer_baseline(model, tokenizer, optimizer, device)

    oracle = RewardOracle()
    num_episodes = 300
    max_length = 40

    for episode in range(1, num_episodes + 1):
        model.train()
        hidden = model.init_hidden(batch_size=1, device=device)
        current_token = torch.tensor([[tokenizer.sos_idx]], dtype=torch.long).to(device)
        
        log_probs = []
        sampled_indices = []

        for step in range(max_length):
            logits, hidden = model(current_token, hidden)
            probs = torch.softmax(logits, dim=-1)
            
            m = Categorical(probs)
            action = m.sample()
            
            log_probs.append(m.log_prob(action))
            sampled_indices.append(action.item())
            current_token = action.unsqueeze(0)
            
            if action.item() == tokenizer.eos_idx:
                break

        generated_smiles = tokenizer.decode(sampled_indices)
        eval_result = oracle.evaluate_smiles(generated_smiles, run_docking=False)
        reward = eval_result["total_reward"]
        
        # REINFORCE loss
        loss = sum([-lp * reward for lp in log_probs])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 25 == 0 or episode == 1:
            print(f"Episode {episode:03d} | Reward: {reward:+.3f} | Valid: {eval_result['valid']} | SMILES: {generated_smiles}")

    print("--- Baseline Training Complete ---")

# Quick local test if run directly
if __name__ == "__main__":
    oracle = RewardOracle()
    sample_smiles = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c3ccc(F)cc3"  # Haloperidol
    res = oracle.evaluate_smiles(sample_smiles, run_docking=False)
    print("Haloperidol Baseline Evaluation:", res)