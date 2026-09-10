import os
import sys
import torch
import torch.optim as optim
from torch.distributions import Categorical
import concurrent.futures

# RDKit imports for fingerprinting, canonicalization, and logging suppression
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

# Suppress RDKit warning floods (SMILES Parse Error, Kekulize, etc.)
RDLogger.DisableLog('rdApp.*')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.policy_network import PretrainedSMILESGenerator
from models.reward_oracle import RewardOracle

# Top-level function for multiprocessing to avoid pickling issues
def evaluate_worker(smiles):
    # Absolute path to your actual DRD2 receptor file
    target_path = os.path.join(project_root, "data", "raw", "drd2_clean.pdbqt")
    raw_dir = os.path.join(project_root, "data", "raw")
    #print(f"Files actually in directory: {os.listdir(raw_dir)}")
    oracle = RewardOracle(receptor_pdbqt_path=target_path)
    return oracle.evaluate_smiles(smiles, run_docking=True)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training started on: {device} ---")

    generator = PretrainedSMILESGenerator(device=device)
    model = generator.model
    tokenizer = generator.tokenizer
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    num_episodes = 200
    max_length = 50
    batch_size = 16
    baseline_reward = 0.0    
    ema_alpha = 0.05          
    
    # --- UPGRADED: GLOBAL MEMORY BUFFER ---
    # Tracks structural fingerprints of valid generated molecules
    global_seen_fps = []

    for episode in range(1, num_episodes + 1):
        model.train()
        
        current_token = torch.tensor([[tokenizer.bos_token_id]] * batch_size, dtype=torch.long).to(device)
        past_key_values = None
        
        batch_log_probs = []
        batch_sampled_indices = [[] for _ in range(batch_size)]
        is_finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

        # GENERATION
        with torch.amp.autocast('cuda'):
            for step in range(max_length):
                logits, past_key_values = generator(current_token, past_key_values)
                next_token_logits = logits[:, -1, :] 
                probs = torch.softmax(next_token_logits, dim=-1)
                
                m = Categorical(probs)
                action = m.sample()
                
                step_log_probs = m.log_prob(action)
                step_log_probs = step_log_probs * (~is_finished).float()
                batch_log_probs.append(step_log_probs)
                
                for i in range(batch_size):
                    if not is_finished[i].item():
                        batch_sampled_indices[i].append(action[i].item())
                
                is_finished = is_finished | (action == tokenizer.eos_token_id)
                if is_finished.all(): break
                current_token = action.unsqueeze(-1) 

        # --- UPGRADED: DECODING EFFICIENCY ---
        generated_smiles_list = []
        for i in range(batch_size):
            raw_smiles = tokenizer.decode(batch_sampled_indices[i], skip_special_tokens=True).strip()
            cleaned_smiles = raw_smiles.replace(" ", "").split(".")[0]
            
            # Use RDKit to ensure chemical validity and canonicalize before evaluation
            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol is not None:
                generated_smiles_list.append(Chem.MolToSmiles(mol))
            else:
                generated_smiles_list.append(cleaned_smiles)

        # MULTITHREADED EVALUATION
        batch_rewards = []
        valid_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(evaluate_worker, generated_smiles_list))
            
        for res in results:
            batch_rewards.append(res["total_reward"])
            if res["valid"]:
                valid_count += 1
                
        # --- UPGRADED: STRUCTURAL DIVERSITY PENALTY LOGIC ---
        batch_seen_fps = []
        
        for i, smiles in enumerate(generated_smiles_list):
            if not results[i]["valid"]:
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            
            # Penalty 1: Intra-batch structural similarity
            batch_max_sim = 0.0
            if batch_seen_fps:
                batch_sims = DataStructs.BulkTanimotoSimilarity(fp, batch_seen_fps)
                batch_max_sim = max(batch_sims)
                
            if batch_max_sim > 0.75:
                batch_rewards[i] -= 2.0  
                continue 
            else:
                batch_seen_fps.append(fp)
                
            # Penalty 2: Global historical structural similarity
            global_max_sim = 0.0
            if global_seen_fps:
                global_sims = DataStructs.BulkTanimotoSimilarity(fp, global_seen_fps)
                global_max_sim = max(global_sims)
                
            if global_max_sim > 0.75:
                batch_rewards[i] -= 3.0  # Massive penalty to force exploration
            else:
                global_seen_fps.append(fp)
        # ----------------------------------------------------
                
        sample_smiles = generated_smiles_list[0]
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).to(device)

        # BASELINE AND ADVANTAGES
        batch_mean_reward = rewards_tensor.mean().item()
        
        if episode == 1:
            baseline_reward = batch_mean_reward
        else:
            baseline_reward = (1 - ema_alpha) * baseline_reward + (ema_alpha * batch_mean_reward)
            
        advantages = rewards_tensor - baseline_reward

        # POLICY GRADIENT UPDATE
        with torch.amp.autocast('cuda'):
            stacked_log_probs = torch.stack(batch_log_probs)
            sum_log_probs = stacked_log_probs.sum(dim=0)
            loss = -(advantages.detach() * sum_log_probs).mean()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Print statement updated to track fingerprint count instead of raw strings
        print(f"Ep {episode:03d} | Mean Reward: {batch_mean_reward:+.3f} | Baseline: {baseline_reward:+.3f} | "
              f"Valid: {valid_count}/{batch_size} | Unique Historically: {len(global_seen_fps)} | Sample: {sample_smiles}")

    print("--- Training Complete ---")
    save_path = "checkpoints/smiles_generator_optimized.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()