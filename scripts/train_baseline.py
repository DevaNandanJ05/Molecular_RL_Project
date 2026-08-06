import os
import sys
import torch
import torch.optim as optim
from torch.distributions import Categorical

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.policy_network import PretrainedSMILESGenerator
from models.reward_oracle import RewardOracle

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training started on: {device} ---")

    # 1. Load the Pretrained Hugging Face Model
    generator = PretrainedSMILESGenerator(device=device)
    model = generator.model
    tokenizer = generator.tokenizer
    
    # Use a very small learning rate since the model is already pretrained
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # 2. Initialize Reward Oracle (Docking is False for rapid testing)
    oracle = RewardOracle()

    num_episodes = 200
    max_length = 50  # Max tokens to generate

    for episode in range(1, num_episodes + 1):
        model.train()
        
        # Start the sequence with the Beginning-Of-Sequence token
        current_token = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
        past_key_values = None
        
        log_probs = []
        sampled_indices = []

        # Generate a SMILES string token-by-token
        for step in range(max_length):
            logits, past_key_values = generator(current_token, past_key_values)
            
            # Extract the logits for the very last token in the sequence
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample an action
            m = Categorical(probs)
            action = m.sample()
            
            log_probs.append(m.log_prob(action))
            sampled_indices.append(action.item())
            
            # The generated token becomes the input for the next time step
            current_token = action.unsqueeze(0)
            
            # Stop if the model decides the molecule is finished
            if action.item() == tokenizer.eos_token_id:
                break

        # Decode the generated tokens into a SMILES string
        # skip_special_tokens removes the <eos> and <bos> tags so RDKit can read it
        # Decode, remove spaces, and take ONLY the first molecule if it generates a mixture
        raw_smiles = tokenizer.decode(sampled_indices, skip_special_tokens=True).strip()
        generated_smiles = raw_smiles.replace(" ", "").split(".")[0]
        
        # Get Reward from Oracle
        eval_result = oracle.evaluate_smiles(generated_smiles, run_docking=False)
        reward = eval_result["total_reward"]
        
        # Policy Gradient Update (REINFORCE)
        loss = sum([-lp * reward for lp in log_probs])
        
        optimizer.zero_grad()
        loss.backward()
        
        # Prevent exploding gradients which are common in Transformer RL
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Print output every 10 episodes to monitor learning
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode:03d} | Reward: {reward:+.3f} | Valid: {eval_result['valid']} | SMILES: {generated_smiles}")

    print("--- Baseline Training Complete ---")

# Start the training loop!
if __name__ == "__main__":
    train()