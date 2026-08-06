import os
import sys
import torch
import torch.optim as optim
from torch.distributions import Categorical

# Add the parent directory to sys.path so we can import our modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.policy_network import RNNSmilesGenerator, SmilesTokenizer
from models.reward_oracle import RewardOracle

def train():
    # Setup Device (CUDA if available for RTX 3040, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training started on: {device} ---")

    # Initialize Modules
    tokenizer = SmilesTokenizer()
    model = RNNSmilesGenerator(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Initialize Oracle (using mock QED reward for now, no docking)
    oracle = RewardOracle()

    num_episodes = 500
    max_length = 40  # Keep sequences short for fast laptop prototyping

    for episode in range(1, num_episodes + 1):
        model.train()
        hidden = model.init_hidden(batch_size=1, device=device)
        
        # Start with the <sos> token
        current_token = torch.tensor([[tokenizer.sos_idx]], dtype=torch.long).to(device)
        
        log_probs = []
        sampled_indices = []

        # Generate a SMILES string token-by-token
        for step in range(max_length):
            logits, hidden = model(current_token, hidden)
            probs = torch.softmax(logits, dim=-1)
            
            # Sample an action (token) from the probability distribution
            m = Categorical(probs)
            action = m.sample()
            
            log_probs.append(m.log_prob(action))
            sampled_indices.append(action.item())
            
            # Set the generated token as the input for the next step
            current_token = action.unsqueeze(0)
            
            if action.item() == tokenizer.eos_idx:
                break

        # Decode the generated sequence into a SMILES string
        generated_smiles = tokenizer.decode(sampled_indices)
        
        # Get Reward from Oracle
        eval_result = oracle.evaluate_smiles(generated_smiles, run_docking=False)
        reward = eval_result["total_reward"]
        
        # Policy Gradient Update (REINFORCE)
        # Loss = -Sum(log_probability * reward)
        loss = sum([-lp * reward for lp in log_probs])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress every 50 episodes
        if episode % 50 == 0 or episode == 1:
            print(f"Episode {episode:03d} | Reward: {reward:+.3f} | Valid: {eval_result['valid']} | SMILES: {generated_smiles}")

    print("--- Baseline Training Complete ---")

if __name__ == "__main__":
    train()