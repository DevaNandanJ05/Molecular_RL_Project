import os
import sys
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from transformers import AutoTokenizer

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

# Suppress RDKit warning floods
RDLogger.DisableLog('rdApp.*')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.policy_network import PretrainedSMILESGenerator
from models.reward_oracle import RewardOracle

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MolGenEnv(gym.Env):
    """
    Standard Gymnasium Environment for Token-by-Token SMILES Generation.
    Maintains the token sequence while delegating generation physics to Vina.
    """
    def __init__(self, max_length=50):
        super(MolGenEnv, self).__init__()
        self.max_length = max_length
        
        # Load ONLY the tokenizer — not the full 500MB causal LM model.
        # The env never runs inference; it only needs token IDs for decoding.
        self.tokenizer = AutoTokenizer.from_pretrained("msb-roshan/molgpt")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
        
        self.vocab_size = len(self.tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        self.action_space = spaces.Discrete(self.vocab_size)
        # SB3 automatically casts Box spaces to float32 internally; we use int32 for token arrays
        self.observation_space = spaces.Box(low=0, high=self.vocab_size, shape=(self.max_length,), dtype=np.int32)
        
        target_path = os.path.join(project_root, "data", "raw", "drd2_clean.pdbqt")
        self.oracle = RewardOracle(receptor_pdbqt_path=target_path)
        
        self.seq = np.full((self.max_length,), self.pad_token_id, dtype=np.int32)
        self.step_idx = 1
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seq = np.full((self.max_length,), self.pad_token_id, dtype=np.int32)
        self.seq[0] = self.bos_token_id
        self.step_idx = 1
        return self.seq.copy(), {}

    def step(self, action):
        self.seq[self.step_idx] = int(action)
        self.step_idx += 1
        
        terminated = False
        reward = 0.0
        
        # Episode ends if EOS is generated or max length is reached
        if action == self.eos_token_id or self.step_idx >= self.max_length:
            terminated = True
            
            token_list = self.seq[:self.step_idx].tolist()
            raw_smiles = self.tokenizer.decode(token_list, skip_special_tokens=True).strip()
            cleaned_smiles = raw_smiles.replace(" ", "").split(".")[0]
            
            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol is not None:
                valid_smiles = Chem.MolToSmiles(mol)
                res = self.oracle.evaluate_smiles(valid_smiles, run_docking=True)
                reward = res["total_reward"]
                
                # Internal diversity penalty is calculated within RewardOracle against recent candidates
                pass
            else:
                reward = -5.0  # Syntax penalty
                
        return self.seq.copy(), float(reward), terminated, False, {}


class MolGPTExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor that mounts MolGPT as the PPO backbone.
    Outputs the transformer's hidden states (768-dim), NOT the full vocab
    logits (30002-dim). SB3's action head then projects 768 -> 30002,
    which is ~40x smaller and fits in 4GB VRAM.
    """
    def __init__(self, observation_space: gym.spaces.Box):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = PretrainedSMILESGenerator(device=device)
        
        # Use hidden dimension (768) instead of vocab_size (30002) as features_dim.
        # This makes SB3 create Linear(768, 30002) instead of Linear(30002, 30002).
        hidden_size = generator.model.config.n_embd
        
        super().__init__(observation_space, features_dim=hidden_size)
        self.generator = generator
        self.pad_token_id = self.generator.tokenizer.pad_token_id if self.generator.tokenizer.pad_token_id is not None else 0
        
        # Freeze MolGPT to prevent 4GB VRAM OOM during local laptop testing.
        # PPO's action head (initialized from lm_head) will be the only trainable part.
        for param in self.generator.model.parameters():
            param.requires_grad = False

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs_long = observations.long().to(self.generator.model.device)
        
        # Run through transformer blocks ONLY (skip lm_head projection).
        # The lm_head weights are copied into SB3's action_net instead.
        with torch.no_grad():
            transformer_outputs = self.generator.model.transformer(input_ids=obs_long)
        hidden_states = transformer_outputs[0]  # (batch, seq_len, 768)
        
        # Extract hidden state at the current generation step
        pad_mask = (obs_long == self.pad_token_id)
        lengths = pad_mask.float().argmax(dim=1)
        no_pad = (~pad_mask).all(dim=1)
        lengths[no_pad] = obs_long.shape[1]
        lengths = torch.clamp(lengths, min=1)
        
        batch_indices = torch.arange(obs_long.shape[0], device=obs_long.device)
        step_hidden = hidden_states[batch_indices, lengths - 1, :]
        
        return step_hidden.float()


def train():
    print("--- PPO Training started ---")
    
    # DummyVecEnv runs all envs in a single process to avoid the 3GB-per-process
    # memory multiplication that SubprocVecEnv causes on 16GB laptops.
    # Vina docking already runs as subprocesses, so parallelism is preserved.
    num_envs = 2
    env = make_vec_env(
        MolGenEnv, 
        n_envs=num_envs, 
        env_kwargs={"max_length": 50},
        vec_env_cls=DummyVecEnv 
    )
    
    # Construct the PPO policy to read from the frozen MolGPT extractor
    policy_kwargs = dict(
        features_extractor_class=MolGPTExtractor,
        features_extractor_kwargs={},
        net_arch=dict(pi=[], vf=[128, 128])
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-5,
        n_steps=64, # Local batch size calculation (64 * 2 envs = 128 buffer)
        batch_size=16,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.05,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # --- LM Head Initialization: Preserve MolGPT's pretrained logits ---
    # The extractor now outputs hidden states (768-dim), and SB3 created
    # action_net = Linear(768, 30002). We initialize it with MolGPT's own
    # lm_head weights so that on Episode 1, the output logits are identical
    # to MolGPT's, immediately restoring high chemical validity.
    action_net = model.policy.action_net
    lm_head = model.policy.features_extractor.generator.model.lm_head
    if isinstance(action_net, torch.nn.Linear):
        with torch.no_grad():
            action_net.weight.copy_(lm_head.weight)
            action_net.bias.zero_()  # GPT-2 lm_head has no bias
        print(f"[Init] Action head initialized from MolGPT lm_head: {action_net.weight.shape}")
    else:
        print(f"[Warning] action_net is {type(action_net)}, skipping lm_head init")
    
    # 200 episodes total * 50 max steps = 10,000 timesteps[cite: 5]
    total_timesteps = 10000 
    
    model.learn(total_timesteps=total_timesteps)
    
    save_path = "checkpoints/ppo_smiles_generator.zip"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"PPO Model saved to {save_path}")

if __name__ == "__main__":
    train()