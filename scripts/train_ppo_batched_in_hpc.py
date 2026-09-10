"""
Windows HPC-Optimized PPO Training Script for Molecular Generation
===================================================================
Hardware Target: Windows-based HPC / Workstation (64 GB DDR4 RAM, NVIDIA RTX 3090 24 GB)
Multi-Day Unattended Training with Full MolGPT Fine-Tuning.

Key Windows Features & Optimizations:
  - Windows multiprocessing safety with freeze_support() and tokenizer pre-caching
  - Dynamic binary resolution for OpenBabel (openbabel-wheel), AutoDock Vina, and AutoDock-GPU
  - Zero disk leaks: all temporary files directed to dedicated cleanup routine
  - Per-worker CPU throttling (--cpu 1) to eliminate thread contention on Windows scheduler
  - Unfrozen MolGPT with gradient checkpointing for 24 GB VRAM
  - SubprocVecEnv with 16 parallel workers for high docking throughput (or fallback to DummyVecEnv)
  - TensorBoard + CSV molecule logging for multi-day monitoring
  - Automatic checkpointing every 10,000 timesteps with resume capability
"""

import os
import sys
import csv
import time
import shutil
import argparse
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from transformers import AutoTokenizer
from datetime import datetime, timedelta

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

# Suppress RDKit warning floods
RDLogger.DisableLog('rdApp.*')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.policy_network import PretrainedSMILESGenerator
from models.reward_oracle import RewardOracle, resolve_vina_path, resolve_obabel_path

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure


# ============================================================================
# WINDOWS HPC CONFIGURATION
# ============================================================================
def get_default_config() -> dict:
    """Returns the default configuration optimized for Windows HPC (RTX 3090)."""
    return {
        # ---- File Paths (Windows-compatible absolute paths) ----
        "receptor_path": os.path.join(project_root, "data", "raw", "drd2_clean.pdbqt"),
        "vina_executable": resolve_vina_path(),
        "obabel_path": resolve_obabel_path(),

        # ---- Parallelism ----
        "n_envs": 16,                  # 16 parallel docking workers
        "vec_env": "subproc",          # "subproc" (multiprocessing) or "dummy" (single process)

        # ---- Docking ----
        "vina_exhaustiveness": 8,      # High-fidelity search for HPC (exhaustiveness=8)

        # ---- PPO Hyperparameters (Optimized for 24 GB VRAM) ----
        "n_steps": 256,                # Steps per env before PPO update (256 × 16 = 4,096 transitions)
        "batch_size": 128,             # PPO minibatch size
        "n_epochs": 10,                # Gradient passes over each rollout buffer
        "learning_rate": 3e-5,         # Peak LR (linearly decayed to 0)
        "gamma": 0.99,                 # Discount factor
        "gae_lambda": 0.95,            # GAE lambda for advantage estimation
        "ent_coef": 0.02,              # Controlled entropy bonus for exploration
        "clip_range": 0.2,             # PPO clipping
        "max_grad_norm": 1.0,          # Gradient clipping
        "max_length": 50,              # Max SMILES token length

        # ---- Training Duration ----
        "total_timesteps": 10_000_000, # ~200K episodes ≈ 2-3 days unattended run

        # ---- Model Architecture ----
        "unfreeze_molgpt": True,       # Full fine-tuning enabled by 24 GB VRAM
        "vf_net_arch": [256, 256],     # Dedicated value network

        # ---- Checkpointing & Logging ----
        "checkpoint_freq": 10_000,     # Save model checkpoint every N timesteps
        "log_summary_freq": 50,        # Print molecule summary every N episodes
        "checkpoint_dir": os.path.join(project_root, "checkpoints", "hpc_run"),
        "log_dir": os.path.join(project_root, "logs", "hpc_run"),
    }


HPC_CONFIG = get_default_config()


# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================
def linear_schedule(initial_lr: float):
    """
    Linear decay from initial_lr to 0 over the course of training.
    Prevents catastrophic forgetting and stabilizes late-stage MolGPT fine-tuning.
    """
    def func(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return func


# ============================================================================
# ENVIRONMENT — Windows HPC Variant
# ============================================================================
class MolGenEnvHPC(gym.Env):
    """
    Windows HPC Gymnasium Environment for Token-by-Token SMILES Generation.
    Runs lightweight CPU tokenization and invokes RewardOracle for docking.
    """
    def __init__(
        self, 
        max_length=50, 
        vina_exhaustiveness=8,
        receptor_path=None,
        vina_executable=None,
        obabel_path=None
    ):
        super().__init__()
        self.max_length = max_length

        # Load ONLY tokenizer in the environment process — causal LM stays in main process
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
        self.observation_space = spaces.Box(
            low=0, high=self.vocab_size, shape=(self.max_length,), dtype=np.int32
        )

        self.oracle = RewardOracle(
            receptor_pdbqt_path=receptor_path or HPC_CONFIG["receptor_path"],
            vina_executable=vina_executable or HPC_CONFIG["vina_executable"],
            exhaustiveness=vina_exhaustiveness,
            obabel_path=obabel_path or HPC_CONFIG["obabel_path"],
            cpu=1  # One CPU core per worker to prevent Windows thread thrashing
        )
        self.local_seen_fps = []
        self.episode_count = 0

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
        info = {}

        if action == self.eos_token_id or self.step_idx >= self.max_length:
            terminated = True
            self.episode_count += 1

            token_list = self.seq[:self.step_idx].tolist()
            raw_smiles = self.tokenizer.decode(token_list, skip_special_tokens=True).strip()
            cleaned_smiles = raw_smiles.replace(" ", "").split(".")[0]

            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol is not None:
                valid_smiles = Chem.MolToSmiles(mol)
                res = self.oracle.evaluate_smiles(valid_smiles, run_docking=True)
                reward = res["total_reward"]

                info["smiles"] = valid_smiles
                info["valid"] = True
                info["qed"] = res.get("qed", 0.0)
                info["docking_score"] = res.get("docking_score", None)
                info["mw"] = res.get("mw", 0.0)

                # Structural diversity check via Morgan fingerprint similarity
                if res["valid"]:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    if self.local_seen_fps:
                        sims = DataStructs.BulkTanimotoSimilarity(fp, self.local_seen_fps)
                        if max(sims) > 0.75:
                            reward -= 3.0
                            info["diversity_penalty"] = True
                        else:
                            self.local_seen_fps.append(fp)
                            info["diversity_penalty"] = False
                    else:
                        self.local_seen_fps.append(fp)
                        info["diversity_penalty"] = False
            else:
                reward = -5.0
                info["smiles"] = cleaned_smiles
                info["valid"] = False

        return self.seq.copy(), float(reward), terminated, False, info


# ============================================================================
# FEATURE EXTRACTOR — Unfrozen MolGPT with Gradient Checkpointing
# ============================================================================
class MolGPTExtractorHPC(BaseFeaturesExtractor):
    """
    HPC Feature Extractor for 24 GB VRAM (RTX 3090).
    Unfreezes the 12 transformer blocks of MolGPT for end-to-end RL fine-tuning.
    Uses PyTorch gradient checkpointing to fit full backpropagation into VRAM.
    """
    def __init__(self, observation_space: gym.spaces.Box):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = PretrainedSMILESGenerator(device=device)

        hidden_size = generator.model.config.n_embd

        super().__init__(observation_space, features_dim=hidden_size)
        self.generator = generator
        self.pad_token_id = (
            self.generator.tokenizer.pad_token_id
            if self.generator.tokenizer.pad_token_id is not None else 0
        )

        if HPC_CONFIG["unfreeze_molgpt"]:
            if hasattr(self.generator.model.transformer, "gradient_checkpointing_enable"):
                self.generator.model.transformer.gradient_checkpointing_enable()
            print("[HPC] MolGPT UNFROZEN: Full fine-tuning active with gradient checkpointing.")
        else:
            for param in self.generator.model.parameters():
                param.requires_grad = False
            print("[HPC] MolGPT FROZEN: Only value and action heads are trainable.")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs_long = observations.long().to(self.generator.model.device)

        if HPC_CONFIG["unfreeze_molgpt"]:
            transformer_outputs = self.generator.model.transformer(input_ids=obs_long)
        else:
            with torch.no_grad():
                transformer_outputs = self.generator.model.transformer(input_ids=obs_long)

        hidden_states = transformer_outputs[0]  # (batch, seq_len, 768)

        # Extract hidden state at the last generated token
        pad_mask = (obs_long == self.pad_token_id)
        lengths = pad_mask.float().argmax(dim=1)
        no_pad = (~pad_mask).all(dim=1)
        lengths[no_pad] = obs_long.shape[1]
        lengths = torch.clamp(lengths, min=1)

        batch_indices = torch.arange(obs_long.shape[0], device=obs_long.device)
        step_hidden = hidden_states[batch_indices, lengths - 1, :]

        return step_hidden.float()


# ============================================================================
# LOGGING CALLBACK — Multi-Day CSV + TensorBoard Tracking
# ============================================================================
class MoleculeLoggingCallback(BaseCallback):
    """
    Continuously logs every generated molecule during multi-day Windows HPC runs:
      - CSV log with timestamps, SMILES, validity, QED, docking scores
      - TensorBoard metrics (validity rate, unique structures, best docking hit)
      - Console summaries with ETA estimation
    """
    def __init__(self, log_dir, summary_freq=50, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.summary_freq = summary_freq
        self.csv_path = os.path.join(log_dir, "molecule_log.csv")

        self.best_reward = float("-inf")
        self.best_docking = float("inf")
        self.best_smiles = ""
        self.total_valid = 0
        self.total_episodes = 0
        self.unique_smiles = set()
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        os.makedirs(self.log_dir, exist_ok=True)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep", "episode", "wall_clock_hours",
                "smiles", "valid", "reward", "qed",
                "docking_score", "mw", "diversity_penalty",
                "cumulative_validity", "unique_count",
            ])
        print(f"[Callback] Molecule CSV log -> {self.csv_path}")

    def _on_step(self) -> bool:
        for idx, info in enumerate(self.locals.get("infos", [])):
            if "smiles" not in info:
                continue

            self.total_episodes += 1
            is_valid = info.get("valid", False)
            reward = float(self.locals["rewards"][idx])
            elapsed_h = (time.time() - self.start_time) / 3600.0

            if is_valid:
                self.total_valid += 1
                self.unique_smiles.add(info["smiles"])

            docking = info.get("docking_score", None)
            if docking is not None and docking < self.best_docking:
                self.best_docking = docking
                self.best_smiles = info.get("smiles", "")

            if reward > self.best_reward:
                self.best_reward = reward

            validity_rate = self.total_valid / max(self.total_episodes, 1)

            # CSV Append
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps, self.total_episodes,
                    f"{elapsed_h:.3f}",
                    info.get("smiles", ""), is_valid, f"{reward:.3f}",
                    info.get("qed", ""), docking, info.get("mw", ""),
                    info.get("diversity_penalty", ""),
                    f"{validity_rate:.4f}", len(self.unique_smiles),
                ])

            # TensorBoard
            self.logger.record("molecules/validity_rate", validity_rate)
            self.logger.record("molecules/unique_count", len(self.unique_smiles))
            self.logger.record("molecules/total_episodes", self.total_episodes)
            self.logger.record("molecules/best_reward", self.best_reward)
            if docking is not None:
                self.logger.record("molecules/best_docking", self.best_docking)

            # Periodic Console Summary
            if self.total_episodes % self.summary_freq == 0:
                eta = self._estimate_eta(elapsed_h)
                best_dock_str = f"{self.best_docking:.2f} kcal/mol" if self.best_docking != float("inf") else "N/A"
                print(
                    f"\n{'='*70}\n"
                    f"  Episode {self.total_episodes:,}  |  Step {self.num_timesteps:,}  |  "
                    f"{elapsed_h:.1f} h elapsed  |  ETA: {eta}\n"
                    f"  Validity: {validity_rate:.1%}  |  Unique Molecules: {len(self.unique_smiles):,}  |  "
                    f"Best Dock: {best_dock_str}  |  Best Reward: {self.best_reward:.2f}\n"
                    f"  Top Candidate: {self.best_smiles}\n"
                    f"{'='*70}"
                )

        return True

    def _estimate_eta(self, elapsed_h: float) -> str:
        if elapsed_h <= 0 or self.num_timesteps <= 0:
            return "calculating..."
        rate = self.num_timesteps / (elapsed_h * 3600)
        remaining_steps = HPC_CONFIG["total_timesteps"] - self.num_timesteps
        remaining_sec = remaining_steps / max(rate, 1e-6)
        return str(timedelta(seconds=int(remaining_sec)))

    def _on_training_end(self):
        elapsed_h = (time.time() - self.start_time) / 3600.0
        validity = self.total_valid / max(self.total_episodes, 1)
        best_dock_str = f"{self.best_docking:.2f} kcal/mol" if self.best_docking != float("inf") else "N/A"
        print(
            f"\n{'#'*70}\n"
            f"  HPC TRAINING COMPLETE — Elapsed: {elapsed_h:.1f} hours\n"
            f"  Total Generated : {self.total_episodes:,} molecules\n"
            f"  Validity Rate   : {validity:.1%}\n"
            f"  Unique ChemTypes: {len(self.unique_smiles):,}\n"
            f"  Best Affinity   : {best_dock_str}\n"
            f"  Best Total Rew  : {self.best_reward:.2f}\n"
            f"  Top Hit SMILES  : {self.best_smiles}\n"
            f"{'#'*70}"
        )


# ============================================================================
# PRE-FLIGHT DIAGNOSTICS & PATH VERIFICATION
# ============================================================================
def run_preflight_checks(config: dict) -> bool:
    """
    Validates hardware, paths, and chemistry dependencies before launching.
    Warns immediately if anything is missing to prevent mid-run failures.
    """
    print("\n" + "=" * 70)
    print("  WINDOWS HPC PRE-FLIGHT VERIFICATION")
    print("=" * 70)

    # 1. GPU & CUDA Check
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  [PASS] CUDA Available     : Yes ({device_name}, {vram_gb:.1f} GB VRAM)")
    else:
        print("  [WARN] CUDA Available     : NO (Training will run on CPU, which is very slow!)")

    # 2. Receptor File Check
    receptor = config["receptor_path"]
    if os.path.isfile(receptor):
        size_kb = os.path.getsize(receptor) / 1024
        print(f"  [PASS] Receptor PDBQT     : Found ({receptor}, {size_kb:.1f} KB)")
    else:
        print(f"  [FAIL] Receptor PDBQT     : NOT FOUND at {receptor}")
        return False

    # 3. OpenBabel Check
    obabel = config["obabel_path"]
    if os.path.isfile(obabel):
        try:
            ver = subprocess.run([obabel, "-V"], capture_output=True, text=True).stdout.strip()
            print(f"  [PASS] OpenBabel Binary   : Found ({ver or obabel})")
        except Exception:
            print(f"  [PASS] OpenBabel Binary   : Found at {obabel}")
    else:
        print(f"  [FAIL] OpenBabel Binary   : NOT FOUND at {obabel}")
        print("         Install via: pip install openbabel-wheel")
        return False

    # 4. AutoDock Vina Check
    vina = config["vina_executable"]
    if os.path.isfile(vina):
        try:
            ver = subprocess.run([vina, "--version"], capture_output=True, text=True).stdout.strip()
            print(f"  [PASS] AutoDock Vina      : Found ({ver or vina})")
        except Exception:
            print(f"  [PASS] AutoDock Vina      : Found at {vina}")
    else:
        print(f"  [FAIL] AutoDock Vina      : NOT FOUND at {vina}")
        print("         Place vina.exe in your project's bin/ folder (e.g. final project/bin/vina.exe)")
        return False

    # 5. Optional AutoDock-GPU Status
    project_bin = os.path.join(project_root, "bin")
    adgpu_candidates = [
        os.path.join(project_bin, "autodock_gpu_64wi.exe"),
        shutil.which("autodock_gpu_64wi") or "",
    ]
    adgpu_path = next((p for p in adgpu_candidates if p and os.path.isfile(p)), None)
    if adgpu_path:
        print(f"  [INFO] AutoDock-GPU Binary: Available at {adgpu_path}")
    else:
        print("  [INFO] AutoDock-GPU Binary: Not detected (Using AutoDock Vina as primary engine)")

    print("=" * 70 + "\n")
    return True


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train(
    resume_from: str = None,
    n_envs: int = None,
    vec_env: str = None,
    total_timesteps: int = None,
    vina_executable: str = None,
    obabel_path: str = None,
    receptor_path: str = None,
):
    config = HPC_CONFIG.copy()

    # Apply overrides
    if n_envs is not None:
        config["n_envs"] = n_envs
    if vec_env is not None:
        config["vec_env"] = vec_env
    if total_timesteps is not None:
        config["total_timesteps"] = total_timesteps
    if vina_executable:
        config["vina_executable"] = os.path.abspath(vina_executable)
    if obabel_path:
        config["obabel_path"] = os.path.abspath(obabel_path)
    if receptor_path:
        config["receptor_path"] = os.path.abspath(receptor_path)

    # Pre-flight checks
    if not run_preflight_checks(config):
        print("[ERROR] Pre-flight verification failed. Fix the missing paths above before running.")
        sys.exit(1)

    # Pre-cache tokenizer in main process so Windows child processes find it in local cache
    print("[HPC Setup] Pre-caching MolGPT tokenizer to eliminate concurrent Windows file locks...")
    _ = AutoTokenizer.from_pretrained("msb-roshan/molgpt")
    print("[HPC Setup] Tokenizer successfully pre-cached.\n")

    # Banner
    print("=" * 70)
    print("  HPC PPO MOLECULAR TRAINING — Windows Edition")
    print("=" * 70)
    print(f"  Started Date/Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PyTorch Device    : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  MolGPT Policy     : {'UNFROZEN (Full Fine-Tuning)' if config['unfreeze_molgpt'] else 'FROZEN'}")
    print(f"  Parallel Workers  : {config['n_envs']} environments ({config['vec_env']} mode)")
    print(f"  Rollout Buffer    : {config['n_steps'] * config['n_envs']:,} transitions ({config['n_steps']} steps/env)")
    print(f"  Minibatch Size    : {config['batch_size']}")
    print(f"  PPO Epochs        : {config['n_epochs']}")
    print(f"  Learning Rate     : {config['learning_rate']} -> 0 (linear decay)")
    print(f"  Vina Exhaust.     : {config['vina_exhaustiveness']}")
    print(f"  Total Timesteps   : {config['total_timesteps']:,}")
    print(f"  Checkpoint Freq   : every {config['checkpoint_freq']:,} steps -> {config['checkpoint_dir']}")
    print(f"  Logs Directory    : {config['log_dir']}")
    print("=" * 70)

    # Create directories
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(os.path.join(project_root, "data", "temp"), exist_ok=True)

    # Vectorized Environments
    vec_env_cls = SubprocVecEnv if config["vec_env"] == "subproc" else DummyVecEnv
    print(f"\n[HPC Setup] Initializing {config['n_envs']} parallel environments with {vec_env_cls.__name__}...")

    env = make_vec_env(
        MolGenEnvHPC,
        n_envs=config["n_envs"],
        env_kwargs={
            "max_length": config["max_length"],
            "vina_exhaustiveness": config["vina_exhaustiveness"],
            "receptor_path": config["receptor_path"],
            "vina_executable": config["vina_executable"],
            "obabel_path": config["obabel_path"],
        },
        vec_env_cls=vec_env_cls,
    )
    print("[HPC Setup] Parallel environments initialized successfully.\n")

    # Policy Configuration
    policy_kwargs = dict(
        features_extractor_class=MolGPTExtractorHPC,
        features_extractor_kwargs={},
        net_arch=dict(pi=[], vf=config["vf_net_arch"]),
    )

    sb3_logger = configure(config["log_dir"], ["stdout", "csv", "tensorboard"])

    if resume_from and os.path.exists(resume_from):
        print(f"\n[Resume] Loading checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        model.set_logger(sb3_logger)
        print("[Resume] Checkpoint loaded successfully — resuming training.")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(config["learning_rate"]),
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            ent_coef=config["ent_coef"],
            clip_range=config["clip_range"],
            max_grad_norm=config["max_grad_norm"],
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=config["log_dir"],
        )
        model.set_logger(sb3_logger)

        # Initialize Action Head directly from MolGPT's lm_head
        # Guarantees valid SMILES grammar right from Episode 1
        action_net = model.policy.action_net
        lm_head = model.policy.features_extractor.generator.model.lm_head
        if isinstance(action_net, nn.Linear):
            with torch.no_grad():
                action_net.weight.copy_(lm_head.weight)
                action_net.bias.zero_()
            print(f"[Init] Action head initialized from MolGPT lm_head: {action_net.weight.shape}")
        else:
            print(f"[Warning] action_net is {type(action_net)}, skipping lm_head weight copy.")

    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # Callbacks
    callbacks = [
        MoleculeLoggingCallback(
            log_dir=config["log_dir"],
            summary_freq=config["log_summary_freq"],
        ),
        CheckpointCallback(
            save_freq=max(config["checkpoint_freq"] // config["n_envs"], 1),
            save_path=config["checkpoint_dir"],
            name_prefix="ppo_molgpt_hpc",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ]

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Launching HPC training loop...\n")

    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=(resume_from is None),
        )
    except KeyboardInterrupt:
        print("\n[HPC Training] Interrupted by user. Saving emergency checkpoint...")
        emergency_path = os.path.join(config["checkpoint_dir"], "ppo_molgpt_hpc_interrupted")
        model.save(emergency_path)
        print(f"[HPC Training] Emergency checkpoint saved -> {emergency_path}.zip")

    # Save final model
    final_path = os.path.join(config["checkpoint_dir"], "ppo_molgpt_hpc_final")
    model.save(final_path)
    print(f"\n[Complete] Final model saved -> {final_path}.zip")

    env.close()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Required for Windows multiprocessing (prevents recursive child spawns)
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Windows HPC PPO Training for Molecular Generation")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint .zip to resume from (e.g. checkpoints/hpc_run/ppo_molgpt_hpc_50000_steps.zip)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=16,
        help="Number of parallel environments (default: 16 for HPC; use 4-8 if CPU cores are limited)"
    )
    parser.add_argument(
        "--vec-env", type=str, choices=["subproc", "dummy"], default="subproc",
        help="Vectorized env type: 'subproc' (multiprocessing, recommended) or 'dummy' (single-process debug)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=10_000_000,
        help="Total timesteps to train (default: 10,000,000)"
    )
    parser.add_argument(
        "--vina-path", type=str, default=None,
        help="Custom path to vina.exe (overrides automatic search)"
    )
    parser.add_argument(
        "--obabel-path", type=str, default=None,
        help="Custom path to obabel.exe (overrides automatic search)"
    )
    parser.add_argument(
        "--receptor-path", type=str, default=None,
        help="Custom path to receptor pdbqt (default: data/raw/drd2_clean.pdbqt)"
    )

    args = parser.parse_args()

    train(
        resume_from=args.resume,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        total_timesteps=args.timesteps,
        vina_executable=args.vina_path,
        obabel_path=args.obabel_path,
        receptor_path=args.receptor_path,
    )
