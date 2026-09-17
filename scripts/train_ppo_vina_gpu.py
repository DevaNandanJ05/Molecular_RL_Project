"""
Windows HPC PPO Training Script with Vina-GPU 2.1 Acceleration
===================================================================
Hardware Target: Windows-based HPC / Workstation (NVIDIA RTX GPUs)
Multi-Day Unattended Training with Full MolGPT Fine-Tuning.

Vina-GPU 2.1 Compilation Instructions (Phase B):
------------------------------------------------
1. Clone the repository: 
   git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git
2. Install dependencies:
   - Visual Studio 2019 (with C++ build tools)
   - CUDA Toolkit (v11.5 or v12.x matching your driver)
   - Boost 1.77.0 (extract to a folder, e.g., C:\\local\\boost_1_77_0)
3. Open the .sln file in Visual Studio 2019.
4. Set configuration to Release, x64.
5. In Project Properties:
   - C/C++ -> General -> Additional Include Directories: Add Boost path & CUDA include path
   - Linker -> General -> Additional Library Directories: Add CUDA lib/x64 path
   - Linker -> Input -> Additional Dependencies: Add OpenCL.lib
6. Build Solution (produces Vina-GPU.exe and Vina-GPU-K.exe).
7. Copy Vina-GPU.exe and Vina-GPU-K.exe into your final project/bin/ directory.
8. Run the kernel generator once: `Vina-GPU-K.exe --config=...` (or let this script handle it).
"""

import os
import sys
import csv
import time
import shutil
from collections import deque
from multiprocessing.managers import BaseManager
import argparse
import multiprocessing
import subprocess
import uuid
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
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit warning floods
RDLogger.DisableLog('rdApp.*')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.policy_network import PretrainedSMILESGenerator
from models.reward_oracle import resolve_obabel_path

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# ============================================================================
# VINA-GPU PATH RESOLUTION
# ============================================================================
def resolve_vina_gpu_path(custom_path=None) -> str:
    """Locates Vina-GPU.exe in the project bin directory or via custom path."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)
    
    bin_path = os.path.join(project_root, "bin", "Vina-GPU.exe")
    if os.path.isfile(bin_path):
        return bin_path
        
    return custom_path or r"C:\Vina-GPU\Vina-GPU.exe"

# ============================================================================
# WINDOWS HPC CONFIGURATION
# ============================================================================
def get_vina_gpu_config() -> dict:
    """Returns the default configuration optimized for Vina-GPU on RTX GPUs."""
    return {
        # ---- File Paths ----
        "receptor_path": os.path.join(project_root, "data", "raw", "drd2_clean.pdbqt"),
        "vina_gpu_executable": resolve_vina_gpu_path(),
        "obabel_path": resolve_obabel_path(),

        # ---- Parallelism ----
        "n_envs": 32,                  # High parallel workers to batch ligands to GPU
        "vec_env": "subproc",

        # ---- Docking ----
        "vina_exhaustiveness": 8,      # Can afford higher exhaustiveness on GPU (search_depth in Vina-GPU)

        # ---- PPO Hyperparameters (Stabilized for RTX 3060 12 GB) ----
        "n_steps": 512,                # 512 × 32 = 16,384 transitions per rollout
        "batch_size": 512,             # Larger minibatch size for GPU throughput
        "n_epochs": 4,                 # Conservative 4 epochs to prevent policy drift / over-optimization
        "learning_rate": 5e-6,         # Stabilized learning rate (proven to protect MolGPT grammar)
        "gamma": 0.99,                 
        "gae_lambda": 0.95,            
        "ent_coef": 0.04,              # Increased entropy bonus
        "clip_range": 0.15,            # Tighter clipping
        "max_grad_norm": 0.5,          # Tighter gradient clipping
        "target_kl": 0.03,             # Early-stop threshold
        "max_length": 50,              

        # ---- Training Duration ----
        "total_timesteps": 10_000_000, 

        # ---- Model Architecture ----
        "unfreeze_molgpt": True,       
        "vf_net_arch": [512, 256],     

        # ---- Checkpointing & Logging ----
        "checkpoint_freq": 10_000,     
        "log_summary_freq": 50,        
        "checkpoint_dir": os.path.join(project_root, "checkpoints", "hpc_vina_gpu_run"),
        "log_dir": os.path.join(project_root, "logs", "hpc_vina_gpu_run"),
    }

VINA_GPU_CONFIG = get_vina_gpu_config()

# ============================================================================
# SHARED TANIMOTO FINGERPRINT BUFFER  (cross-process via Manager server)
# ============================================================================
class _FingerprintBufferServer:
    """
    Runs inside a dedicated Manager server process — completely independent
    of the 32 SubprocVecEnv workers. Each worker holds a lightweight proxy
    that transparently forwards push/get_all calls over IPC.

    Fingerprints are stored as UTF-8 bit-strings because RDKit ExplicitBitVect
    objects are not directly picklable across process boundaries.
    """
    def __init__(self, maxlen: int = 3200):  # default: 100 per worker × 32 workers
        self._maxlen = maxlen
        self._buffer = deque(maxlen=maxlen)
        self._lock = multiprocessing.Lock()

    def push(self, fp_bytes: bytes) -> None:
        """Register a new fingerprint into the global shared buffer."""
        with self._lock:
            self._buffer.append(fp_bytes)

    def get_all(self) -> list:
        """Return a snapshot of all fingerprints currently in the buffer."""
        with self._lock:
            return list(self._buffer)

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


class SharedFingerprintManager(BaseManager):
    """Custom Manager that exposes the fingerprint buffer server to subprocesses."""
    pass

SharedFingerprintManager.register('FingerprintBuffer', _FingerprintBufferServer)


def create_shared_buffer(maxlen: int = 3200):
    """
    Call ONCE in the main process before SubprocVecEnv is created.
    Returns (manager, proxy). The caller must keep `manager` alive for
    the entire duration of training (shutdown in a finally block).
    """
    mgr = SharedFingerprintManager()
    mgr.start()
    proxy = mgr.FingerprintBuffer(maxlen=maxlen)
    return mgr, proxy


def _fp_to_bytes(fp) -> bytes:
    """Serialize an RDKit ExplicitBitVect to a cross-process-safe byte string."""
    return fp.ToBitString().encode('utf-8')


def _bytes_to_fp(b: bytes):
    """Deserialize a byte string back to an RDKit ExplicitBitVect."""
    bit_string = b.decode('utf-8')
    fp = DataStructs.ExplicitBitVect(len(bit_string))
    for i, ch in enumerate(bit_string):
        if ch == '1':
            fp.SetBit(i)
    return fp


# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================
def linear_schedule(initial_lr: float):
    def func(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return func

# ============================================================================
# REWARD ORACLE FOR VINA-GPU
# ============================================================================
class RewardOracleVinaGPU:
    def __init__(
        self,
        receptor_pdbqt_path="data/raw/drd2_clean.pdbqt",
        vina_gpu_executable=None,
        exhaustiveness=8,
        obabel_path=None,
        similarity_threshold=0.75,
        diversity_penalty=3.0,
        recent_buffer_size=100,
        shared_buffer=None,          # Manager proxy — shared across all 32 workers
    ):
        if receptor_pdbqt_path and not os.path.isabs(receptor_pdbqt_path):
            self.receptor_path = os.path.abspath(os.path.join(project_root, receptor_pdbqt_path))
        else:
            self.receptor_path = receptor_pdbqt_path

        self.vina_gpu_executable = resolve_vina_gpu_path(vina_gpu_executable)
        self.obabel_path = resolve_obabel_path(obabel_path)
        self.exhaustiveness = exhaustiveness

        self.similarity_threshold = float(similarity_threshold)
        self.diversity_penalty = float(diversity_penalty)
        self.recent_buffer_size = int(recent_buffer_size)

        # Prefer the globally shared cross-worker buffer.
        # Falls back to a per-instance local deque when running without IPC
        # (e.g., unit tests, DummyVecEnv, or standalone evaluation).
        self.shared_buffer = shared_buffer
        self.local_fallback = deque(maxlen=self.recent_buffer_size)

    def reset_diversity_buffer(self):
        """Clears whichever diversity buffer is active."""
        if self.shared_buffer is not None:
            try:
                self.shared_buffer.clear()
            except Exception:
                pass
        else:
            self.local_fallback.clear()

    def _check_and_register_diversity(self, mol) -> tuple:
        """
        Computes the Tanimoto diversity penalty for `mol` against the active
        fingerprint buffer (shared global or local fallback), then registers
        the molecule's fingerprint into that buffer.

        Returns:
            (max_similarity: float, penalty_applied: float)
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        max_sim = 0.0
        penalty_applied = 0.0

        if self.shared_buffer is not None:
            # ── Shared cross-worker path ──────────────────────────────────────
            try:
                all_fp_bytes = self.shared_buffer.get_all()
                if all_fp_bytes:
                    all_fps = [_bytes_to_fp(b) for b in all_fp_bytes]
                    sims = DataStructs.BulkTanimotoSimilarity(fp, all_fps)
                    max_sim = float(max(sims))
                    if max_sim > self.similarity_threshold:
                        penalty_applied = self.diversity_penalty
                # Register the new fingerprint into the global shared buffer
                self.shared_buffer.push(_fp_to_bytes(fp))
            except Exception:
                # IPC failure safety: silently skip penalty rather than crash worker
                pass
        else:
            # ── Local per-instance fallback path ──────────────────────────────
            if self.local_fallback:
                sims = DataStructs.BulkTanimotoSimilarity(fp, list(self.local_fallback))
                max_sim = float(max(sims))
                if max_sim > self.similarity_threshold:
                    penalty_applied = self.diversity_penalty
            self.local_fallback.append(fp)

        return max_sim, penalty_applied

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

        # Tanimoto diversity check — uses the globally shared buffer if available,
        # otherwise falls back to the local per-instance deque.
        max_sim, diversity_penalty_applied = self._check_and_register_diversity(mol)
        penalty += diversity_penalty_applied

        if not run_docking or self.receptor_path is None or not os.path.exists(self.receptor_path):
            total_reward = (qed_score * 3.0) - penalty
            return {
                "valid": True, "qed": qed_score, "mw": mw, 
                "docking_score": None, "total_reward": total_reward,
                "max_tanimoto": max_sim, "diversity_penalty": diversity_penalty_applied
            }

        # 3D Docking Evaluation
        docking_score = self._run_vina_gpu_docking(mol)
        
        if docking_score is not None and docking_score < 0:
            total_reward = abs(docking_score) + (qed_score * 1.5) - penalty
        else:
            total_reward = -2.0 + (qed_score * 1.5) - penalty

        return {
            "valid": True, "qed": qed_score, "mw": mw,
            "docking_score": docking_score, "total_reward": total_reward,
            "max_tanimoto": max_sim, "diversity_penalty": diversity_penalty_applied
        }

    def _run_vina_gpu_docking(self, mol) -> float:
        temp_dir = os.path.join(project_root, "data", "temp")
        os.makedirs(temp_dir, exist_ok=True)

        thread_id = uuid.uuid4().hex
        
        # Vina-GPU requires a folder containing ligand(s)
        ligand_dir = os.path.join(temp_dir, f"ligand_{thread_id}")
        os.makedirs(ligand_dir, exist_ok=True)
        
        temp_sdf = os.path.join(temp_dir, f"temp_{thread_id}.sdf")
        temp_pdbqt = os.path.join(ligand_dir, "ligand.pdbqt")
        config_file = os.path.join(temp_dir, f"config_{thread_id}.txt")
        
        try:
            mol_3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) != 0:
                return 0.0 
            
            AllChem.UFFOptimizeMolecule(mol_3d)

            writer = Chem.SDWriter(temp_sdf)
            writer.write(mol_3d)
            writer.close()

            # Set BABEL_DATADIR automatically
            env = os.environ.copy()
            obabel_dir = os.path.dirname(self.obabel_path)
            candidate_data_dirs = [
                os.path.join(obabel_dir, "data"),
                os.path.join(obabel_dir, "bin", "data"),
                os.path.join(os.path.dirname(obabel_dir), "share", "openbabel"),
            ]
            for candidate in candidate_data_dirs:
                if os.path.exists(candidate):
                    env["BABEL_DATADIR"] = candidate
                    break

            babel_cmd = [self.obabel_path, temp_sdf, "-O", temp_pdbqt, "-h"]
            babel_res = subprocess.run(babel_cmd, capture_output=True, text=True, env=env)
            
            if babel_res.returncode != 0 or not os.path.exists(temp_pdbqt) or os.path.getsize(temp_pdbqt) == 0:
                return 0.0

            # Write config file for Vina-GPU 2.1
            with open(config_file, "w") as f:
                f.write(f"receptor = {self.receptor_path}\n")
                f.write(f"ligand_directory = {ligand_dir}\n")
                f.write("center_x = 9.5\n")
                f.write("center_y = 5.2\n")
                f.write("center_z = -11.4\n")
                f.write("size_x = 20.0\n")
                f.write("size_y = 20.0\n")
                f.write("size_z = 20.0\n")
                f.write("thread = 8000\n")
                f.write(f"search_depth = {self.exhaustiveness}\n")

            vina_cmd = [
                self.vina_gpu_executable,
                f"--config={config_file}"
            ]

            result = subprocess.run(vina_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                err_msg = result.stderr.strip() or result.stdout.strip()
                # Suppress printing for every failure to avoid console spam, return 0.0
                return 0.0

            # Parse standard output for best docking score
            for line in result.stdout.split('\n'):
                if "   1 " in line:
                    parts = line.split()
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue

        except Exception as e:
            return 0.0
        finally:
            for f in [temp_sdf, config_file]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except OSError: pass
            if os.path.exists(ligand_dir):
                try: shutil.rmtree(ligand_dir)
                except OSError: pass
                
        return 0.0

# ============================================================================
# ENVIRONMENT
# ============================================================================
class MolGenEnvVinaGPU(gym.Env):
    def __init__(
        self,
        max_length=50,
        vina_exhaustiveness=8,
        receptor_path=None,
        vina_gpu_executable=None,
        obabel_path=None,
        shared_buffer=None,           # Manager proxy passed in from main process
    ):
        super().__init__()
        self.max_length = max_length

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

        self.oracle = RewardOracleVinaGPU(
            receptor_pdbqt_path=receptor_path or VINA_GPU_CONFIG["receptor_path"],
            vina_gpu_executable=vina_gpu_executable or VINA_GPU_CONFIG["vina_gpu_executable"],
            exhaustiveness=vina_exhaustiveness,
            obabel_path=obabel_path or VINA_GPU_CONFIG["obabel_path"],
            shared_buffer=shared_buffer,   # Wire in the cross-worker global buffer
        )
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

                # Internal diversity check is managed by RewardOracleVinaGPU against recent candidates
                info["max_tanimoto"] = res.get("max_tanimoto", 0.0)
                info["diversity_penalty"] = (res.get("diversity_penalty", 0.0) > 0.0)
            else:
                reward = -5.0
                info["smiles"] = cleaned_smiles
                info["valid"] = False

        return self.seq.copy(), float(reward), terminated, False, info

# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================
class MolGPTExtractorGPU(BaseFeaturesExtractor):
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

        if VINA_GPU_CONFIG["unfreeze_molgpt"]:
            if hasattr(self.generator.model.transformer, "gradient_checkpointing_enable"):
                self.generator.model.transformer.gradient_checkpointing_enable()
        else:
            for param in self.generator.model.parameters():
                param.requires_grad = False

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs_long = observations.long().to(self.generator.model.device)

        if VINA_GPU_CONFIG["unfreeze_molgpt"]:
            transformer_outputs = self.generator.model.transformer(input_ids=obs_long)
        else:
            with torch.no_grad():
                transformer_outputs = self.generator.model.transformer(input_ids=obs_long)

        hidden_states = transformer_outputs[0]
        pad_mask = (obs_long == self.pad_token_id)
        lengths = pad_mask.float().argmax(dim=1)
        no_pad = (~pad_mask).all(dim=1)
        lengths[no_pad] = obs_long.shape[1]
        lengths = torch.clamp(lengths, min=1)

        batch_indices = torch.arange(obs_long.shape[0], device=obs_long.device)
        step_hidden = hidden_states[batch_indices, lengths - 1, :]

        return step_hidden.float()

# ============================================================================
# LOGGING CALLBACK
# ============================================================================
class MoleculeLoggingCallback(BaseCallback):
    def __init__(self, log_dir, summary_freq=50, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.summary_freq = summary_freq
        self.csv_path = os.path.join(log_dir, "molecule_log_gpu.csv")

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

            self.logger.record("molecules/validity_rate", validity_rate)
            self.logger.record("molecules/unique_count", len(self.unique_smiles))
            self.logger.record("molecules/total_episodes", self.total_episodes)
            self.logger.record("molecules/best_reward", self.best_reward)
            if docking is not None:
                self.logger.record("molecules/best_docking", self.best_docking)
            if "max_tanimoto" in info:
                self.logger.record("molecules/max_tanimoto", info["max_tanimoto"])

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
        remaining_steps = VINA_GPU_CONFIG["total_timesteps"] - self.num_timesteps
        remaining_sec = remaining_steps / max(rate, 1e-6)
        return str(timedelta(seconds=int(remaining_sec)))

    def _on_training_end(self):
        elapsed_h = (time.time() - self.start_time) / 3600.0
        validity = self.total_valid / max(self.total_episodes, 1)
        best_dock_str = f"{self.best_docking:.2f} kcal/mol" if self.best_docking != float("inf") else "N/A"
        print(
            f"\n{'#'*70}\n"
            f"  HPC VINA-GPU TRAINING COMPLETE — Elapsed: {elapsed_h:.1f} hours\n"
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
    print("\n" + "=" * 70)
    print("  WINDOWS HPC PRE-FLIGHT VERIFICATION (VINA-GPU)")
    print("=" * 70)

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  [PASS] CUDA Available     : Yes ({device_name}, {vram_gb:.1f} GB VRAM)")
    else:
        print("  [FAIL] CUDA Available     : NO (Vina-GPU requires an NVIDIA GPU!)")
        return False

    receptor = config["receptor_path"]
    if os.path.isfile(receptor):
        size_kb = os.path.getsize(receptor) / 1024
        print(f"  [PASS] Receptor PDBQT     : Found ({receptor}, {size_kb:.1f} KB)")
    else:
        print(f"  [FAIL] Receptor PDBQT     : NOT FOUND at {receptor}")
        return False

    obabel = config["obabel_path"]
    if os.path.isfile(obabel):
        print(f"  [PASS] OpenBabel Binary   : Found at {obabel}")
    else:
        print(f"  [FAIL] OpenBabel Binary   : NOT FOUND at {obabel}")
        return False

    vina_gpu = config["vina_gpu_executable"]
    if os.path.isfile(vina_gpu):
        print(f"  [PASS] Vina-GPU Binary    : Found at {vina_gpu}")
    else:
        print(f"  [FAIL] Vina-GPU Binary    : NOT FOUND at {vina_gpu}")
        print("         Please compile Vina-GPU 2.1 and place Vina-GPU.exe in the bin folder.")
        return False

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
    vina_gpu_executable: str = None,
    obabel_path: str = None,
    receptor_path: str = None,
):
    config = VINA_GPU_CONFIG.copy()

    if n_envs is not None: config["n_envs"] = n_envs
    if vec_env is not None: config["vec_env"] = vec_env
    if total_timesteps is not None: config["total_timesteps"] = total_timesteps
    if vina_gpu_executable: config["vina_gpu_executable"] = os.path.abspath(vina_gpu_executable)
    if obabel_path: config["obabel_path"] = os.path.abspath(obabel_path)
    if receptor_path: config["receptor_path"] = os.path.abspath(receptor_path)

    if not run_preflight_checks(config):
        print("[ERROR] Pre-flight verification failed. Fix the missing paths above before running.")
        sys.exit(1)

    print("[HPC Setup] Pre-caching MolGPT tokenizer...")
    _ = AutoTokenizer.from_pretrained("msb-roshan/molgpt")

    # ── Shared Tanimoto Buffer ────────────────────────────────────────────────
    # Start the Manager server process BEFORE SubprocVecEnv spawns workers.
    # Each worker receives a proxy to this single global buffer so that the
    # Tanimoto diversity penalty is enforced across ALL parallel environments.
    shared_manager = None
    shared_fp_buffer = None
    if config["vec_env"] == "subproc":
        buffer_maxlen = config["n_envs"] * 100   # 100 fp slots per worker
        print(f"[HPC Setup] Starting shared Tanimoto buffer server (capacity: {buffer_maxlen} fingerprints)...")
        shared_manager, shared_fp_buffer = create_shared_buffer(maxlen=buffer_maxlen)
        print("[HPC Setup] Shared buffer online — all workers will share one global diversity memory.")
    else:
        print("[HPC Setup] DummyVecEnv detected — shared buffer not needed (single process).")
    # ─────────────────────────────────────────────────────────────────────────

    print("=" * 70)
    print("  HPC PPO MOLECULAR TRAINING — Vina-GPU Edition")
    print("=" * 70)
    print(f"  Workers         : {config['n_envs']} environments")
    print(f"  Rollout Buffer  : {config['n_steps'] * config['n_envs']:,} transitions")
    print(f"  Minibatch Size  : {config['batch_size']}")
    print(f"  PPO Epochs      : {config['n_epochs']}")
    print(f"  Learning Rate   : {config['learning_rate']}")
    print(f"  Target KL       : {config['target_kl']}")
    print(f"  Diversity Buffer: {'Shared (global)' if shared_fp_buffer else 'Local (per-worker)'}")
    print("=" * 70)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    vec_env_cls = SubprocVecEnv if config["vec_env"] == "subproc" else DummyVecEnv
    env = make_vec_env(
        MolGenEnvVinaGPU,
        n_envs=config["n_envs"],
        env_kwargs={
            "max_length": config["max_length"],
            "vina_exhaustiveness": config["vina_exhaustiveness"],
            "receptor_path": config["receptor_path"],
            "vina_gpu_executable": config["vina_gpu_executable"],
            "obabel_path": config["obabel_path"],
            "shared_buffer": shared_fp_buffer,   # Global diversity memory for all workers
        },
        vec_env_cls=vec_env_cls,
    )

    policy_kwargs = dict(
        features_extractor_class=MolGPTExtractorGPU,
        features_extractor_kwargs={},
        net_arch=dict(pi=[], vf=config["vf_net_arch"]),
    )

    sb3_logger = configure(config["log_dir"], ["stdout", "csv", "tensorboard"])

    if resume_from and os.path.exists(resume_from):
        model = PPO.load(resume_from, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        model.set_logger(sb3_logger)
    else:
        model = PPO(
            "MlpPolicy", env,
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
            target_kl=config.get("target_kl", None),
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=config["log_dir"],
        )
        model.set_logger(sb3_logger)

        action_net = model.policy.action_net
        lm_head = model.policy.features_extractor.generator.model.lm_head
        if isinstance(action_net, nn.Linear):
            with torch.no_grad():
                action_net.weight.copy_(lm_head.weight)
                action_net.bias.zero_()

    callbacks = [
        MoleculeLoggingCallback(log_dir=config["log_dir"], summary_freq=config["log_summary_freq"]),
        CheckpointCallback(
            save_freq=max(config["checkpoint_freq"] // config["n_envs"], 1),
            save_path=config["checkpoint_dir"],
            name_prefix="ppo_molgpt_hpc",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ]

    try:
        model.learn(total_timesteps=config["total_timesteps"], callback=callbacks, reset_num_timesteps=(resume_from is None))
    except KeyboardInterrupt:
        model.save(os.path.join(config["checkpoint_dir"], "ppo_molgpt_hpc_interrupted"))
    finally:
        env.close()
        # Cleanly shut down the Manager server process to release the IPC socket.
        if shared_manager is not None:
            print("[Cleanup] Shutting down shared fingerprint buffer server...")
            shared_manager.shutdown()

    model.save(os.path.join(config["checkpoint_dir"], "ppo_molgpt_hpc_final"))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Vina-GPU PPO Training")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--vec-env", type=str, choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--vina-gpu-path", type=str, default=None)
    parser.add_argument("--obabel-path", type=str, default=None)
    parser.add_argument("--receptor-path", type=str, default=None)

    args = parser.parse_args()
    train(
        resume_from=args.resume,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        total_timesteps=args.timesteps,
        vina_gpu_executable=args.vina_gpu_path,
        obabel_path=args.obabel_path,
        receptor_path=args.receptor_path,
    )
