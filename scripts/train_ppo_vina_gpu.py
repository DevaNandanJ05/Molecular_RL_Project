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
# Ensure UTF-8 output with fallback on Windows consoles (prevents cp1252 charmap crashes)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import csv
import re
import time
import shutil
import glob
import logging
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
# VINA CPU PATH RESOLUTION
# NOTE: Vina-GPU (CUDA) is disabled due to current CUDA build incompatibility.
#       All docking runs on the CPU AutoDock Vina engine (vina.exe).
# ============================================================================
def resolve_vina_cpu_path(custom_path=None) -> str:
    """Locates AutoDock Vina CPU (vina.exe) in the project bin directory."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    # Primary: project-local bin/vina.exe
    bin_cpu = os.path.join(project_root, "bin", "vina.exe")
    if os.path.isfile(bin_cpu):
        return bin_cpu

    # Secondary: system PATH (installed globally)
    import shutil as _shutil
    sys_vina = _shutil.which("vina")
    if sys_vina:
        return sys_vina

    return custom_path or bin_cpu  # Return expected path even if missing (pre-flight will catch it)

def resolve_vina_gpu_path(custom_path=None) -> str:
    """Compatibility alias for resolve_vina_cpu_path while GPU docking is disabled."""
    return resolve_vina_cpu_path(custom_path)

# ============================================================================
# WINDOWS HPC CONFIGURATION
# ============================================================================
def get_vina_gpu_config() -> dict:
    """Returns the default configuration.
    NOTE: Docking now runs on CPU AutoDock Vina (vina.exe). GPU docking is
    disabled until Vina-GPU-2.1 is rebuilt with matching CUDA kernel assets."""
    cpu_vina = resolve_vina_cpu_path()
    return {
        # ---- File Paths ----
        "receptor_path": os.path.join(project_root, "data", "raw", "drd2_clean.pdbqt"),
        "vina_cpu_executable": cpu_vina,   # CPU vina.exe (GPU disabled)
        "vina_gpu_executable": cpu_vina,   # Alias to avoid KeyError in any caller
        "obabel_path": resolve_obabel_path(),

        # ---- Parallelism ----
        "n_envs": 32,                  # Parallel CPU docking workers
        "vec_env": "subproc",

        # ---- Docking (CPU AutoDock Vina) ----
        "vina_exhaustiveness": 4,      # Reduced for CPU throughput (re-dock top hits at 32 post-training)
        "vina_cpu_threads": 1,         # 1 CPU core per worker to prevent Windows thread thrashing

        # ---- PPO Hyperparameters ----
        "n_steps": 512,                # 512 × 32 = 16,384 transitions per rollout
        "batch_size": 512,
        "n_epochs": 4,                 # Conservative 4 epochs to prevent policy drift
        "learning_rate": 5e-6,         # Stabilized LR to protect MolGPT grammar
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.04,              # Entropy bonus for exploration diversity
        "clip_range": 0.15,            # Tighter clipping for stability
        "max_grad_norm": 0.5,
        "target_kl": 0.03,             # Early-stop PPO epochs on KL divergence
        "max_length": 50,

        # ---- Training Duration ----
        "total_timesteps": 10_000_000,

        # ---- Model Architecture ----
        "unfreeze_molgpt": True,
        "vf_net_arch": [1024, 512, 256],

        # ---- Curriculum Learning ----
        # Phase 1 (warmup): Only QED-based rewards, no docking (runs ~50× faster)
        # Phase 2 (full):   Full CPU docking enabled once grammar is learned
        "curriculum_warmup_episodes": 5000,

        # ---- Top-K Experience Replay ----
        "topk_buffer_size": 100,
        "topk_similarity_bonus": 1.0,
        "topk_sim_low": 0.3,
        "topk_sim_high": 0.7,

        # ---- Docking Result Cache ----
        "docking_cache_size": 10000,   # LRU cache: canonical SMILES → docking score

        # ---- Checkpointing & Logging ----
        "checkpoint_freq": 10_000,
        "log_summary_freq": 50,
        "checkpoint_dir": os.path.join(project_root, "checkpoints", "hpc_cpu_run"),
        "log_dir": os.path.join(project_root, "logs", "hpc_cpu_run"),
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

    Fingerprints are stored as compact numpy byte arrays for efficient
    cross-process IPC serialization (replaces slow character-by-character
    UTF-8 bit-string encoding).
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


# ============================================================================
# SHARED DOCKING RESULT CACHE  (cross-process via Manager server)
# ============================================================================
class _DockingCacheServer:
    """
    Cross-process LRU cache keyed by canonical SMILES → docking score.
    Avoids redundant Vina-GPU calls when the same molecule is generated
    by different workers or across different episodes.
    """
    def __init__(self, maxlen: int = 10000):
        self._maxlen = maxlen
        self._cache = {}       # smiles → score
        self._order = deque()  # LRU eviction order
        self._lock = multiprocessing.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, smiles: str):
        """Return cached docking score, or None if not cached."""
        with self._lock:
            if smiles in self._cache:
                self._hits += 1
                return self._cache[smiles]
            self._misses += 1
            return None

    def put(self, smiles: str, score) -> None:
        """Store a docking result. Evicts oldest entry if at capacity."""
        with self._lock:
            if smiles not in self._cache:
                if len(self._cache) >= self._maxlen:
                    oldest = self._order.popleft()
                    self._cache.pop(oldest, None)
                self._order.append(smiles)
            self._cache[smiles] = score

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(total, 1),
                "size": len(self._cache),
            }


# ============================================================================
# SHARED TOP-K EXPERIENCE REPLAY BUFFER  (cross-process via Manager server)
# ============================================================================
class _TopKReplayServer:
    """
    Maintains the top-K best molecules (by total_reward) discovered across
    all workers. Workers query this buffer for scaffold similarity bonuses
    to guide exploration toward high-affinity chemical neighborhoods.
    """
    def __init__(self, k: int = 100):
        self._k = k
        self._buffer = []  # list of (reward, smiles, fp_bytes)
        self._lock = multiprocessing.Lock()

    def add(self, reward: float, smiles: str, fp_bytes: bytes) -> None:
        """Add a candidate. Buffer auto-sorts and truncates to top-K."""
        with self._lock:
            # Avoid exact duplicates
            existing_smiles = {s for _, s, _ in self._buffer}
            if smiles in existing_smiles:
                return
            self._buffer.append((reward, smiles, fp_bytes))
            self._buffer.sort(key=lambda x: x[0], reverse=True)
            self._buffer = self._buffer[:self._k]

    def get_fps(self) -> list:
        """Return fingerprint bytes for all top-K molecules."""
        with self._lock:
            return [fp_b for _, _, fp_b in self._buffer]

    def get_best(self, n: int = 5) -> list:
        """Return the top-n (reward, smiles) tuples."""
        with self._lock:
            return [(r, s) for r, s, _ in self._buffer[:n]]

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)


class SharedFingerprintManager(BaseManager):
    """Custom Manager that exposes shared servers to subprocesses."""
    pass

SharedFingerprintManager.register('FingerprintBuffer', _FingerprintBufferServer)
SharedFingerprintManager.register('DockingCache', _DockingCacheServer)
SharedFingerprintManager.register('TopKReplay', _TopKReplayServer)


def create_shared_servers(fp_maxlen: int = 3200, cache_maxlen: int = 10000, topk_size: int = 100):
    """
    Call ONCE in the main process before SubprocVecEnv is created.
    Starts a single Manager server hosting three shared services:
      1. FingerprintBuffer — Tanimoto diversity enforcement
      2. DockingCache      — LRU cache for Vina-GPU results
      3. TopKReplay        — Top-K best molecules for scaffold bonus

    Returns (manager, fp_proxy, cache_proxy, topk_proxy).
    The caller must keep `manager` alive for the entire duration of training.
    """
    mgr = SharedFingerprintManager()
    mgr.start()
    fp_proxy = mgr.FingerprintBuffer(maxlen=fp_maxlen)
    cache_proxy = mgr.DockingCache(maxlen=cache_maxlen)
    topk_proxy = mgr.TopKReplay(k=topk_size)
    return mgr, fp_proxy, cache_proxy, topk_proxy


def _fp_to_bytes(fp) -> bytes:
    """Serialize an RDKit ExplicitBitVect to a compact numpy byte array.
    
    ~8× smaller than UTF-8 bit-strings and avoids O(2048) Python-loop
    deserialization per fingerprint.
    """
    arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tobytes()


def _bytes_to_fp(b: bytes, nbits: int = 2048):
    """Deserialize a numpy byte array back to an RDKit ExplicitBitVect."""
    arr = np.frombuffer(b, dtype=np.uint8)
    fp = DataStructs.ExplicitBitVect(nbits)
    for idx in np.nonzero(arr)[0]:
        fp.SetBit(int(idx))
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
        docking_cache=None,          # Manager proxy — shared docking result cache
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

        # Shared docking result cache — avoids redundant Vina-GPU calls
        # when the same canonical SMILES is generated by different workers.
        self.docking_cache = docking_cache

        # IPC health monitoring — tracks consecutive failures so we can
        # warn the user before diversity silently collapses.
        self._ipc_failure_count = 0
        self._ipc_failure_warn_threshold = 50
        self._logger = logging.getLogger(self.__class__.__name__)

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
                # Reset failure counter on success
                self._ipc_failure_count = 0
            except Exception as e:
                # IPC failure safety: skip penalty rather than crash worker,
                # but track failures so we can warn before diversity collapses.
                self._ipc_failure_count += 1
                if self._ipc_failure_count == self._ipc_failure_warn_threshold:
                    self._logger.warning(
                        f"[RewardOracle] Shared fingerprint buffer has failed "
                        f"{self._ipc_failure_count} consecutive times! "
                        f"Diversity penalty is effectively DISABLED. "
                        f"Last error: {e}"
                    )
                elif self._ipc_failure_count % 500 == 0:
                    self._logger.error(
                        f"[RewardOracle] IPC failure count: {self._ipc_failure_count}. "
                        f"Shared buffer may be dead. Consider restarting training."
                    )
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

        # 3D Docking Evaluation — check shared cache first to avoid
        # redundant Vina-GPU subprocess calls for previously docked molecules.
        canonical = smiles  # Already canonicalized by MolToSmiles in step()
        docking_score = None
        cache_hit = False

        if self.docking_cache is not None:
            try:
                cached = self.docking_cache.get(canonical)
                if cached is not None:
                    docking_score = cached
                    cache_hit = True
            except Exception:
                pass  # IPC failure — fall through to actual docking

        if not cache_hit:
            docking_score = self._run_vina_gpu_docking(mol)
            # Store result in shared cache for other workers
            if self.docking_cache is not None:
                try:
                    self.docking_cache.put(canonical, docking_score)
                except Exception:
                    pass

        if docking_score is not None and docking_score < 0:
            # Successful docking with a negative (favorable) affinity
            total_reward = abs(docking_score) + (qed_score * 1.5) - penalty
        elif docking_score is None:
            # Docking failed entirely (3D embedding, OpenBabel, Vina timeout, etc.)
            # Penalize more than a valid-but-weak dock, but less than invalid SMILES (-5.0)
            total_reward = -3.0 + (qed_score * 0.5) - penalty
        else:
            # Docking returned 0 or positive (very weak / no binding)
            total_reward = -2.0 + (qed_score * 1.5) - penalty

        return {
            "valid": True, "qed": qed_score, "mw": mw,
            "docking_score": docking_score, "total_reward": total_reward,
            "max_tanimoto": max_sim, "diversity_penalty": diversity_penalty_applied
        }

    # Regex for robustly parsing Vina/Vina-GPU docking output.
    # Matches lines like:  "   1     -8.123   0.000   0.000" (mode, affinity, dist1, dist2)
    _VINA_SCORE_RE = re.compile(r'^\s*1\s+(-?\d+\.\d+)', re.MULTILINE)

    def _run_vina_gpu_docking(self, mol) -> float:
        # Use a per-worker subdirectory keyed by PID to reduce NTFS
        # directory-metadata contention when 32 workers write concurrently.
        worker_temp_dir = os.path.join(project_root, "data", "temp", f"worker_{os.getpid()}")
        os.makedirs(worker_temp_dir, exist_ok=True)

        thread_id = uuid.uuid4().hex
        
        # Vina-GPU requires a folder containing ligand(s)
        ligand_dir = os.path.join(worker_temp_dir, f"ligand_{thread_id}")
        os.makedirs(ligand_dir, exist_ok=True)
        
        temp_sdf = os.path.join(worker_temp_dir, f"temp_{thread_id}.sdf")
        temp_pdbqt = os.path.join(ligand_dir, "ligand.pdbqt")
        config_file = os.path.join(worker_temp_dir, f"config_{thread_id}.txt")
        
        try:
            mol_3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) != 0:
                return None   # Distinct from 0.0 — signals "undockable" to caller
            
            try:
                AllChem.UFFOptimizeMolecule(mol_3d)
            except Exception:
                pass  # Use unoptimized 3D coords from EmbedMolecule (still valid)

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
            try:
                babel_res = subprocess.run(
                    babel_cmd, capture_output=True, text=True, env=env,
                    timeout=30,   # P0 fix: prevent OpenBabel hangs on pathological molecules
                )
            except subprocess.TimeoutExpired:
                self._logger.warning(f"[Docking] OpenBabel timed out (30s) for molecule, skipping.")
                return None
            
            if babel_res.returncode != 0 or not os.path.exists(temp_pdbqt) or os.path.getsize(temp_pdbqt) == 0:
                return None

            # Docking is CPU-only (AutoDock Vina). Vina-GPU is disabled until
            # the CUDA kernel assets are rebuilt from DeltaGroupNJUPT/Vina-GPU-2.1.
            # Write a standard AutoDock Vina CPU config file.
            with open(config_file, "w") as f:
                f.write(f"receptor = {self.receptor_path}\n")
                f.write(f"ligand = {temp_pdbqt}\n")
                f.write("center_x = 9.5\n")
                f.write("center_y = 5.2\n")
                f.write("center_z = -11.4\n")
                f.write("size_x = 20.0\n")
                f.write("size_y = 20.0\n")
                f.write("size_z = 20.0\n")
                f.write(f"exhaustiveness = {self.exhaustiveness}\n")
                f.write("cpu = 1\n")  # 1 core per worker prevents Windows thread thrashing

            vina_cmd = [
                self.vina_gpu_executable,
                f"--config={config_file}"
            ]

            try:
                result = subprocess.run(
                    vina_cmd, capture_output=True, text=True,
                    timeout=120,  # P0 fix: prevent Vina-GPU hangs on pathological ligands
                )
            except subprocess.TimeoutExpired:
                self._logger.warning(f"[Docking] Vina-GPU timed out (120s) for molecule, skipping.")
                return None

            if result.returncode != 0:
                # Detect fatal Windows native crash codes (not normal Vina errors).
                # 0xC0000409 = STATUS_STACK_BUFFER_OVERRUN: the compiled Vina-GPU.exe
                # binary or its OpenCL kernels are corrupt / mismatched with the driver.
                # Treating this as a silent docking failure masks a critical build issue.
                FATAL_WIN32_CRASH_CODES = {
                    3221226505,  # 0xC0000409 STATUS_STACK_BUFFER_OVERRUN (bad kernel build)
                    3221225477,  # 0xC0000005 ACCESS_VIOLATION
                    3221225725,  # 0xC000009D STATUS_DEVICE_NOT_CONNECTED
                }
                if result.returncode in FATAL_WIN32_CRASH_CODES or result.returncode < -1:
                    if not hasattr(self, '_gpu_crash_count'):
                        self._gpu_crash_count = 0
                    self._gpu_crash_count += 1
                    if self._gpu_crash_count == 1:
                        self._logger.critical(
                            f"\n{'='*70}\n"
                            f"[CRITICAL] Vina-GPU.exe crashed with fatal Windows error code: "
                            f"{result.returncode} (0x{result.returncode & 0xFFFFFFFF:08X}).\n"
                            f"This means the compiled Vina-GPU binary or its OpenCL kernels are "
                            f"INCOMPATIBLE with this GPU/driver. This is NOT a Python error.\n"
                            f"ACTION REQUIRED: Rebuild Vina-GPU from the correct source:\n"
                            f"  git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git\n"
                            f"See vscode_agent_hpc_instructions.md Phase 1 for full instructions.\n"
                            f"Automatically falling back to CPU Vina for this worker.\n"
                            f"{'='*70}"
                        )
                    # After 3 crashes, permanently switch this oracle instance to CPU mode
                    if self._gpu_crash_count >= 3:
                        cpu_vina = os.path.join(project_root, "bin", "vina.exe")
                        if os.path.isfile(cpu_vina):
                            self._logger.warning(
                                f"[Docking] GPU crashed {self._gpu_crash_count}x. "
                                f"Permanently switching worker to CPU Vina: {cpu_vina}"
                            )
                            self.vina_gpu_executable = cpu_vina
                return None

            # Parse docking score using robust regex
            # Matches the first mode line: "   1     -8.123   ..."
            match = self._VINA_SCORE_RE.search(result.stdout)
            if match:
                return float(match.group(1))

        except Exception as e:
            return None
        finally:
            # Clean up this docking run's temp files
            for f_path in [temp_sdf, config_file]:
                if os.path.exists(f_path):
                    try: os.remove(f_path)
                    except OSError: pass
            if os.path.exists(ligand_dir):
                try: shutil.rmtree(ligand_dir)
                except OSError: pass
            # Periodic orphan cleanup: every ~100 docking calls, sweep the
            # worker temp dir for stale files older than 10 minutes.
            try:
                if hasattr(self, '_cleanup_counter'):
                    self._cleanup_counter += 1
                else:
                    self._cleanup_counter = 0
                if self._cleanup_counter % 100 == 0:
                    cutoff = time.time() - 600  # 10 minutes
                    for entry in os.scandir(worker_temp_dir):
                        try:
                            if entry.stat().st_mtime < cutoff:
                                if entry.is_dir():
                                    shutil.rmtree(entry.path)
                                else:
                                    os.remove(entry.path)
                        except OSError:
                            pass
            except Exception:
                pass
                
        return None

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
        docking_cache=None,           # Manager proxy — shared docking result cache
        topk_replay=None,             # Manager proxy — Top-K replay buffer
        curriculum_warmup_episodes=5000,  # Episodes before enabling docking
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
            vina_gpu_executable=vina_gpu_executable or VINA_GPU_CONFIG["vina_cpu_executable"],
            exhaustiveness=vina_exhaustiveness,
            obabel_path=obabel_path or VINA_GPU_CONFIG["obabel_path"],
            shared_buffer=shared_buffer,
            docking_cache=docking_cache,
        )
        self.episode_count = 0
        self.curriculum_warmup_episodes = curriculum_warmup_episodes
        self._warmup_logged = False

        # Top-K replay buffer proxy for scaffold similarity bonus
        self.topk_replay = topk_replay
        self._topk_sim_bonus = VINA_GPU_CONFIG.get("topk_similarity_bonus", 1.0)
        self._topk_sim_low = VINA_GPU_CONFIG.get("topk_sim_low", 0.3)
        self._topk_sim_high = VINA_GPU_CONFIG.get("topk_sim_high", 0.7)

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

                # ── Curriculum Learning ────────────────────────────────────
                # Phase 1 (warmup): Skip expensive Vina-GPU docking so the
                # policy can rapidly learn basic SMILES grammar via QED-only
                # rewards (~50× faster per episode).
                # Phase 2 (full): Enable docking once grammar is learned.
                use_docking = self.episode_count > self.curriculum_warmup_episodes
                if not self._warmup_logged and use_docking:
                    self._warmup_logged = True
                    print(f"[Worker {os.getpid()}] Curriculum warmup complete "
                          f"({self.curriculum_warmup_episodes} episodes). "
                          f"Docking is now ENABLED.")

                res = self.oracle.evaluate_smiles(valid_smiles, run_docking=use_docking)
                reward = res["total_reward"]

                info["smiles"] = valid_smiles
                info["valid"] = True
                info["qed"] = res.get("qed", 0.0)
                info["docking_score"] = res.get("docking_score", None)
                info["mw"] = res.get("mw", 0.0)
                info["curriculum_phase"] = "docking" if use_docking else "warmup"

                # Internal diversity check is managed by RewardOracleVinaGPU
                info["max_tanimoto"] = res.get("max_tanimoto", 0.0)
                info["diversity_penalty"] = (res.get("diversity_penalty", 0.0) > 0.0)

                # ── Top-K Scaffold Similarity Bonus ────────────────────────
                # Give a small reward bonus when the new molecule is
                # structurally similar (but not identical) to the best known
                # hits, guiding exploration toward high-affinity neighborhoods.
                topk_bonus = 0.0
                try:
                    if self.topk_replay is not None and reward > 0:
                        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        topk_fp_bytes = self.topk_replay.get_fps()
                        if topk_fp_bytes:
                            topk_fps = [_bytes_to_fp(b) for b in topk_fp_bytes]
                            sims = DataStructs.BulkTanimotoSimilarity(mol_fp, topk_fps)
                            max_topk_sim = float(max(sims))
                            if self._topk_sim_low < max_topk_sim < self._topk_sim_high:
                                topk_bonus = self._topk_sim_bonus
                                reward += topk_bonus
                        # Register this molecule if it's a strong hit
                        if reward > 2.0:
                            fp_bytes = _fp_to_bytes(mol_fp)
                            self.topk_replay.add(float(reward), valid_smiles, fp_bytes)
                except Exception:
                    pass  # IPC failure — skip bonus silently

                info["topk_bonus"] = topk_bonus
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

        # Build attention mask so GPT-2 does not attend to padding tokens.
        # Without this, hidden states for real tokens are contaminated by
        # attention to pad positions, degrading value function quality.
        attention_mask = (obs_long != self.pad_token_id).long()

        if VINA_GPU_CONFIG["unfreeze_molgpt"]:
            transformer_outputs = self.generator.model.transformer(
                input_ids=obs_long,
                attention_mask=attention_mask,
            )
        else:
            with torch.no_grad():
                transformer_outputs = self.generator.model.transformer(
                    input_ids=obs_long,
                    attention_mask=attention_mask,
                )

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
# LOGGING & CHECKPOINT CALLBACKS
# ============================================================================
class CleanCheckpointCallback(CheckpointCallback):
    """
    Saves a checkpoint every `save_freq` steps, but automatically deletes
    older checkpoints to prevent disk space exhaustion (Errno 28).
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", keep_last: int = 2, **kwargs):
        super().__init__(save_freq, save_path, name_prefix=name_prefix, **kwargs)
        self.keep_last = keep_last

    def _on_step(self) -> bool:
        # Save the checkpoint using the parent class
        result = super()._on_step()
        
        if self.n_calls % self.save_freq == 0:
            # Find and sort checkpoints
            pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
            files = glob.glob(pattern)
            # Sort by modification time (oldest first)
            files.sort(key=os.path.getmtime)
            
            # Delete older checkpoints, keeping only the last `keep_last`
            if len(files) > self.keep_last:
                for f in files[:-self.keep_last]:
                    try:
                        os.remove(f)
                    except OSError as e:
                        print(f"Warning: Could not delete old checkpoint {f}: {e}")
                        
        return result

# ============================================================================
# LOGGING CALLBACK — Multi-Day CSV + TensorBoard Tracking
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
                "curriculum_phase", "topk_bonus",
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
                    info.get("curriculum_phase", ""),
                    info.get("topk_bonus", 0.0),
                ])

            self.logger.record("molecules/validity_rate", validity_rate)
            self.logger.record("molecules/unique_count", len(self.unique_smiles))
            self.logger.record("molecules/total_episodes", self.total_episodes)
            self.logger.record("molecules/best_reward", self.best_reward)
            if docking is not None:
                self.logger.record("molecules/best_docking", self.best_docking)
            if "max_tanimoto" in info:
                self.logger.record("molecules/max_tanimoto", info["max_tanimoto"])
            # Curriculum & Top-K metrics
            phase = info.get("curriculum_phase", "warmup")
            self.logger.record("curriculum/phase", 1.0 if phase == "docking" else 0.0)
            self.logger.record("curriculum/topk_bonus", info.get("topk_bonus", 0.0))

            if self.total_episodes % self.summary_freq == 0:
                eta = self._estimate_eta(elapsed_h)
                best_dock_str = f"{self.best_docking:.2f} kcal/mol" if self.best_docking != float("inf") else "N/A"
                phase_str = "[DOCKING]" if phase == "docking" else "[WARMUP (QED-only)]"
                print(
                    f"\n{'='*70}\n"
                    f"  Episode {self.total_episodes:,}  |  Step {self.num_timesteps:,}  |  "
                    f"{elapsed_h:.1f} h elapsed  |  ETA: {eta}\n"
                    f"  Validity: {validity_rate:.1%}  |  Unique Molecules: {len(self.unique_smiles):,}  |  "
                    f"Best Dock: {best_dock_str}  |  Best Reward: {self.best_reward:.2f}\n"
                    f"  Top Candidate: {self.best_smiles}\n"
                    f"  Phase: {phase_str}\n"
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
            f"  HPC VINA-GPU TRAINING COMPLETE - Elapsed: {elapsed_h:.1f} hours\n"
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
    print("  WINDOWS HPC PRE-FLIGHT VERIFICATION (CPU DOCKING MODE)")
    print("=" * 70)

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [PASS] CUDA Available     : Yes ({device_name}, {vram_gb:.1f} GB VRAM) [PyTorch/PPO]")
    else:
        print("  [INFO] CUDA Available     : No (PyTorch/PPO running on CPU)")

    receptor = config.get("receptor_path")
    if receptor and os.path.isfile(receptor):
        size_kb = os.path.getsize(receptor) / 1024
        print(f"  [PASS] Receptor PDBQT     : Found ({receptor}, {size_kb:.1f} KB)")
    else:
        print(f"  [FAIL] Receptor PDBQT     : NOT FOUND at {receptor}")
        return False

    obabel = config.get("obabel_path")
    if obabel and os.path.isfile(obabel):
        print(f"  [PASS] OpenBabel Binary   : Found at {obabel}")
    else:
        print(f"  [FAIL] OpenBabel Binary   : NOT FOUND at {obabel}")
        return False

    vina_path = config.get("vina_cpu_executable") or config.get("vina_gpu_executable") or resolve_vina_cpu_path()
    if vina_path and os.path.isfile(vina_path):
        config["vina_cpu_executable"] = vina_path
        config["vina_gpu_executable"] = vina_path
        print(f"  [PASS] Docking Engine     : Found at {vina_path} [AutoDock Vina CPU]")
    else:
        # Check if CPU vina exists in bin as fallback
        cpu_vina = os.path.join(project_root, "bin", "vina.exe")
        if os.path.isfile(cpu_vina):
            config["vina_cpu_executable"] = cpu_vina
            config["vina_gpu_executable"] = cpu_vina
            print(f"  [PASS] Docking Engine     : Found at {cpu_vina} [AutoDock Vina CPU]")
        else:
            print(f"  [FAIL] Docking Engine     : NOT FOUND at {vina_path} or bin/vina.exe")
            print("         Please place vina.exe in the bin folder.")
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
    n_steps: int = None,
    batch_size: int = None,
    no_save: bool = False,
    summary_freq: int = None,
    vina_cpu_executable: str = None,
    vina_gpu_executable: str = None,
    obabel_path: str = None,
    receptor_path: str = None,
):
    config = VINA_GPU_CONFIG.copy()

    if n_envs is not None: config["n_envs"] = n_envs
    if vec_env is not None: config["vec_env"] = vec_env
    if total_timesteps is not None: config["total_timesteps"] = total_timesteps
    if n_steps is not None: config["n_steps"] = n_steps
    if batch_size is not None: config["batch_size"] = batch_size
    if summary_freq is not None: config["log_summary_freq"] = summary_freq
    chosen_vina = vina_cpu_executable or vina_gpu_executable
    if chosen_vina:
        config["vina_cpu_executable"] = os.path.abspath(chosen_vina)
        config["vina_gpu_executable"] = config["vina_cpu_executable"]
    if obabel_path: config["obabel_path"] = os.path.abspath(obabel_path)
    if receptor_path: config["receptor_path"] = os.path.abspath(receptor_path)

    if not run_preflight_checks(config):
        print("[ERROR] Pre-flight verification failed. Fix the missing paths above before running.")
        sys.exit(1)

    print("[HPC Setup] Pre-caching MolGPT tokenizer...")
    _ = AutoTokenizer.from_pretrained("msb-roshan/molgpt")

    # ── Shared IPC Servers ─────────────────────────────────────────────────────
    # Start a single Manager server process hosting three shared services:
    #   1. FingerprintBuffer — Tanimoto diversity enforcement across workers
    #   2. DockingCache      — LRU cache to avoid redundant Vina-GPU calls
    #   3. TopKReplay        — Top-K best molecules for scaffold similarity bonus
    shared_manager = None
    shared_fp_buffer = None
    shared_docking_cache = None
    shared_topk_replay = None

    if config["vec_env"] == "subproc":
        buffer_maxlen = config["n_envs"] * 100   # 100 fp slots per worker
        cache_maxlen = config.get("docking_cache_size", 10000)
        topk_size = config.get("topk_buffer_size", 100)
        print(f"[HPC Setup] Starting shared IPC servers...")
        print(f"  Fingerprint buffer : {buffer_maxlen} slots")
        print(f"  Docking cache      : {cache_maxlen} entries")
        print(f"  Top-K replay       : {topk_size} molecules")
        shared_manager, shared_fp_buffer, shared_docking_cache, shared_topk_replay = \
            create_shared_servers(fp_maxlen=buffer_maxlen, cache_maxlen=cache_maxlen, topk_size=topk_size)
        print("[HPC Setup] All shared servers online.")
    else:
        print("[HPC Setup] DummyVecEnv detected - shared servers not needed (single process).")
    # -------------------------------------------------------------------------

    warmup_eps = config.get("curriculum_warmup_episodes", 5000)
    print("=" * 70)
    print("  HPC PPO MOLECULAR TRAINING - CPU Vina Edition (GPU disabled)")
    print("=" * 70)
    print(f"  Workers         : {config['n_envs']} environments")
    print(f"  Rollout Buffer  : {config['n_steps'] * config['n_envs']:,} transitions")
    print(f"  Minibatch Size  : {config['batch_size']}")
    print(f"  PPO Epochs      : {config['n_epochs']}")
    print(f"  Learning Rate   : {config['learning_rate']}")
    print(f"  Target KL       : {config['target_kl']}")
    print(f"  Diversity Buffer: {'Shared (global)' if shared_fp_buffer else 'Local (per-worker)'}")
    print(f"  Docking Cache   : {'Shared (global)' if shared_docking_cache else 'Disabled'}")
    print(f"  Top-K Replay    : {'Shared (global)' if shared_topk_replay else 'Disabled'}")
    print(f"  Curriculum      : {warmup_eps} warmup episodes (QED-only) -> full docking")
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
            "vina_gpu_executable": config["vina_cpu_executable"],  # CPU vina.exe
            "obabel_path": config["obabel_path"],
            "shared_buffer": shared_fp_buffer,
            "docking_cache": shared_docking_cache,
            "topk_replay": shared_topk_replay,
            "curriculum_warmup_episodes": warmup_eps,
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
    ]
    if not no_save:
        callbacks.append(
            CleanCheckpointCallback(
                save_freq=max(config["checkpoint_freq"] // config["n_envs"], 1),
                save_path=config["checkpoint_dir"],
                name_prefix="ppo_molgpt_hpc",
                keep_last=2,
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )

    try:
        model.learn(total_timesteps=config["total_timesteps"], callback=callbacks, reset_num_timesteps=(resume_from is None))
    except KeyboardInterrupt:
        print("[HPC] Training interrupted by user. Saving checkpoint...")
    except Exception as e:
        print(f"[HPC] Training stopped due to error: {e}. Saving checkpoint...")
    finally:
        # Save model unless --no-save is active (prevents disk thrashing during smoke tests)
        if not no_save:
            try:
                model.save(os.path.join(config["checkpoint_dir"], "ppo_molgpt_hpc_final"))
                print(f"[HPC] Final model saved to {config['checkpoint_dir']}")
            except Exception as save_err:
                print(f"[HPC] WARNING: Could not save final model: {save_err}")
        else:
            print("[HPC] Skipping final model save (--no-save enabled).")
        env.close()
        # Cleanly shut down the Manager server process to release the IPC socket.
        if shared_manager is not None:
            print("[Cleanup] Shutting down shared fingerprint buffer server...")
            shared_manager.shutdown()
        # Clean up any remaining temp files from all workers
        temp_root = os.path.join(project_root, "data", "temp")
        if os.path.exists(temp_root):
            print(f"[Cleanup] Removing temp docking directory: {temp_root}")
            try:
                shutil.rmtree(temp_root)
            except OSError as e:
                print(f"[Cleanup] Warning: Could not fully clean temp dir: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Vina-GPU PPO Training")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--vec-env", type=str, choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-steps", type=int, default=None, help="PPO rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=None, help="PPO minibatch size")
    parser.add_argument("--no-save", action="store_true", help="Skip saving large checkpoint files (prevents disk thrashing during tests)")
    parser.add_argument("--summary-freq", type=int, default=None, help="Molecule summary frequency")
    parser.add_argument("--vina-path", type=str, default=None, help="Path to vina.exe (CPU docking)")
    parser.add_argument("--vina-gpu-path", type=str, default=None, help="Compatibility alias for --vina-path")
    parser.add_argument("--obabel-path", type=str, default=None)
    parser.add_argument("--receptor-path", type=str, default=None)

    args = parser.parse_args()
    chosen_vina = args.vina_path or args.vina_gpu_path
    train(
        resume_from=args.resume,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        no_save=args.no_save,
        summary_freq=args.summary_freq,
        vina_cpu_executable=chosen_vina,
        vina_gpu_executable=chosen_vina,
        obabel_path=args.obabel_path,
        receptor_path=args.receptor_path,
    )
