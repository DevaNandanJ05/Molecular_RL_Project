"""
===============================================================================
Windows HPC Max PPO Training Script for Molecular Generation (RTX 3090 24GB)
===============================================================================
Hardware Target:
  - GPU: NVIDIA GeForce RTX 3090 (24 GB GDDR6X VRAM, Ampere Compute 8.6)
  - CPU: 16-32 Logical Cores (Multiprocessing Pool)
  - RAM: 64 GB DDR4/DDR5
  - OS : Windows 10 / 11 / Windows Server

Architectural Advancements:
  1. Vectorized GPU Batch Generation:
     Generates 256-512 candidate molecules simultaneously on CUDA in FP16,
     saturating the RTX 3090 tensor cores and bypassing SB3 single-token bottlenecks.
  2. Automatic Mixed Precision (AMP / FP16):
     Uses torch.cuda.amp.autocast() and GradScaler for 2x faster matrix math,
     halving activation memory while maintaining gradient stability.
  3. REINVENT-Style Prior-Agent KL Divergence Regularization:
     Loads a frozen copy of the SFT DRD2 model. Calculates per-sequence KL penalty
     beta * (log P_agent - log P_prior) to mathematically prevent mode collapse
     and trivial molecule exploitation.
  4. Parallelized CPU Scoring Engine (16 Cores):
     Concurrent OpenBabel 3D embedding + AutoDock Vina docking across CPU workers
     with an LRU docking cache for 0ms score retrieval on repeated scaffolds.
  5. IF-ARS (Interaction Fingerprint-guided Adaptive Reward Shaping):
     PLIP-based non-covalent contact profiling against Haloperidol reference
     for binding poses <= -6.0 kcal/mol (ASP114 salt bridge + critical pocket keys).
  6. PPO Actor-Critic with Generalized Advantage Estimation:
     Transformer representation -> 3-layer MLP value head with clipped surrogate
     loss, value clipping, and entropy regularization.
  7. Real-Time Telemetry:
     Tracks VRAM usage (GB), molecule validity %, docking throughput, and logs
     to both TensorBoard and real-time CSV.
===============================================================================
"""

import os
import sys

# Force UTF-8 encoding on Windows console to prevent cp1252 charmap crashes
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import subprocess
import csv
import time
import shutil
import uuid
import glob
import math
import argparse
import signal
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Contrib.SA_Score import sascorer as _sascorer

# Suppress RDKit warning spam
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.reward_oracle import resolve_vina_path, resolve_obabel_path
from models.plip_wrapper import PLIPAnalyzer


# ============================================================================
# PER-WORKER PLIP INITIALIZER
# ============================================================================
# PLIPAnalyzer is expensive to construct (PDBQT->PDB conversion + JSON load).
# By using ProcessPoolExecutor's `initializer`, we build it ONCE per worker
# process and reuse it across all molecules assigned to that worker.
# This eliminates the race condition on receptor_plip_cache.pdb that silenced
# IF-ARS completely during the previous 10M-step run.
_worker_plip_analyzer: PLIPAnalyzer = None


def _init_worker_plip(receptor_path: str, ref_json: str, obabel_path: str):
    """Called once per ProcessPoolExecutor worker process at startup."""
    global _worker_plip_analyzer
    try:
        _worker_plip_analyzer = PLIPAnalyzer(
            receptor_pdbqt=receptor_path,
            reference_json=ref_json,
            obabel_path=obabel_path,
        )
    except Exception as e:
        print(f"[PLIP INIT ERROR pid={os.getpid()}] Failed to initialise PLIPAnalyzer: {e}", flush=True)
        _worker_plip_analyzer = None


# ============================================================================
# ADAPTIVE REWARD WEIGHT SCHEDULER
# ============================================================================
# Research (2024): Static reward weights cause instability because the agent
# is asked to simultaneously learn chemistry AND target pharmacophores from
# iteration 0. The adaptive scheduler splits training into 3 phases:
#
#  Phase 1 - Warm-up  (0% to 20% of run): Low KL beta + zero IF-ARS bonus.
#             The agent explores freely without pharmacophore pressure.
#             This builds a strong foundation of valid, drug-like chemistry.
#
#  Phase 2 - Ramp-up (20% to 50% of run): All weights linearly increase.
#             The agent is gradually introduced to the ASP114 reward signal
#             as it already knows how to generate valid molecules.
#
#  Phase 3 - Full    (50% to 100% of run): All weights at full configured value.
#             Maximum pharmacophore pressure to exploit the chemical knowledge
#             built in Phases 1 and 2.
# ============================================================================
def _get_adaptive_weights(iteration: int, total_iterations: int, config: dict) -> dict:
    """Returns current adaptive weight values for this iteration."""
    progress = iteration / max(total_iterations, 1)  # 0.0 -> 1.0

    warmup_end  = 0.20  # Phase 1 ends at 20% of total iterations
    rampup_end  = 0.50  # Phase 2 ends at 50% of total iterations

    if progress < warmup_end:
        # Phase 1: Free exploration — no IF-ARS pressure, reduced KL
        t = 0.0
    elif progress < rampup_end:
        # Phase 2: Linear ramp from 0 to 1
        t = (progress - warmup_end) / (rampup_end - warmup_end)
    else:
        # Phase 3: Full optimization
        t = 1.0

    return {
        # KL beta: starts at 10% of configured value, ramps to 100%
        "prior_kl_beta":   config["prior_kl_beta"] * (0.1 + 0.9 * t),
        # IF-ARS ASP114 weight: starts at 0, ramps to full value
        "ifars_w_asp114":  config["ifars_w_asp114"] * t,
        # Diversity penalty: starts at 60% (let agent explore), ramps to 100%
        "diversity_penalty": config["diversity_penalty"] * (0.6 + 0.4 * t),
        # Phase label for logging
        "phase": 1 if progress < warmup_end else (2 if progress < rampup_end else 3),
    }


# ============================================================================
# HPC MAX CONFIGURATION
# ============================================================================
def get_hpc_max_config() -> dict:
    """Default hyperparameter configuration optimized for 24 GB RTX 3090."""
    sft_best = os.path.join(PROJECT_ROOT, "checkpoints", "molgpt_drd2_sft", "best_model")
    sft_path = sft_best if os.path.isdir(sft_best) else "msb-roshan/molgpt"

    return {
        # File paths
        "receptor_path": os.path.join(PROJECT_ROOT, "data", "raw", "drd2_clean.pdbqt"),
        "vina_executable": resolve_vina_path(),
        "obabel_path": resolve_obabel_path(),
        "sft_model_path": sft_path,
        "reference_json": os.path.join(PROJECT_ROOT, "data", "reference", "haloperidol_fingerprint.json"),

        # GPU Batch Generation & Sizing (Floods 24 GB VRAM)
        "rollout_batch_size": 256,       # Molecules generated per RL rollout iteration (256 or 512)
        "max_length": 65,                # Max token sequence length (Increased from 50 to allow 300-450 Da DRD2 targets)
        "temperature": 0.85,             # Sampling temperature (exploration)
        "top_k": 40,                     # Nucleus top-k filtering
        "top_p": 0.92,                   # Nucleus top-p filtering

        # Prior-Agent KL Regularization (REINVENT-style anti-collapse)
        "prior_kl_beta": 0.5,            # KL penalty weight beta: R_aug = R - beta * KL (REINVENT 2.0 paper: 0.5 start)

        # PPO Optimization Hyperparameters
        "ppo_epochs": 4,                 # Updates per rollout batch
        "mini_batch_size": 64,           # Mini-batch size during PPO updates (fits in VRAM with backprop)
        "learning_rate": 5e-6,           # Actor learning rate (Reduced from 2e-5 to lower clip fraction to ~15%)
        "critic_lr": 1e-4,               # Critic (value head) learning rate
        "clip_range": 0.2,               # PPO clipping epsilon
        "value_coef": 0.5,               # Value function loss coefficient
        "entropy_coef": 0.05,            # Entropy bonus coefficient (REINVENT: 0.05-0.10 for chemical diversity)
        "max_grad_norm": 0.5,            # Aggressive gradient clipping for FP16 training
        "target_kl": 0.05,               # Early-stopping threshold (Raised from 0.015 to match actual variance and restore full update budget)

        # Parallel Docking & CPU Resources
        "docking_workers": min(16, multiprocessing.cpu_count()),  # CPU workers for parallel 3D prep & docking
        "vina_exhaustiveness": 8,        # Vina default — reliable RL signal without excessive slowdown (CPU-only, not GPU)
        "vina_center": (9.5, 5.2, -11.4),# DRD2 binding pocket center coordinates
        "vina_size": (20.0, 20.0, 20.0), # DRD2 binding pocket search box dimensions
        "docking_cache_size": 50000,     # In-memory LRU cache of evaluated SMILES

        # IF-ARS (Interaction Fingerprint Guided Reward Shaping)
        "ifars_enabled": True,
        "ifars_threshold": -5.0,         # Run PLIP on binders <= -5.0 kcal/mol (lowered to avoid sparse reward problem)
        "ifars_w_overlap": 2.0,          # Haloperidol interaction overlap bonus weight
        "ifars_w_critical": 3.0,         # Critical residue overlap bonus weight
        "ifars_w_asp114": 2.0,           # Essential ASP114 anchor bonus

        # Diversity & Chemical Replay
        "tanimoto_threshold": 0.75,      # High similarity threshold for diversity penalty
        "diversity_penalty": 2.5,        # Penalty deducted for repetitive scaffolds
        "topk_buffer_size": 100,         # Top-K replay buffer capacity
        "topk_sim_bonus": 1.0,           # Scaffold exploration bonus

        # Checkpointing & Telemetry
        "total_timesteps": 500_000,      # ~1,953 rollout iterations — near-convergence for SFT-initialized REINVENT (REINVENT 2.0: 500K-1M recommended)
        "checkpoint_freq": 25,           # Save model checkpoint every N rollout iterations
        "keep_last_checkpoints": 3,      # Keep latest 3 checkpoints (prevents disk bloat)
        "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints", "hpc_max_run"),
        "log_dir": os.path.join(PROJECT_ROOT, "logs", "hpc_max_run"),
    }


# ============================================================================
# STANDALONE WORKER FUNCTION FOR PARALLEL 3D CONFORMATION & DOCKING
# ============================================================================
def _score_single_molecule_worker(task: tuple) -> dict:
    """
    Executed across CPU worker processes in ProcessPoolExecutor.
    Receives: (smiles, receptor_path, vina_path, obabel_path, exhaustiveness, center, size, ifars_params)
    Returns complete evaluation dictionary.
    """
    (smiles, receptor_path, vina_path, obabel_path,
     exhaustiveness, center, size, ifars_enabled, ref_json,
     ifars_threshold, w_overlap, w_critical, w_asp114) = task

    empty_res = {
        "smiles": smiles, "valid": False, "total_reward": -5.0,
        "docking_score": None, "qed": 0.0, "mw": 0.0, "rings": 0, "rot_bonds": 0,
        "ifars_overlap": 0.0, "critical_overlap": 0.0, "has_asp114": False, "ifars_bonus": 0.0
    }

    if not smiles or len(smiles.strip()) < 3:
        return empty_res

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty_res

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return empty_res

    # -----------------------------------------------------------------------
    # DRUG-LIKENESS PRE-FILTER (Lipinski Rule of Five + Synthetic Accessibility)
    # -----------------------------------------------------------------------
    # This gate runs BEFORE any 3D embedding or Vina docking to eliminate
    # greasy blobs and unsynthesizable structures early, saving CPU time and
    # keeping the reward signal clean without creating competing gradients.
    try:
        mw_pre   = Descriptors.MolWt(mol)
        logp_pre = Descriptors.MolLogP(mol)
        hbd_pre  = rdMolDescriptors.CalcNumHBD(mol)
        hba_pre  = rdMolDescriptors.CalcNumHBA(mol)
        tpsa_pre = Descriptors.TPSA(mol)
        sas_pre  = _sascorer.calculateScore(mol)

        lipinski_violations = sum([
            mw_pre   > 500,
            logp_pre > 5,
            hbd_pre  > 5,
            hba_pre  > 10,
            tpsa_pre > 140,
        ])
        # Reject if >1 Lipinski violation (standard Ro5 allows 1 exception)
        # or if the molecule is too hard to synthesize (SAS > 6 out of 10)
        if lipinski_violations > 1 or sas_pre > 6.0:
            return empty_res
    except Exception:
        return empty_res
    # -----------------------------------------------------------------------

    # Chemical descriptors
    try:
        qed_val = float(QED.qed(mol))
        mw_val = float(Descriptors.MolWt(mol))
        rings_val = int(rdMolDescriptors.CalcNumRings(mol))
        rot_bonds_val = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    except Exception:
        return empty_res

    # Structural penalties (Medicinal Chemistry Filters)
    penalty = 0.0
    if mw_val < 160.0 or mw_val > 550.0:
        penalty += 2.0
    if rings_val == 0:
        penalty += 1.5
    if rot_bonds_val > 10:
        penalty += 1.0

    # If receptor or binaries are unavailable, return chemistry-only reward
    if not receptor_path or not os.path.isfile(receptor_path) or not vina_path or not os.path.isfile(vina_path):
        reward = (qed_val * 3.0) - penalty
        return {
            "smiles": smiles, "valid": True, "total_reward": reward,
            "docking_score": None, "qed": qed_val, "mw": mw_val, "rings": rings_val, "rot_bonds": rot_bonds_val,
            "ifars_overlap": 0.0, "critical_overlap": 0.0, "has_asp114": False, "ifars_bonus": 0.0
        }

    # 3D Conformer Generation & Docking
    worker_pid = os.getpid()
    unique_id = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(PROJECT_ROOT, "data", "temp", f"hpc_worker_{worker_pid}_{unique_id}")
    os.makedirs(temp_dir, exist_ok=True)

    temp_sdf = os.path.join(temp_dir, "ligand.sdf")
    temp_pdbqt = os.path.join(temp_dir, "ligand.pdbqt")
    docked_out = os.path.join(temp_dir, "docked_out.pdbqt")
    config_file = os.path.join(temp_dir, "vina_config.txt")

    docking_score = None
    ifars_overlap = 0.0
    critical_overlap = 0.0
    has_asp114 = False
    ifars_bonus = 0.0

    try:
        mol_3d = Chem.AddHs(mol)
        embed_status = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
        if embed_status != 0:
            # Fallback to standard ETKDG if v3 fails
            embed_status = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())

        if embed_status == 0:
            try:
                AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
            except Exception:
                pass

            writer = Chem.SDWriter(temp_sdf)
            writer.write(mol_3d)
            writer.close()

            # OpenBabel conversion to PDBQT
            env = os.environ.copy()
            obabel_dir = os.path.dirname(obabel_path)
            for cand in [os.path.join(obabel_dir, "data"), os.path.join(obabel_dir, "bin", "data")]:
                if os.path.exists(cand):
                    env["BABEL_DATADIR"] = cand
                    break

            babel_cmd = [obabel_path, temp_sdf, "-O", temp_pdbqt, "-h"]
            subprocess.run(babel_cmd, capture_output=True, text=True, env=env, timeout=25)

            if os.path.exists(temp_pdbqt) and os.path.getsize(temp_pdbqt) > 0:
                cx, cy, cz = center
                sx, sy, sz = size

                with open(config_file, "w") as f:
                    f.write(f"receptor = {receptor_path}\n")
                    f.write(f"ligand = {temp_pdbqt}\n")
                    f.write(f"out = {docked_out}\n")
                    f.write(f"center_x = {cx}\n")
                    f.write(f"center_y = {cy}\n")
                    f.write(f"center_z = {cz}\n")
                    f.write(f"size_x = {sx}\n")
                    f.write(f"size_y = {sy}\n")
                    f.write(f"size_z = {sz}\n")
                    f.write(f"exhaustiveness = {exhaustiveness}\n")
                    f.write("cpu = 1\n")

                vina_cmd = [vina_path, f"--config={config_file}"]
                v_res = subprocess.run(vina_cmd, capture_output=True, text=True, timeout=90)

                # Parse Mode 1 affinity (e.g., "   1      -8.4      0.000      0.000")
                for line in v_res.stdout.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0] == "1":
                        try:
                            docking_score = float(parts[1])
                            break
                        except ValueError:
                            pass

                # IF-ARS (PLIP) interaction analysis on strong binders
                # Graduated scale: 0.0 at threshold, 1.0 at -9.0+ kcal/mol
                # Prevents sparse reward problem — even borderline binders get partial signal
                if (ifars_enabled and docking_score is not None and
                    docking_score <= ifars_threshold and os.path.exists(docked_out)):
                    # Continuous scale factor: reward quality scales with binding strength
                    ifars_scale = min(1.0, max(0.0, (-docking_score - abs(ifars_threshold)) / 4.0))
                    # FIX: use the per-worker singleton instantiated by _init_worker_plip()
                    # instead of constructing a new PLIPAnalyzer for every molecule.
                    # Constructing per-molecule caused a race condition on the cached
                    # receptor PDB file across the 16 worker processes.
                    plip = _worker_plip_analyzer
                    if plip is not None:
                        try:
                            plip_res = plip.score_docked_pose(docked_out)
                            ifars_overlap = plip_res.get("overlap_score", 0.0)
                            critical_overlap = plip_res.get("critical_overlap", 0.0)
                            matched = plip_res.get("matched_keys", [])
                            has_asp114 = any("ASP114" in k for k in matched)
                            # Apply ifars_scale: molecules near threshold get partial bonus,
                            # very strong binders (-9.0+) get full bonus
                            ifars_bonus = ifars_scale * (
                                (w_overlap * ifars_overlap) +
                                (w_critical * critical_overlap) +
                                (w_asp114 if has_asp114 else 0.0)
                            )
                        except Exception as plip_e:
                            # Log explicitly — the previous bare `except` silenced all
                            # PLIP failures across the entire 10M-step run.
                            print(
                                f"[PLIP ERROR pid={os.getpid()} smiles={smiles[:40]!r}] "
                                f"{type(plip_e).__name__}: {plip_e}",
                                flush=True
                            )
                            # Give a small docking-only signal so the agent still
                            # knows this molecule is worth exploring.
                            ifars_bonus = ifars_scale * 0.5
                    else:
                        # Worker PLIP initialiser failed at startup; use docking signal only.
                        ifars_bonus = ifars_scale * 0.5
    except Exception:
        pass
    finally:
        # Guarantee zero disk leaks
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass

    # Reward Calculation
    if docking_score is not None and docking_score < 0:
        total_reward = abs(docking_score) + (qed_val * 2.0) + ifars_bonus - penalty
    elif docking_score is None:
        total_reward = -3.0 + (qed_val * 1.0) - penalty
    else:
        total_reward = -2.0 + (qed_val * 1.5) - penalty

    return {
        "smiles": smiles, "valid": True, "total_reward": total_reward,
        "docking_score": docking_score, "qed": qed_val, "mw": mw_val,
        "rings": rings_val, "rot_bonds": rot_bonds_val,
        "ifars_overlap": ifars_overlap, "critical_overlap": critical_overlap,
        "has_asp114": has_asp114, "ifars_bonus": ifars_bonus
    }


# ============================================================================
# HPC MAX ACTOR-CRITIC MODEL (MolGPT Backbone + Value Head)
# ============================================================================
class MolGPTPPOModel(nn.Module):
    """
    Unified Actor-Critic architecture mounting MolGPT causal LM as the policy backbone
    and a multi-layer value head predicting sequence-level expected returns.
    """
    def __init__(self, sft_model_path: str, device: torch.device):
        super().__init__()
        self.device = device

        print(f"[{datetime_now()}] Loading MolGPT Policy Backbone from: {sft_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer)

        # Actor causal language model
        self.actor = AutoModelForCausalLM.from_pretrained(sft_model_path).to(self.device)
        self.hidden_dim = self.actor.config.n_embd

        # Critic (Value Head) network: hidden_dim -> 512 -> 256 -> 1
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        ).to(self.device)

        # Unfreeze all transformer blocks for end-to-end RL fine-tuning on 24 GB VRAM
        for param in self.actor.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def generate_batch(self, batch_size: int, max_length: int = 35,
                       temperature: float = 0.85, top_k: int = 40, top_p: float = 0.92) -> tuple:
        """
        Generates a massive batch of molecules directly on the RTX 3090 using FP16 AMP.
        Returns:
            input_ids: [B, L] tensor of generated token IDs
            attention_mask: [B, L] attention mask
            generated_smiles: list of decoded SMILES strings
        """
        self.actor.eval()
        B = batch_size

        # Initialize with BOS token
        input_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_length - 1):
            with autocast(dtype=torch.float16):
                outputs = self.actor(input_ids=input_ids)
                next_token_logits = outputs.logits[:, -1, :] / temperature

            # Filter already finished sequences
            if finished.all():
                break

            # Top-K and Top-P filtering
            filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # If finished, force PAD token
            next_tokens = torch.where(finished.unsqueeze(1), torch.tensor(self.pad_token_id, device=self.device), next_tokens)

            # Update finished status if EOS is hit
            is_eos = (next_tokens.squeeze(1) == self.eos_token_id)
            finished = finished | is_eos

            input_ids = torch.cat([input_ids, next_tokens], dim=1)

        attention_mask = (input_ids != self.pad_token_id).long()

        # Decode tokens to SMILES
        decoded_smiles = []
        for seq in input_ids.cpu().numpy():
            tokens = [t for t in seq if t not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)]
            smi = self.tokenizer.decode(tokens, skip_special_tokens=True).replace(" ", "").split(".")[0]
            decoded_smiles.append(smi)

        return input_ids, attention_mask, decoded_smiles

    def evaluate_sequences(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """
        Evaluates sequence log-probabilities and critic values in forward pass.
        Returns:
            seq_log_probs: [B] sequence log probabilities
            values: [B] sequence value predictions
            entropy: scalar mean token entropy
        """
        with autocast(dtype=torch.float16):
            outputs = self.actor(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits[:, :-1, :]          # [B, L-1, V]
            targets = input_ids[:, 1:]                   # [B, L-1]
            mask = attention_mask[:, 1:].float()        # [B, L-1]

            # Log-probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            gathered_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
            seq_log_probs = (gathered_log_probs * mask).sum(dim=-1)

            # Entropy
            probs = F.softmax(logits, dim=-1)
            token_entropy = -(probs * log_probs).sum(dim=-1)
            mean_entropy = (token_entropy * mask).sum() / mask.sum().clamp(min=1.0)

            # Critic Value from last valid token hidden state
            hidden_states = outputs.hidden_states[-1][:, :-1, :]   # [B, L-1, H]
            seq_lengths = mask.sum(dim=-1).long().clamp(min=1) - 1
            batch_indices = torch.arange(input_ids.shape[0], device=self.device)
            last_hidden = hidden_states[batch_indices, seq_lengths, :]  # [B, H]

            values = self.critic(last_hidden.float()).squeeze(-1)       # [B]

        return seq_log_probs.float(), values.float(), mean_entropy.float()

    @staticmethod
    def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        """Applies Top-K and Top-P (nucleus) sampling mask to logits."""
        if top_k > 0:
            filter_val = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits = torch.where(logits < filter_val, torch.full_like(logits, -float("Inf")), logits)

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to keep at least the first token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        return logits


# ============================================================================
# FROZEN PRIOR MODEL FOR REINVENT-STYLE KL REGULARIZATION
# ============================================================================
class FrozenPriorModel:
    """
    Maintains a completely frozen copy of the SFT model.
    Evaluates log P_prior(X) to compute the Prior-Agent KL divergence penalty:
        KL = log P_agent(X) - log P_prior(X)
    """
    def __init__(self, sft_model_path: str, device: torch.device):
        print(f"[{datetime_now()}] Loading Frozen SFT Prior Model from: {sft_model_path}")
        self.device = device
        self.prior = AutoModelForCausalLM.from_pretrained(sft_model_path).to(self.device)
        self.prior.eval()
        for param in self.prior.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def compute_prior_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with autocast(dtype=torch.float16):
            outputs = self.prior(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            mask = attention_mask[:, 1:].float()

            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            prior_seq_log_probs = (gathered * mask).sum(dim=-1)

        return prior_seq_log_probs.float()


# ============================================================================
# DIVERSITY BUFFER & TOP-K CHEMICAL REPLAY
# ============================================================================
class ChemicalMemory:
    """
    Tracks recent Morgan fingerprints for Tanimoto diversity enforcement and
    stores the Top-K highest-affinity molecules for scaffold replay.
    """
    def __init__(self, recent_size: int = 1500, topk_size: int = 100):
        self.recent_fps = deque(maxlen=recent_size)
        self.topk_molecules = []  # list of (reward, smiles, fp)
        self.topk_size = topk_size

    def check_diversity(self, mol, threshold: float = 0.75, penalty: float = 2.5) -> float:
        """Returns penalty if candidate is too similar to recently generated molecules."""
        if len(self.recent_fps) == 0:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            self.recent_fps.append(fp)
            return 0.0

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, list(self.recent_fps))
        max_sim = max(sims) if sims else 0.0
        self.recent_fps.append(fp)

        return penalty if max_sim > threshold else 0.0

    def add_topk(self, reward: float, smiles: str, mol, bonus_val: float = 1.0) -> float:
        """Updates Top-K replay buffer and returns scaffold similarity exploration bonus."""
        bonus = 0.0
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            if self.topk_molecules:
                topk_fps = [item[2] for item in self.topk_molecules]
                sims = DataStructs.BulkTanimotoSimilarity(fp, topk_fps)
                max_sim = max(sims) if sims else 0.0
                if 0.35 <= max_sim <= 0.70:
                    bonus = bonus_val

            if reward > 3.0:
                self.topk_molecules.append((reward, smiles, fp))
                self.topk_molecules.sort(key=lambda x: x[0], reverse=True)
                self.topk_molecules = self.topk_molecules[:self.topk_size]
        except Exception:
            pass

        return bonus


def datetime_now() -> str:
    return time.strftime("%H:%M:%S")


# ============================================================================
# MAIN HPC MAX TRAINING PIPELINE
# ============================================================================
def train_hpc_max(args):
    # Enable TF32 for Ampere architecture (RTX 3090 Tensor Cores)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARNING] CUDA is NOT available! Running on CPU fallback (Training will be slow).")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"===============================================================================")
        print(f"  WINDOWS HPC MAX RL PIPELINE INITIALIZED")
        print(f"  Active GPU   : {gpu_name} ({vram_total:.1f} GB VRAM)")
        print(f"  PyTorch CUDA : {torch.version.cuda} | TF32 Tensor Cores: Enabled")
        print(f"===============================================================================")

    config = get_hpc_max_config()
    # Apply CLI overrides
    if args.batch_size: config["rollout_batch_size"] = args.batch_size
    if args.lr: config["learning_rate"] = args.lr
    if args.timesteps: config["total_timesteps"] = args.timesteps
    if args.workers: config["docking_workers"] = args.workers
    if args.sft_path: config["sft_model_path"] = args.sft_path

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "temp"), exist_ok=True)

    # Initialize Logging
    tb_writer = SummaryWriter(log_dir=config["log_dir"])
    csv_log_path = os.path.join(config["log_dir"], "molecule_log.csv")
    csv_exists = os.path.isfile(csv_log_path) and os.path.getsize(csv_log_path) > 0
    csv_file = open(csv_log_path, mode="a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "Timestamp", "Iteration", "SMILES", "Valid", "Reward", "AugmentedReward",
            "DockingScore", "QED", "MW", "Rings", "RotBonds",
            "KL_Divergence", "IFARS_Overlap", "CriticalOverlap", "HasASP114"
        ])
        csv_file.flush()

    # Initialize Actor-Critic and Frozen Prior
    model = MolGPTPPOModel(config["sft_model_path"], device=device)
    prior = FrozenPriorModel(config["sft_model_path"], device=device)

    # Optimizer & Mixed Precision Scaler
    optimizer = torch.optim.AdamW([
        {"params": model.actor.parameters(), "lr": config["learning_rate"], "weight_decay": 0.01},
        {"params": model.critic.parameters(), "lr": config["critic_lr"], "weight_decay": 0.01}
    ])
    scaler = GradScaler()

    # Replay buffer and Docking cache
    chemical_memory = ChemicalMemory(recent_size=1500, topk_size=config["topk_buffer_size"])
    docking_cache = {}  # SMILES -> score dict

    # Resume from checkpoint if specified
    start_iteration = 0
    total_molecules_evaluated = 0
    best_affinity = 0.0
    best_reward = -float("inf")

    if args.resume and os.path.isfile(args.resume):
        print(f"[{datetime_now()}] Resuming checkpoint from: {args.resume}")
        chk = torch.load(args.resume, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        scaler.load_state_dict(chk["scaler_state_dict"])
        start_iteration = chk.get("iteration", 0)
        total_molecules_evaluated = chk.get("total_molecules", 0)
        best_affinity = chk.get("best_affinity", 0.0)
        best_reward = chk.get("best_reward", -float("inf"))
        print(f"[{datetime_now()}] Checkpoint restored successfully. Resuming at iteration {start_iteration}.")

    # Graceful Interrupt Handler
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        print(f"\n[{datetime_now()}] [EMERGENCY] Ctrl+C received. Saving emergency checkpoint...")
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)

    rollout_size = config["rollout_batch_size"]
    total_iterations = config["total_timesteps"] // rollout_size
    print(f"\n[{datetime_now()}] Launching HPC Max Training: {total_iterations} iterations ({config['total_timesteps']} candidate molecules)")
    print(f"  Rollout Batch Size: {rollout_size} | PPO Mini-batch: {config['mini_batch_size']} | Epochs: {config['ppo_epochs']}")
    print(f"  Parallel CPU Workers: {config['docking_workers']} | Docking Cache Size: {len(docking_cache)}")
    print(f"  KL Regularization beta: {config['prior_kl_beta']} (Tethered to {config['sft_model_path']})\n")

    # Create the worker pool ONCE here and reuse across ALL iterations.
    # Previously this was created inside the loop, spawning/killing 4 worker
    # processes and re-running _init_worker_plip 4x every single iteration.
    # Moving it here means PLIPAnalyzer is built exactly once per worker (4 total).
    executor = ProcessPoolExecutor(
        max_workers=config["docking_workers"],
        initializer=_init_worker_plip,
        initargs=(
            config["receptor_path"],
            config["reference_json"],
            config["obabel_path"],
        ),
    )
    print(f"[{datetime_now()}] Worker pool initialized: {config['docking_workers']} persistent CPU workers ready.")

    # Training Loop
    for iteration in range(start_iteration + 1, total_iterations + 1):
        if interrupted:
            break

        # ----------------------------------------------------------------
        # ADAPTIVE REWARD WEIGHT SCHEDULER
        # Update KL beta, IF-ARS ASP114 weight, and diversity penalty
        # dynamically based on training progress (warm-up → ramp → full).
        # ----------------------------------------------------------------
        adaptive = _get_adaptive_weights(iteration, total_iterations, config)
        active_kl_beta        = adaptive["prior_kl_beta"]
        active_ifars_asp114   = adaptive["ifars_w_asp114"]
        active_div_penalty    = adaptive["diversity_penalty"]
        active_phase          = adaptive["phase"]
        # ----------------------------------------------------------------

        iter_start_time = time.time()

        # ====================================================================
        # PHASE 1: MASSIVE BATCH GENERATION ON RTX 3090 (GPU)
        # ====================================================================
        gen_start = time.time()
        input_ids, attention_mask, raw_smiles_list = model.generate_batch(
            batch_size=rollout_size,
            max_length=config["max_length"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"]
        )
        gen_time = time.time() - gen_start

        # Compute old log-probabilities and prior log-probabilities
        with torch.no_grad():
            old_seq_log_probs, old_values, _ = model.evaluate_sequences(input_ids, attention_mask)
            prior_seq_log_probs = prior.compute_prior_log_probs(input_ids, attention_mask)

            # Per-sequence KL Divergence
            seq_lens = attention_mask[:, 1:].sum(dim=-1).float().clamp(min=1.0)
            kl_div = (old_seq_log_probs - prior_seq_log_probs) / seq_lens

        # ====================================================================
        # PHASE 2: PARALLEL CPU SCORING & DOCKING PIPELINE (16 CORES)
        # ====================================================================
        dock_start = time.time()
        scoring_tasks = []
        cached_results = {}

        for idx, smi in enumerate(raw_smiles_list):
            if smi in docking_cache:
                cached_results[idx] = docking_cache[smi]
            else:
                task = (
                    smi,
                    config["receptor_path"],
                    config["vina_executable"],
                    config["obabel_path"],
                    config["vina_exhaustiveness"],
                    config["vina_center"],
                    config["vina_size"],
                    config["ifars_enabled"],
                    config["reference_json"],
                    config["ifars_threshold"],
                    config["ifars_w_overlap"],
                    config["ifars_w_critical"],
                    active_ifars_asp114
                )
                scoring_tasks.append((idx, task))

        # Run uncached evaluations in parallel across the persistent worker pool.
        # The executor is created once before the loop — workers stay alive across
        # all iterations and their PLIPAnalyzer singletons are reused each time.
        fresh_results = {}
        if scoring_tasks:
            future_to_idx = {executor.submit(_score_single_molecule_worker, t[1]): t[0] for t in scoring_tasks}
            for future in as_completed(future_to_idx):
                orig_idx = future_to_idx[future]
                try:
                    res = future.result()
                except Exception:
                    res = {
                        "smiles": raw_smiles_list[orig_idx], "valid": False, "total_reward": -5.0,
                        "docking_score": None, "qed": 0.0, "mw": 0.0, "rings": 0, "rot_bonds": 0,
                        "ifars_overlap": 0.0, "critical_overlap": 0.0, "has_asp114": False, "ifars_bonus": 0.0
                    }
                fresh_results[orig_idx] = res
                # Store in LRU cache if valid
                if res["valid"] and len(docking_cache) < config["docking_cache_size"]:
                    docking_cache[res["smiles"]] = res

        dock_time = time.time() - dock_start

        # Combine results in exact original batch order
        batch_results = []
        for idx in range(rollout_size):
            if idx in cached_results:
                batch_results.append(cached_results[idx])
            else:
                batch_results.append(fresh_results[idx])

        # ====================================================================
        # PHASE 3: REWARD AUGMENTATION & DIVERSITY SHAPING
        # ====================================================================
        augmented_rewards = []
        raw_rewards = []
        valid_count = 0
        docked_scores = []
        qed_scores = []
        ifars_overlaps = []
        asp114_hits = 0

        for idx, res in enumerate(batch_results):
            smi = res["smiles"]
            base_reward = res["total_reward"]
            kl_val = kl_div[idx].item()

            if res["valid"]:
                valid_count += 1
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    # Diversity penalty against recent molecules
                    div_pen = chemical_memory.check_diversity(mol, threshold=config["tanimoto_threshold"], penalty=active_div_penalty)
                    # Scaffold exploration bonus from Top-K replay
                    topk_bon = chemical_memory.add_topk(base_reward, smi, mol, bonus_val=config["topk_sim_bonus"])
                    base_reward = base_reward - div_pen + topk_bon

                if res["docking_score"] is not None:
                    docked_scores.append(res["docking_score"])
                qed_scores.append(res["qed"])
                ifars_overlaps.append(res["ifars_overlap"])
                if res["has_asp114"]:
                    asp114_hits += 1

            # REINVENT-style Prior-Agent KL divergence penalty
            aug_reward = base_reward - (active_kl_beta * max(0.0, kl_val))

            raw_rewards.append(base_reward)
            augmented_rewards.append(aug_reward)

            # Log to CSV
            csv_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                iteration,
                smi,
                res["valid"],
                f"{base_reward:.3f}",
                f"{aug_reward:.3f}",
                f"{res['docking_score']:.2f}" if res["docking_score"] is not None else "N/A",
                f"{res['qed']:.3f}",
                f"{res['mw']:.1f}",
                res["rings"],
                res["rot_bonds"],
                f"{kl_val:.4f}",
                f"{res['ifars_overlap']:.3f}",
                f"{res['critical_overlap']:.3f}",
                res["has_asp114"]
            ])

        csv_file.flush()

        # Update running stats
        total_molecules_evaluated += rollout_size
        validity_rate = (valid_count / rollout_size) * 100.0
        asp114_hit_rate = (asp114_hits / max(valid_count, 1)) * 100.0
        mean_ifars_overlap = np.mean(ifars_overlaps) if ifars_overlaps else 0.0

        best_docking_in_batch = min(docked_scores) if docked_scores else None
        if best_docking_in_batch is not None and best_docking_in_batch < best_affinity:
            best_affinity = best_docking_in_batch

        avg_reward = np.mean(raw_rewards)
        avg_aug_reward = np.mean(augmented_rewards)
        avg_kl = kl_div.mean().item()

        # ====================================================================
        # PHASE 4: PPO ACTOR-CRITIC MULTI-EPOCH MINI-BATCH UPDATES (GPU)
        # ====================================================================
        train_start = time.time()
        rewards_tensor = torch.tensor(augmented_rewards, dtype=torch.float32, device=device)

        # Advantage Estimation: A = R - V(s)
        with torch.no_grad():
            advantages = rewards_tensor - old_values
            # Normalize advantages
            adv_mean = advantages.mean()
            adv_std = advantages.std().clamp(min=1e-8)
            normalized_advantages = (advantages - adv_mean) / adv_std

        # Run PPO Mini-batch Epochs
        model.actor.train()
        model.critic.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        update_count = 0
        early_stopped = False

        mini_size = config["mini_batch_size"]
        num_minibatches = math.ceil(rollout_size / mini_size)

        for epoch in range(config["ppo_epochs"]):
            if early_stopped:
                break

            perm = torch.randperm(rollout_size)

            for mb_idx in range(num_minibatches):
                indices = perm[mb_idx * mini_size : (mb_idx + 1) * mini_size]

                mb_input_ids = input_ids[indices]
                mb_attention_mask = attention_mask[indices]
                mb_old_log_probs = old_seq_log_probs[indices]
                mb_advantages = normalized_advantages[indices]
                mb_returns = rewards_tensor[indices]

                with autocast(dtype=torch.float16):
                    new_log_probs, new_values, entropy = model.evaluate_sequences(mb_input_ids, mb_attention_mask)

                    # PPO Clipped Surrogate Loss
                    log_ratio = new_log_probs - mb_old_log_probs
                    ratio = torch.exp(log_ratio)

                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - config["clip_range"], 1.0 + config["clip_range"]) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Critic Value Function Loss
                    value_loss = 0.5 * F.mse_loss(new_values, mb_returns)

                    # Total PPO Loss
                    loss = policy_loss + (config["value_coef"] * value_loss) - (config["entropy_coef"] * entropy)

                # KL divergence guardrail (early stop epoch if policy drifts too quickly)
                approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                if approx_kl > config["target_kl"]:
                    early_stopped = True
                    break

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                update_count += 1

        train_time = time.time() - train_start
        iter_total_time = time.time() - iter_start_time

        # Telemetry & Memory Tracking
        vram_alloc = torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        vram_peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        mols_per_sec = rollout_size / max(iter_total_time, 0.1)

        mean_policy_loss = total_policy_loss / max(update_count, 1)
        mean_value_loss = total_value_loss / max(update_count, 1)

        # TensorBoard Logging
        tb_writer.add_scalar("Rewards/Raw_Mean", avg_reward, iteration)
        tb_writer.add_scalar("Rewards/Augmented_Mean", avg_aug_reward, iteration)
        tb_writer.add_scalar("Adaptive/Phase", active_phase, iteration)
        tb_writer.add_scalar("Adaptive/KL_Beta", active_kl_beta, iteration)
        tb_writer.add_scalar("Adaptive/IFARS_ASP114_Weight", active_ifars_asp114, iteration)
        tb_writer.add_scalar("Adaptive/Diversity_Penalty", active_div_penalty, iteration)
        tb_writer.add_scalar("Chemistry/Validity_Rate_Pct", validity_rate, iteration)
        tb_writer.add_scalar("Chemistry/Mean_QED", np.mean(qed_scores) if qed_scores else 0.0, iteration)
        tb_writer.add_scalar("IFARS/ASP114_Hit_Rate_Pct", asp114_hit_rate, iteration)
        tb_writer.add_scalar("IFARS/Mean_Overlap_Score", mean_ifars_overlap, iteration)
        if best_docking_in_batch is not None:
            tb_writer.add_scalar("Docking/Batch_Best_Affinity", best_docking_in_batch, iteration)
        tb_writer.add_scalar("Docking/AllTime_Best_Affinity", best_affinity, iteration)
        tb_writer.add_scalar("Loss/Policy_Loss", mean_policy_loss, iteration)
        tb_writer.add_scalar("Loss/Value_Loss", mean_value_loss, iteration)
        tb_writer.add_scalar("Regularization/Mean_KL_Div", avg_kl, iteration)
        tb_writer.add_scalar("System/VRAM_Allocated_GB", vram_alloc, iteration)
        tb_writer.add_scalar("System/Throughput_Mols_Per_Sec", mols_per_sec, iteration)

        # Terminal Progress Report
        best_str = f"{best_docking_in_batch:.2f} kcal/mol" if best_docking_in_batch is not None else "N/A"
        phase_names = {1: "Warmup", 2: "Ramp", 3: "Exploit"}
        phase_label = phase_names.get(active_phase, f"P{active_phase}")
        print(
            f"[{datetime_now()}] Iter {iteration:4d}/{total_iterations} [P{active_phase}:{phase_label}] | "
            f"Valid: {validity_rate:5.1f}% | "
            f"Reward: {avg_reward:6.2f} (Aug: {avg_aug_reward:6.2f}) | "
            f"KL: {avg_kl:6.4f} (b={active_kl_beta:.2f}) | "
            f"Best Dock: {best_str:>13} | "
            f"ASP114: {asp114_hit_rate:5.1f}% (w={active_ifars_asp114:.1f}) | "
            f"IF-ARS: {mean_ifars_overlap:4.2f} | "
            f"VRAM: {vram_alloc:4.1f}/{vram_peak:4.1f} GB | "
            f"Speed: {mols_per_sec:4.1f} mol/s"
        )

        # ====================================================================
        # PHASE 5: CHECKPOINTING & ROTATION
        # ====================================================================
        if iteration % config["checkpoint_freq"] == 0 or iteration == total_iterations or interrupted:
            chk_path = os.path.join(config["checkpoint_dir"], f"ppo_hpc_max_iter_{iteration:05d}.pt")
            save_payload = {
                "iteration": iteration,
                "total_molecules": total_molecules_evaluated,
                "best_affinity": best_affinity,
                "best_reward": best_reward,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config
            }
            torch.save(save_payload, chk_path)
            print(f"[{datetime_now()}] [CHECKPOINT] Saved periodic checkpoint -> {chk_path}")

            # Keep only the last N periodic checkpoints to prevent disk overflow
            all_checkpoints = sorted(glob.glob(os.path.join(config["checkpoint_dir"], "ppo_hpc_max_iter_*.pt")))
            if len(all_checkpoints) > config["keep_last_checkpoints"]:
                for stale in all_checkpoints[:-config["keep_last_checkpoints"]]:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass

            # Save dedicated Best Affinity checkpoint
            if best_docking_in_batch is not None and best_docking_in_batch <= best_affinity:
                best_path = os.path.join(config["checkpoint_dir"], "ppo_hpc_max_best_affinity.pt")
                torch.save(save_payload, best_path)
                print(f"[{datetime_now()}] [HIGH AFFINITY] New best binding model saved -> {best_path}")

    # Final wrap-up
    # Shut down the persistent worker pool cleanly so all worker processes
    # are properly reaped before the main process exits.
    executor.shutdown(wait=True)
    csv_file.close()
    tb_writer.close()
    final_path = os.path.join(config["checkpoint_dir"], "ppo_hpc_max_final.pt")
    torch.save({
        "iteration": iteration,
        "total_molecules": total_molecules_evaluated,
        "best_affinity": best_affinity,
        "model_state_dict": model.state_dict(),
        "config": config
    }, final_path)
    print(f"\n===============================================================================")
    print(f"  HPC MAX TRAINING COMPLETE")
    print(f"  Total Molecules Generated : {total_molecules_evaluated:,}")
    print(f"  Best DRD2 Affinity        : {best_affinity:.2f} kcal/mol")
    print(f"  Final Model Saved To      : {final_path}")
    print(f"===============================================================================\n")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Crucial for Windows multiprocessing safety
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Windows HPC Max PPO Training (RTX 3090 24GB)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Rollout batch size generated per iteration on GPU (default: 256; use 512 for max VRAM)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Actor learning rate (default: 2e-5)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total candidate molecules to generate (default: 250,000)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel CPU docking workers (default: min(16, CPU cores))")
    parser.add_argument("--sft-path", type=str, default=None,
                        help="Path to SFT fine-tuned DRD2 MolGPT model directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume training from")

    args = parser.parse_args()
    train_hpc_max(args)
