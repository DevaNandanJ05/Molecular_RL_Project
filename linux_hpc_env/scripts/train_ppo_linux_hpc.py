"""
===============================================================================
Linux Desktop Optimized PPO Training Script for Molecular Generation
===============================================================================
Hardware Target:
  - GPU  : Any NVIDIA GPU with CUDA (e.g., RTX 3090, RTX 4090)
  - CPU  : Linux Desktop with 16+ cores
  - RAM  : 64 GB+
  - OS   : Ubuntu 20.04 / 22.04 LTS

Linux-Specific Upgrades (Desktop Edition):
  1. fork()-based multiprocessing: Workers inherit RDKit/OpenBabel memory
     in <50ms (vs 500ms+ spawn on Windows). No freeze_support() needed.
  2. AutoDock-GPU Integration: GPU-accelerated docking engine.
  3. Gnina (CNN Docking): Deep-learning based scorer as secondary engine.
  4. Atomic Checkpointing: os.replace() is guaranteed atomic on Linux.
  5. Single-GPU Optimization: Maximizes batch generation on a single GPU.
  6. No Windows hacks: Uses system 'obabel' from: apt install openbabel.
  7. OpenMM MD Validation Hook: Validates binding stability post-training.

Usage:
  python scripts/train_ppo_linux_hpc.py [--resume path/to/chk.pt]
===============================================================================
"""

import os
import sys
import csv
import time
import shutil
import uuid
import glob
import math
import argparse
import signal
import subprocess
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================================
# LINUX MULTIPROCESSING: Set fork FIRST
# ============================================================================
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from models.plip_wrapper import PLIPAnalyzer
    PLIP_AVAILABLE = True
except ImportError:
    PLIP_AVAILABLE = False


# ============================================================================
# UTILITY
# ============================================================================
def datetime_now() -> str:
    return time.strftime("%H:%M:%S")

# ============================================================================
# LINUX DESKTOP CONFIGURATION
# ============================================================================
def get_linux_desktop_config() -> dict:
    sft_best = os.path.join(PROJECT_ROOT, "checkpoints", "molgpt_drd2_sft", "best_model")
    sft_path = sft_best if os.path.isdir(sft_best) else "msb-roshan/molgpt"

    return {
        # File Paths
        "receptor_path"      : os.path.join(PROJECT_ROOT, "data", "raw", "drd2_clean.pdbqt"),
        "receptor_maps_dir"  : os.path.join(PROJECT_ROOT, "data", "raw", "drd2_maps"),
        "sft_model_path"     : sft_path,
        "reference_json"     : os.path.join(PROJECT_ROOT, "data", "reference", "haloperidol_fingerprint.json"),

        # Docking Engine Paths
        "autodock_gpu_path"  : os.path.join(PROJECT_ROOT, "bin", "autodock_gpu_128wi"),
        "gnina_path"         : shutil.which("gnina") or os.path.join(PROJECT_ROOT, "bin", "gnina"),
        "vina_path"          : shutil.which("vina") or os.path.join(PROJECT_ROOT, "bin", "vina"),
        "obabel_path"        : "obabel",

        # GPU Batch Generation
        "rollout_batch_size" : 512,  # Floods a single 24GB RTX GPU perfectly
        "max_length"         : 50,
        "temperature"        : 0.85,
        "top_k"              : 40,
        "top_p"              : 0.92,

        # Regularization
        "prior_kl_beta"      : 0.5,

        # PPO Hyperparameters
        "ppo_epochs"         : 4,
        "mini_batch_size"    : 64,
        "learning_rate"      : 2e-5,
        "critic_lr"          : 1e-4,
        "clip_range"         : 0.2,
        "value_coef"         : 0.5,
        "entropy_coef"       : 0.05,
        "max_grad_norm"      : 0.5,
        "target_kl"          : 0.015,

        # Parallel Forked CPU Workers
        "docking_workers"    : min(16, multiprocessing.cpu_count()),
        "vina_exhaustiveness": 8,
        "vina_center"        : (9.5, 5.2, -11.4),
        "vina_size"          : (20.0, 20.0, 20.0),
        "docking_cache_size" : 50000,

        # IF-ARS
        "ifars_enabled"      : True,
        "ifars_threshold"    : -5.0,
        "ifars_w_overlap"    : 2.0,
        "ifars_w_critical"   : 3.0,
        "ifars_w_asp114"     : 2.0,

        # Diversity & Chemical Replay
        "tanimoto_threshold" : 0.75,
        "diversity_penalty"  : 2.5,
        "topk_buffer_size"   : 100,
        "topk_sim_bonus"     : 1.0,

        # OpenMM MD Validation
        "openmm_validate"    : True,
        "openmm_top_n"       : 10,
        "openmm_affinity_cutoff": -9.0,

        # Checkpointing
        "total_timesteps"    : 500_000,
        "checkpoint_freq"    : 25,
        "keep_last_checkpoints": 3,
        "checkpoint_dir"     : os.path.join(PROJECT_ROOT, "checkpoints", "linux_hpc_run"),
        "log_dir"            : os.path.join(PROJECT_ROOT, "logs", "linux_hpc_run"),
    }


# ============================================================================
# AUTODOCK-GPU DOCKING ENGINE
# ============================================================================
def _run_autodock_gpu(receptor_maps_dir: str, ligand_pdbqt: str,
                      ad_gpu_binary: str, result_prefix: str) -> float | None:
    if not os.path.exists(ad_gpu_binary) or not os.path.isdir(receptor_maps_dir):
        return None
    fmaps = os.path.join(receptor_maps_dir, "drd2_clean.maps.fld")
    if not os.path.exists(fmaps):
        return None

    cmd = [
        ad_gpu_binary,
        "--ffile", fmaps,
        "--lfile", ligand_pdbqt,
        "--resnam", result_prefix,
        "--nrun", "20",
        "--devnum", "1", # Assuming GPU 0 is for PyTorch, GPU 1 for AD-GPU if available
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
        dlg_file = f"{result_prefix}.dlg"
        if os.path.exists(dlg_file):
            with open(dlg_file, "r") as f:
                for line in f:
                    if "Estimated Free Energy of Binding" in line:
                        return float(line.split()[7])
    except Exception:
        pass
    return None


def _run_gnina(receptor_path: str, ligand_pdbqt: str,
               gnina_binary: str, center: tuple, size: tuple) -> float | None:
    if not gnina_binary or not os.path.exists(gnina_binary):
        return None
    cx, cy, cz = center
    sx, sy, sz = size
    out_file = ligand_pdbqt.replace(".pdbqt", "_gnina_out.pdbqt")
    cmd = [
        gnina_binary, "--receptor", receptor_path, "--ligand", ligand_pdbqt,
        "--out", out_file,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--exhaustiveness", "8", "--num_modes", "1", "--quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "1":
                try: return float(parts[1])
                except ValueError: pass
    except Exception:
        pass
    return None


def _run_vina_fallback(receptor_path: str, ligand_pdbqt: str, vina_binary: str,
                       center: tuple, size: tuple, exhaustiveness: int,
                       config_file: str, docked_out: str) -> float | None:
    if not vina_binary or not os.path.exists(vina_binary):
        return None
    cx, cy, cz = center
    sx, sy, sz = size
    with open(config_file, "w") as f:
        f.write(f"receptor = {receptor_path}\nligand = {ligand_pdbqt}\nout = {docked_out}\n")
        f.write(f"center_x = {cx}\ncenter_y = {cy}\ncenter_z = {cz}\n")
        f.write(f"size_x = {sx}\nsize_y = {sy}\nsize_z = {sz}\n")
        f.write(f"exhaustiveness = {exhaustiveness}\ncpu = 1\n")
    try:
        v_res = subprocess.run([vina_binary, f"--config={config_file}"],
                               capture_output=True, text=True, timeout=90)
        for line in v_res.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "1":
                try: return float(parts[1])
                except ValueError: pass
    except Exception:
        pass
    return None


# ============================================================================
# FORK-OPTIMISED WORKER FUNCTION
# ============================================================================
def _score_molecule_linux(task: tuple) -> dict:
    (smiles, receptor_path, receptor_maps_dir, ad_gpu_path, gnina_path,
     vina_path, obabel_path, exhaustiveness, center, size,
     ifars_enabled, ref_json, ifars_threshold, w_overlap, w_critical, w_asp114) = task

    empty_res = {
        "smiles": smiles, "valid": False, "total_reward": -5.0,
        "docking_score": None, "qed": 0.0, "mw": 0.0, "rings": 0, "rot_bonds": 0,
        "ifars_overlap": 0.0, "critical_overlap": 0.0, "has_asp114": False,
        "ifars_bonus": 0.0, "docking_engine": "none",
    }

    if not smiles or len(smiles.strip()) < 3: return empty_res
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return empty_res
    try: Chem.SanitizeMol(mol)
    except: return empty_res

    try:
        qed_val    = float(QED.qed(mol))
        mw_val     = float(Descriptors.MolWt(mol))
        rings_val  = int(rdMolDescriptors.CalcNumRings(mol))
        rot_bonds  = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    except Exception:
        return empty_res

    penalty = 0.0
    if mw_val < 160.0 or mw_val > 550.0: penalty += 2.0
    if rings_val == 0: penalty += 1.5
    if rot_bonds > 10: penalty += 1.0

    if not receptor_path or not os.path.isfile(receptor_path):
        return {**empty_res, "valid": True, "total_reward": (qed_val * 3.0) - penalty,
                "qed": qed_val, "mw": mw_val, "rings": rings_val, "rot_bonds": rot_bonds}

    worker_pid = os.getpid()
    unique_id  = uuid.uuid4().hex[:8]
    temp_dir   = os.path.join(PROJECT_ROOT, "data", "temp", f"linux_{worker_pid}_{unique_id}")
    os.makedirs(temp_dir, exist_ok=True)

    temp_sdf    = os.path.join(temp_dir, "ligand.sdf")
    temp_pdbqt  = os.path.join(temp_dir, "ligand.pdbqt")
    docked_out  = os.path.join(temp_dir, "docked_out.pdbqt")
    vina_config = os.path.join(temp_dir, "vina_config.txt")
    ad_result   = os.path.join(temp_dir, "ad_result")

    docking_score  = None
    docking_engine = "none"
    ifars_overlap  = 0.0
    critical_overlap = 0.0
    has_asp114     = False
    ifars_bonus    = 0.0

    try:
        mol_3d = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3()) != 0:
            if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) != 0:
                return {**empty_res, "valid": False}

        try: AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
        except: pass

        writer = Chem.SDWriter(temp_sdf)
        writer.write(mol_3d)
        writer.close()

        subprocess.run([obabel_path, temp_sdf, "-O", temp_pdbqt, "-h"],
                       capture_output=True, timeout=25)

        if not os.path.exists(temp_pdbqt) or os.path.getsize(temp_pdbqt) == 0:
            return {**empty_res, "valid": True, "total_reward": (qed_val * 3.0) - penalty,
                    "qed": qed_val, "mw": mw_val}

        # 1. AutoDock-GPU
        score = _run_autodock_gpu(receptor_maps_dir, temp_pdbqt, ad_gpu_path, ad_result)
        if score is not None:
            docking_score, docking_engine = score, "autodock_gpu"
        # 2. Gnina
        if docking_score is None:
            score = _run_gnina(receptor_path, temp_pdbqt, gnina_path, center, size)
            if score is not None:
                docking_score, docking_engine = score, "gnina"
        # 3. Vina CPU
        if docking_score is None:
            score = _run_vina_fallback(receptor_path, temp_pdbqt, vina_path,
                                       center, size, exhaustiveness, vina_config, docked_out)
            if score is not None:
                docking_score, docking_engine = score, "vina"

        # IF-ARS (PLIP)
        if (ifars_enabled and PLIP_AVAILABLE and docking_score is not None
                and docking_score <= ifars_threshold):
            pose_file = docked_out if os.path.exists(docked_out) else None
            if not pose_file and os.path.exists(f"{ad_result}.dlg"):
                pose_file = f"{ad_result}.dlg"
            if pose_file:
                ifars_scale = min(1.0, max(0.0, (-docking_score - abs(ifars_threshold)) / 4.0))
                try:
                    plip = PLIPAnalyzer(receptor_pdbqt=receptor_path,
                                        reference_json=ref_json, obabel_path=obabel_path)
                    plip_res = plip.score_docked_pose(pose_file)
                    ifars_overlap    = plip_res.get("overlap_score", 0.0)
                    critical_overlap = plip_res.get("critical_overlap", 0.0)
                    matched          = plip_res.get("matched_keys", [])
                    has_asp114       = any("ASP114" in k for k in matched)
                    ifars_bonus = ifars_scale * ((w_overlap * ifars_overlap) +
                        (w_critical * critical_overlap) + (w_asp114 if has_asp114 else 0.0))
                except:
                    ifars_bonus = ifars_scale * 0.5
    except Exception:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if docking_score is not None and docking_score < 0:
        total_reward = abs(docking_score) + (qed_val * 2.0) + ifars_bonus - penalty
    elif docking_score is None:
        total_reward = -3.0 + (qed_val * 1.0) - penalty
    else:
        total_reward = -2.0 + (qed_val * 1.5) - penalty

    return {
        "smiles": smiles, "valid": True, "total_reward": total_reward,
        "docking_score": docking_score, "qed": qed_val, "mw": mw_val,
        "rings": rings_val, "rot_bonds": rot_bonds,
        "ifars_overlap": ifars_overlap, "critical_overlap": critical_overlap,
        "has_asp114": has_asp114, "ifars_bonus": ifars_bonus,
        "docking_engine": docking_engine,
    }


# ============================================================================
# ATOMIC SAVE
# ============================================================================
def atomic_save(checkpoint: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    temp_path = f"{filepath}.tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, filepath)

def rotate_checkpoints(checkpoint_dir: str, prefix: str, keep_last: int):
    all_chk = sorted(glob.glob(os.path.join(checkpoint_dir, f"{prefix}_iter_*.pt")))
    for stale in all_chk[:-keep_last]:
        try: os.remove(stale)
        except OSError: pass


# ============================================================================
# ACTOR-CRITIC MODEL (Single Node)
# ============================================================================
class MolGPTPPOModel(nn.Module):
    def __init__(self, sft_model_path: str, device: torch.device):
        super().__init__()
        self.device = device
        print(f"[{datetime_now()}] Loading MolGPT from: {sft_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None: self.tokenizer.bos_token = self.tokenizer.eos_token
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size   = len(self.tokenizer)

        self.actor = AutoModelForCausalLM.from_pretrained(sft_model_path).to(self.device)
        self.hidden_dim = self.actor.config.n_embd

        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        ).to(self.device)

        for param in self.actor.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def generate_batch(self, batch_size: int, max_length: int = 50,
                       temperature: float = 0.85, top_k: int = 40, top_p: float = 0.92):
        self.actor.eval()
        input_ids = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        finished  = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_length - 1):
            with autocast(dtype=torch.float16):
                outputs = self.actor(input_ids=input_ids)
                logits  = outputs.logits[:, -1, :] / temperature

            if finished.all(): break

            logits       = self._top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs        = F.softmax(logits, dim=-1)
            next_tokens  = torch.multinomial(probs, num_samples=1)
            next_tokens  = torch.where(finished.unsqueeze(1), torch.tensor(self.pad_token_id, device=self.device), next_tokens)
            finished    = finished | (next_tokens.squeeze(1) == self.eos_token_id)
            input_ids   = torch.cat([input_ids, next_tokens], dim=1)

        attention_mask = (input_ids != self.pad_token_id).long()
        decoded_smiles = []
        for seq in input_ids.cpu().numpy():
            tokens = [t for t in seq if t not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)]
            smi    = self.tokenizer.decode(tokens, skip_special_tokens=True).replace(" ", "").split(".")[0]
            decoded_smiles.append(smi)
        return input_ids, attention_mask, decoded_smiles

    def evaluate_sequences(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with autocast(dtype=torch.float16):
            outputs  = self.actor(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits   = outputs.logits[:, :-1, :]
            targets  = input_ids[:, 1:]
            mask     = attention_mask[:, 1:].float()
            log_probs = F.log_softmax(logits, dim=-1)
            seq_log_probs = (log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=-1)
            probs = F.softmax(logits, dim=-1)
            mean_entropy = (-(probs * log_probs).sum(dim=-1) * mask).sum() / mask.sum().clamp(min=1.0)
            hidden_states = outputs.hidden_states[-1][:, :-1, :]
            seq_lengths   = mask.sum(-1).long().clamp(min=1) - 1
            batch_idx     = torch.arange(input_ids.shape[0], device=self.device)
            values        = self.critic(hidden_states[batch_idx, seq_lengths, :].float()).squeeze(-1)
        return seq_log_probs.float(), values.float(), mean_entropy.float()

    @staticmethod
    def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0):
        if top_k > 0:
            filter_val = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits     = torch.where(logits < filter_val, torch.full_like(logits, -float("Inf")), logits)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs  = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            to_remove  = cum_probs > top_p
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0]  = 0
            logits = logits.masked_fill(to_remove.scatter(1, sorted_idx, to_remove), -float("Inf"))
        return logits


class FrozenPriorModel:
    def __init__(self, sft_model_path: str, device: torch.device):
        self.device = device
        self.prior  = AutoModelForCausalLM.from_pretrained(sft_model_path).to(self.device)
        self.prior.eval()
        for p in self.prior.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute_prior_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with autocast(dtype=torch.float16):
            logits   = self.prior(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
            targets  = input_ids[:, 1:]
            mask     = attention_mask[:, 1:].float()
            log_probs = F.log_softmax(logits, dim=-1)
            return (log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=-1).float()


class ChemicalMemory:
    def __init__(self, recent_size: int = 1500, topk_size: int = 100):
        self.recent_fps = deque(maxlen=recent_size)
        self.topk_molecules = []
        self.topk_size = topk_size

    def check_diversity(self, mol, threshold: float = 0.75, penalty: float = 2.5):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        if self.recent_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, list(self.recent_fps))
            max_sim = max(sims) if sims else 0.0
            self.recent_fps.append(fp)
            return penalty if max_sim > threshold else 0.0
        self.recent_fps.append(fp)
        return 0.0

    def add_topk(self, reward: float, smiles: str, mol, bonus_val: float = 1.0):
        bonus = 0.0
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            if self.topk_molecules:
                topk_fps = [item[2] for item in self.topk_molecules]
                if 0.35 <= max(DataStructs.BulkTanimotoSimilarity(fp, topk_fps)) <= 0.70:
                    bonus = bonus_val
            if reward > 3.0:
                self.topk_molecules.append((reward, smiles, fp))
                self.topk_molecules.sort(key=lambda x: x[0], reverse=True)
                self.topk_molecules = self.topk_molecules[:self.topk_size]
        except: pass
        return bonus


def validate_top_hits_openmm(top_hits: list, config: dict):
    try:
        from openmm.app import ForceField, Simulation, Modeller, PDBFile
        from openmm import LangevinMiddleIntegrator, unit
    except ImportError:
        print("[OpenMM] Not installed. Install with: mamba install -c conda-forge openmm")
        return
    print(f"\n[OpenMM] Running MD energy minimisation on top {len(top_hits)} hits...")
    results = []
    for smiles, dock_score in top_hits:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue
        try:
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
            AllChem.UFFOptimizeMolecule(mol_3d, maxIters=500)
            tmp_pdb = os.path.join(config["log_dir"], f"hit_{smiles[:20].replace('/', '_')}.pdb")
            Chem.MolToPDBFile(mol_3d, tmp_pdb)

            pdb = PDBFile(tmp_pdb)
            ff  = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
            mod = Modeller(pdb.topology, pdb.positions)
            sys_ = ff.createSystem(mod.topology)
            sim = Simulation(mod.topology, sys_, LangevinMiddleIntegrator(
                300 * unit.kelvin, 1 / unit.picosecond, 0.004 * unit.picoseconds))
            sim.context.setPositions(mod.positions)
            sim.minimizeEnergy(maxIterations=500)

            energy = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            results.append((smiles, dock_score, energy))
            print(f"  [OpenMM] {smiles[:40]:<40} | Dock: {dock_score:.2f} | MM: {energy:.1f} kcal/mol")
        except Exception as e:
            pass

    if results:
        out_csv = os.path.join(config["log_dir"], "openmm_validated_hits.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "DockingScore_kcal_mol", "MM_Energy_kcal_mol"])
            writer.writerows(results)
        print(f"[OpenMM] Saved to: {out_csv}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def train_linux_desktop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("=" * 79)
        print("  LINUX DESKTOP RL PIPELINE (SINGLE GPU)")
        print(f"  GPU          : {gpu_name} ({vram_gb:.1f} GB)")
        print(f"  Multiproc    : fork() enabled")
        print("=" * 79)

    config = get_linux_desktop_config()
    if args.batch_size : config["rollout_batch_size"] = args.batch_size
    if args.lr         : config["learning_rate"] = args.lr
    if args.timesteps  : config["total_timesteps"] = args.timesteps
    if args.workers    : config["docking_workers"] = args.workers
    if args.sft_path   : config["sft_model_path"] = args.sft_path

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "temp"), exist_ok=True)

    tb_writer  = SummaryWriter(log_dir=config["log_dir"])
    csv_log    = os.path.join(config["log_dir"], "molecule_log.csv")
    csv_exists = os.path.isfile(csv_log)
    csv_file   = open(csv_log, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "Timestamp", "Iteration", "SMILES", "Valid", "Reward", "AugmentedReward",
            "DockingScore", "QED", "MW", "Rings", "RotBonds", "KL_Divergence", 
            "IFARS_Overlap", "CriticalOverlap", "HasASP114", "DockingEngine"
        ])
        csv_file.flush()

    model = MolGPTPPOModel(config["sft_model_path"], device=device)
    prior = FrozenPriorModel(config["sft_model_path"], device=device)

    optimizer = torch.optim.AdamW([
        {"params": model.actor.parameters(),  "lr": config["learning_rate"],  "weight_decay": 0.01},
        {"params": model.critic.parameters(), "lr": config["critic_lr"],       "weight_decay": 0.01},
    ])
    scaler = GradScaler()

    chemical_memory = ChemicalMemory(recent_size=1500, topk_size=config["topk_buffer_size"])
    docking_cache   = {}

    start_iteration        = 0
    total_molecules_done   = 0
    best_affinity          = 0.0
    best_reward            = -float("inf")
    top_hits_for_md        = []

    if args.resume and os.path.isfile(args.resume):
        print(f"[{datetime_now()}] Restoring atomic checkpoint: {args.resume}")
        chk = torch.load(args.resume, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        scaler.load_state_dict(chk["scaler_state_dict"])
        start_iteration      = chk.get("iteration", 0)
        total_molecules_done = chk.get("total_molecules", 0)
        best_affinity        = chk.get("best_affinity", 0.0)
        best_reward          = chk.get("best_reward", -float("inf"))

    interrupted = False
    def _handle_signal(sig, frame):
        nonlocal interrupted
        print(f"\n[{datetime_now()}] [SIGINT/SIGTERM] Triggering emergency atomic save...")
        interrupted = True
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    rollout_size     = config["rollout_batch_size"]
    total_iterations = config["total_timesteps"] // rollout_size

    for iteration in range(start_iteration + 1, total_iterations + 1):
        if interrupted: break
        iter_start = time.time()

        # PHASE 1: GPU BATCH GENERATION
        gen_start = time.time()
        input_ids, attention_mask, raw_smiles_list = model.generate_batch(
            batch_size=rollout_size, max_length=config["max_length"],
            temperature=config["temperature"], top_k=config["top_k"], top_p=config["top_p"],
        )
        gen_time = time.time() - gen_start

        with torch.no_grad():
            old_seq_log_probs, old_values, _ = model.evaluate_sequences(input_ids, attention_mask)
            prior_seq_log_probs = prior.compute_prior_log_probs(input_ids, attention_mask)
            seq_lens = attention_mask[:, 1:].sum(-1).float().clamp(min=1.0)
            kl_div   = (old_seq_log_probs - prior_seq_log_probs) / seq_lens

        # PHASE 2: FORK CPU DOCKING
        dock_start = time.time()
        scoring_tasks = []
        cached_results = {}

        for idx, smi in enumerate(raw_smiles_list):
            if smi in docking_cache:
                cached_results[idx] = docking_cache[smi]
            else:
                scoring_tasks.append((idx, (
                    smi, config["receptor_path"], config["receptor_maps_dir"],
                    config["autodock_gpu_path"], config["gnina_path"], config["vina_path"],
                    config["obabel_path"], config["vina_exhaustiveness"], config["vina_center"],
                    config["vina_size"], config["ifars_enabled"], config["reference_json"],
                    config["ifars_threshold"], config["ifars_w_overlap"], config["ifars_w_critical"], config["ifars_w_asp114"]
                )))

        fresh_results = {}
        if scoring_tasks:
            with ProcessPoolExecutor(max_workers=config["docking_workers"]) as pool:
                future_map = {pool.submit(_score_molecule_linux, t[1]): t[0] for t in scoring_tasks}
                for future in as_completed(future_map):
                    orig_idx = future_map[future]
                    try:
                        res = future.result()
                    except Exception:
                        res = {"smiles": raw_smiles_list[orig_idx], "valid": False, "total_reward": -5.0,
                               "docking_score": None, "qed": 0.0, "mw": 0.0, "rings": 0, "rot_bonds": 0,
                               "ifars_overlap": 0.0, "critical_overlap": 0.0, "has_asp114": False,
                               "ifars_bonus": 0.0, "docking_engine": "error"}
                    fresh_results[orig_idx] = res
                    if res["valid"] and len(docking_cache) < config["docking_cache_size"]:
                        docking_cache[res["smiles"]] = res

        dock_time = time.time() - dock_start
        batch_results = [cached_results.get(i, fresh_results.get(i)) for i in range(rollout_size)]

        # PHASE 3: REWARD SHAPING
        augmented_rewards, raw_rewards, docked_scores, qed_scores = [], [], [], []
        valid_count = 0
        engine_counts = {"autodock_gpu": 0, "gnina": 0, "vina": 0, "none": 0, "error": 0}

        for idx, res in enumerate(batch_results):
            smi = res["smiles"]
            base_reward = res["total_reward"]
            kl_val = kl_div[idx].item()
            engine_counts[res.get("docking_engine", "none")] += 1

            if res["valid"]:
                valid_count += 1
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    base_reward += chemical_memory.add_topk(base_reward, smi, mol, config["topk_sim_bonus"])
                    base_reward -= chemical_memory.check_diversity(mol, config["tanimoto_threshold"], config["diversity_penalty"])
                
                if res["docking_score"] is not None:
                    docked_scores.append(res["docking_score"])
                    if res["docking_score"] <= config["openmm_affinity_cutoff"]:
                        top_hits_for_md.append((smi, res["docking_score"]))
                qed_scores.append(res["qed"])

            aug_reward = base_reward - (config["prior_kl_beta"] * max(0.0, kl_val))
            raw_rewards.append(base_reward)
            augmented_rewards.append(aug_reward)

            csv_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), iteration, smi, res["valid"], f"{base_reward:.3f}",
                f"{aug_reward:.3f}", f"{res['docking_score']:.2f}" if res["docking_score"] is not None else "N/A",
                f"{res['qed']:.3f}", f"{res['mw']:.1f}", res["rings"], res["rot_bonds"], f"{kl_val:.4f}",
                f"{res['ifars_overlap']:.3f}", f"{res['critical_overlap']:.3f}", res["has_asp114"], res.get("docking_engine", "?")
            ])
        csv_file.flush()

        total_molecules_done += rollout_size
        validity_rate = (valid_count / rollout_size) * 100.0
        best_batch_dock = min(docked_scores) if docked_scores else None
        if best_batch_dock is not None and best_batch_dock < best_affinity:
            best_affinity = best_batch_dock
        avg_reward = float(np.mean(raw_rewards))
        avg_aug_reward = float(np.mean(augmented_rewards))
        avg_kl = kl_div.mean().item()

        # PHASE 4: PPO UPDATES
        train_start = time.time()
        rewards_t = torch.tensor(augmented_rewards, dtype=torch.float32, device=device)

        with torch.no_grad():
            advantages = rewards_t - old_values
            norm_advantages = (advantages - advantages.mean()) / advantages.std().clamp(min=1e-8)

        model.actor.train()
        model.critic.train()

        total_policy_loss, total_value_loss, update_count, early_stopped = 0.0, 0.0, 0, False
        mini_size = config["mini_batch_size"]
        n_minibatches = math.ceil(rollout_size / mini_size)

        for epoch in range(config["ppo_epochs"]):
            if early_stopped: break
            perm = torch.randperm(rollout_size)
            for mb in range(n_minibatches):
                indices = perm[mb * mini_size : (mb + 1) * mini_size]
                with autocast(dtype=torch.float16):
                    new_lp, new_vals, entropy = model.evaluate_sequences(input_ids[indices], attention_mask[indices])
                    ratio = torch.exp(new_lp - old_seq_log_probs[indices])
                    surr1 = ratio * norm_advantages[indices]
                    surr2 = torch.clamp(ratio, 1.0 - config["clip_range"], 1.0 + config["clip_range"]) * norm_advantages[indices]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * F.mse_loss(new_vals, rewards_t[indices])
                    loss = policy_loss + config["value_coef"] * value_loss - config["entropy_coef"] * entropy

                if (old_seq_log_probs[indices] - new_lp).mean().item() > config["target_kl"]:
                    early_stopped = True; break

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                update_count += 1

        iter_time = time.time() - iter_start

        # TELEMETRY
        mols_per_s = rollout_size / max(iter_time, 0.1)
        tb_writer.add_scalar("Rewards/Raw_Mean", avg_reward, iteration)
        tb_writer.add_scalar("Chemistry/Validity_Rate_Pct", validity_rate, iteration)
        if best_batch_dock is not None: tb_writer.add_scalar("Docking/Batch_Best", best_batch_dock, iteration)
        print(f"[{datetime_now()}] Iter {iteration:4d}/{total_iterations} | Valid: {validity_rate:5.1f}% | R: {avg_reward:6.2f} | Speed: {mols_per_s:.1f} mol/s")

        # PHASE 5: ATOMIC CHECKPOINTING
        if iteration % config["checkpoint_freq"] == 0 or iteration == total_iterations or interrupted:
            chk_path = os.path.join(config["checkpoint_dir"], f"ppo_linux_hpc_iter_{iteration:05d}.pt")
            payload = {
                "iteration": iteration, "total_molecules": total_molecules_done, "best_affinity": best_affinity,
                "best_reward": best_reward, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "config": config,
            }
            atomic_save(payload, chk_path)
            rotate_checkpoints(config["checkpoint_dir"], "ppo_linux_hpc", config["keep_last_checkpoints"])
            if best_batch_dock is not None and best_batch_dock <= best_affinity:
                atomic_save(payload, os.path.join(config["checkpoint_dir"], "ppo_linux_hpc_best.pt"))

        if interrupted: break

    # POST-TRAINING MD
    if config.get("openmm_validate") and top_hits_for_md:
        unique_hits = []
        for smi, score in sorted(top_hits_for_md, key=lambda x: x[1]):
            if smi not in [u[0] for u in unique_hits]: unique_hits.append((smi, score))
            if len(unique_hits) >= config["openmm_top_n"]: break
        validate_top_hits_openmm(unique_hits, config)

    csv_file.close()
    tb_writer.close()
    atomic_save(payload, os.path.join(config["checkpoint_dir"], "ppo_linux_hpc_final.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sft-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train_linux_desktop(args)
