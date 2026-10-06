"""
Extract and rank elite molecules from the HPC Max Molecular RL run.
Elite threshold: docking_score <= -8.5 kcal/mol with Lipinski Ro5 and SAS validation.
Output: elite_molecules.csv sorted by docking score
"""
import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HPC_MAX_LOG = os.path.join(PROJECT_ROOT, "logs", "hpc_max_run", "molecule_log.csv")
FROZEN_LOG = os.path.join(PROJECT_ROOT, "logs", "hpc_frozen_run", "molecule_log_gpu.csv")

CSV_PATH = HPC_MAX_LOG if os.path.isfile(HPC_MAX_LOG) else FROZEN_LOG
OUT_PATH = os.path.join(PROJECT_ROOT, "logs", "elite_molecules.csv")
ELITE_THRESHOLD = -8.5

std_cols = [
    "timestamp", "iteration", "smiles", "valid", "reward", "aug_reward",
    "docking_score", "qed", "mw", "rings", "rot_bonds",
    "kl_div", "ifars_overlap", "critical_overlap", "has_asp114"
]

with open(CSV_PATH, "r", encoding="utf-8") as f:
    first_line = f.readline().lower()

if "smiles" in first_line:
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower().replace("_", "") for c in df.columns]
    col_map = {
        "dockingscore": "docking_score", "augmentedreward": "aug_reward",
        "rotbonds": "rot_bonds", "kldivergence": "kl_div",
        "ifarsoverlap": "ifars_overlap", "criticaloverlap": "critical_overlap",
        "hasasp114": "has_asp114", "asp114hit": "has_asp114", "episode": "iteration"
    }
    df = df.rename(columns=col_map)
else:
    df = pd.read_csv(CSV_PATH, header=None, names=std_cols)

for col in ["docking_score", "qed", "mw", "reward", "ifars_overlap"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "valid" in df.columns:
    df["valid"] = df["valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    valid_df = df[df["valid"] == True].copy()
else:
    valid_df = df.copy()

valid_df = valid_df.dropna(subset=["docking_score"])
valid_df = valid_df.sort_values("docking_score").drop_duplicates(subset=["smiles"], keep="first")

elite = valid_df[valid_df["docking_score"] <= ELITE_THRESHOLD].copy()

# Re-validate with RDKit and compute fresh QED, SAS, and Ro5
rows = []
for _, row in elite.iterrows():
    mol = Chem.MolFromSmiles(str(row["smiles"]))
    if mol is None:
        continue
    canon = Chem.MolToSmiles(mol)
    try:
        qed_val = float(QED.qed(mol))
        mw_val  = float(Descriptors.MolWt(mol))
        logp    = float(Descriptors.MolLogP(mol))
        hbd     = int(rdMolDescriptors.CalcNumHBD(mol))
        hba     = int(rdMolDescriptors.CalcNumHBA(mol))
        tpsa    = float(Descriptors.TPSA(mol))
        rotb    = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        rings   = int(rdMolDescriptors.CalcNumRings(mol))
        sas     = float(sascorer.calculateScore(mol))
    except Exception:
        continue

    ro5_violations = sum([mw_val > 500.0, logp > 5.0, hbd > 5, hba > 10, tpsa > 140.0])

    rows.append({
        "smiles":         canon,
        "docking_score":  round(float(row["docking_score"]), 2),
        "reward":         round(float(row.get("reward", 0.0)), 2),
        "qed":            round(qed_val, 3),
        "mol_weight":     round(mw_val, 1),
        "logp":           round(logp, 2),
        "sas":            round(sas, 2),
        "ro5_violations": ro5_violations,
        "rot_bonds":      rotb,
        "rings":          rings,
        "asp114_hit":     bool(row.get("has_asp114", False)),
        "ifars_overlap":  round(float(row.get("ifars_overlap", 0.0)), 3),
        "iteration":      row.get("iteration", "?"),
    })

if not rows:
    print(f"No elite molecules found at threshold {ELITE_THRESHOLD}. Showing best 15 overall:")
    valid_df = valid_df.sort_values("docking_score").head(15)
    print(valid_df[["smiles", "docking_score", "qed", "mw"]].to_string())
else:
    out = pd.DataFrame(rows).sort_values("docking_score").reset_index(drop=True)
    out.index += 1  # rank from 1

    print("=" * 80)
    print(f"  ELITE MOLECULE EXTRACTION -- HPC Max Molecular RL")
    print("=" * 80)
    print(f"  Source log file        : {CSV_PATH}")
    print(f"  Total molecules in log : {len(df):,}")
    print(f"  Valid docked molecules : {len(valid_df):,}")
    print(f"  Elite (< {ELITE_THRESHOLD} kcal/mol): {len(out)}")
    print()
    print(out[["smiles", "docking_score", "asp114_hit", "sas", "qed", "mol_weight", "ro5_violations"]].head(20).to_string())
    print()
    print(f"  Saved to: {OUT_PATH}")
    out.to_csv(OUT_PATH, index_label="rank")
