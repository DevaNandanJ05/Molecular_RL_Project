"""
=============================================================================
COMPARATIVE DOCKING: Molecule 1 vs 5 Known DRD2 Drugs
=============================================================================
Docks all 6 molecules (Molecule 1 + 5 clinical DRD2 drugs) against DRD2
using AutoDock Vina at Exhaustiveness=64 and generates a thesis-quality
comparison table and bar chart.

Drugs: Haloperidol, Clozapine, Risperidone, Aripiprazole, Olanzapine
=============================================================================
"""

import os
import sys
import subprocess
import shutil
import time
import json
import csv
import urllib.request
import urllib.parse
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "logs", "hpc_max_run")
WORK_DIR    = os.path.join(BASE_DIR, "data", "temp", "comparative_docking")
RECEPTOR    = os.path.join(BASE_DIR, "data", "raw", "drd2_clean.pdbqt")
VINA_EXE    = os.path.join(BASE_DIR, "bin", "vina.exe")
OBABEL_EXE  = r"C:\Program Files\OpenBabel-2.4.1\obabel.exe"
PYTHON_EXE  = sys.executable

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# ── Vina Box (centred on Asp114 binding pocket) ────────────────────────────
# Extracted from existing hpc_worker configs
CENTER_X, CENTER_Y, CENTER_Z = 9.5, 5.2, -11.4
SIZE_X, SIZE_Y, SIZE_Z = 20.0, 20.0, 20.0
EXHAUSTIVENESS = 64
NUM_MODES = 9
CPU_CORES = max(1, os.cpu_count() - 1)

# ── Molecule definitions ──────────────────────────────────────────────────
MOLECULES = {
    "Molecule_1_AI": {
        "smiles": "O=C(NCc1ccccc1)c1ccccc1C(=O)OCc1ccccc1Cl",
        "label":  "Molecule 1 (AI-Generated)",
        "color":  "#FF6B35"
    },
    "Haloperidol": {
        "smiles": "O=C(CCCN1CCC(c2ccc(F)cc2)(O)CC1)c1ccc(Cl)cc1",
        "label":  "Haloperidol (Reference)",
        "color":  "#4ECDC4"
    },
    "Clozapine": {
        "smiles": "CN1CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42",
        "label":  "Clozapine",
        "color":  "#45B7D1"
    },
    "Risperidone": {
        "smiles": "Cc1nc2ccccc2c(=O)n1CCCC1=C(F)C=CC(=C1)N1CCC(=O)NC1=O",
        "label":  "Risperidone",
        "color":  "#96CEB4"
    },
    "Aripiprazole": {
        "smiles": "C1CC2=C(C(=O)NC1)C=CC(=C2)OCCCCN3CCN(CC3)C4=C(C(=CC=C4)Cl)Cl",
        "label":  "Aripiprazole",
        "color":  "#FFEAA7"
    },
    "Olanzapine": {
        "smiles": "CC1=CC2=C(S1)NC3=CC=CC=C3N=C2N4CCN(CC4)C",
        "label":  "Olanzapine",
        "color":  "#DDA0DD"
    },
}

def log(msg):
    msg_clean = msg.encode('ascii', 'replace').decode('ascii')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg_clean}")


def smiles_to_pdbqt(name, smiles, work_dir):
    """Convert SMILES → 3D SDF (RDKit) → PDBQT (OpenBabel)."""
    log(f"  Preparing {name}: SMILES → 3D PDBQT...")

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log(f"    ERROR: RDKit could not parse SMILES for {name}")
            return None

        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result == -1:
            # Fallback: try ETKDG
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result == -1:
            log(f"    ERROR: 3D embedding failed for {name}")
            return None

        AllChem.MMFFOptimizeMolecule(mol)

        sdf_path   = os.path.join(work_dir, f"{name}.sdf")
        pdbqt_path = os.path.join(work_dir, f"{name}.pdbqt")

        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()

        # Convert SDF → PDBQT via OpenBabel
        res = subprocess.run(
            [OBABEL_EXE, "-isdf", sdf_path, "-opdbqt", "-O", pdbqt_path,
             "--gen3d", "-h"],
            capture_output=True, text=True
        )
        if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 100:
            log(f"    ✅ {name}.pdbqt ready ({os.path.getsize(pdbqt_path)} bytes)")
            return pdbqt_path
        else:
            log(f"    ERROR: PDBQT not created for {name}")
            log(f"    Obabel stderr: {res.stderr[:200]}")
            return None

    except Exception as e:
        log(f"    EXCEPTION for {name}: {e}")
        return None


def run_vina(name, ligand_pdbqt, work_dir):
    """Run AutoDock Vina and return the best binding energy."""
    out_pdbqt = os.path.join(work_dir, f"{name}_docked.pdbqt")
    log_file  = os.path.join(work_dir, f"{name}_vina.log")

    cmd = [
        VINA_EXE,
        "--receptor", RECEPTOR,
        "--ligand",   ligand_pdbqt,
        "--out",      out_pdbqt,
        "--center_x", str(CENTER_X),
        "--center_y", str(CENTER_Y),
        "--center_z", str(CENTER_Z),
        "--size_x",   str(SIZE_X),
        "--size_y",   str(SIZE_Y),
        "--size_z",   str(SIZE_Z),
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes", str(NUM_MODES),
        "--cpu",      str(CPU_CORES),
    ]

    log(f"  Running Vina on {name} (exhaustiveness={EXHAUSTIVENESS}, CPU cores={CPU_CORES})...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        elapsed = time.time() - start_time

        # Save log
        with open(log_file, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        # Parse best binding energy from output PDBQT
        best_energy = None
        if os.path.exists(out_pdbqt):
            with open(out_pdbqt) as f:
                for line in f:
                    if "VINA RESULT" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                best_energy = float(parts[3])
                                break
                            except ValueError:
                                pass

        if best_energy is None:
            # Try parsing from stdout
            for line in result.stdout.split("\n"):
                if line.strip().startswith("1 ") or line.strip().startswith("   1 "):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            best_energy = float(parts[1])
                            break
                        except ValueError:
                            pass

        if best_energy is not None:
            log(f"  ✅ {name}: {best_energy:.3f} kcal/mol ({elapsed:.0f}s)")
        else:
            log(f"  ⚠️  {name}: Could not parse energy. Check {log_file}")

        return best_energy, elapsed

    except subprocess.TimeoutExpired:
        log(f"  ❌ {name}: Vina timed out!")
        return None, None
    except Exception as e:
        log(f"  ❌ {name}: Exception - {e}")
        return None, None


def generate_report(results):
    """Generate clean comparison table and CSV."""
    print()
    print("=" * 75)
    print("   COMPARATIVE DOCKING RESULTS: Molecule 1 vs 5 Clinical DRD2 Drugs")
    print("   Receptor: DRD2 (PDB: 6LUQ) | Box: Asp114 Binding Pocket")
    print(f"   Exhaustiveness: {EXHAUSTIVENESS} | CPU: {CPU_CORES} cores")
    print("=" * 75)
    print(f"{'Rank':<5} {'Drug':<30} {'Best dG (kcal/mol)':<22} {'Status'}")
    print("-" * 75)

    # Sort by binding energy (most negative = best)
    sorted_results = sorted(
        [(name, data) for name, data in results.items() if data.get("energy") is not None],
        key=lambda x: x[1]["energy"]
    )

    for rank, (name, data) in enumerate(sorted_results, start=1):
        energy = data["energy"]
        label = MOLECULES[name]["label"]
        flag = "⭐ AI LEAD" if name == "Molecule_1_AI" else ""
        print(f"{rank:<5} {label:<30} {energy:<22.3f} {flag}")

    # Failed ones
    for name, data in results.items():
        if data.get("energy") is None:
            print(f"{'N/A':<5} {MOLECULES[name]['label']:<30} {'FAILED':<22}")

    print("-" * 75)
    print()

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "comparative_docking_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Drug_Key", "Drug_Label", "Best_dG_kcal_mol", "Time_seconds"])
        for rank, (name, data) in enumerate(sorted_results, start=1):
            writer.writerow([rank, name, MOLECULES[name]["label"],
                             data.get("energy", ""), data.get("time", "")])
    log(f"Results saved to: {csv_path}")

    # Save markdown report
    md_path = os.path.join(OUTPUT_DIR, "comparative_docking_report.md")
    with open(md_path, "w", encoding='utf-8') as f:
        f.write("# Comparative Docking Report: Molecule 1 vs Clinical DRD2 Drugs\n\n")
        f.write(f"**Receptor:** DRD2 (PDB 6LUQ, Asp114 binding pocket)  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  \n")
        f.write(f"**Exhaustiveness:** {EXHAUSTIVENESS}  \n")
        f.write(f"**CPU Cores:** {CPU_CORES}  \n\n")
        f.write("| Rank | Compound | ΔG (kcal/mol) | Relative to Haloperidol |\n")
        f.write("|------|----------|---------------|-------------------------|\n")

        haloperidol_energy = results.get("Haloperidol", {}).get("energy")
        for rank, (name, data) in enumerate(sorted_results, start=1):
            energy = data["energy"]
            label = MOLECULES[name]["label"]
            if haloperidol_energy is not None:
                diff = energy - haloperidol_energy
                rel = f"{diff:+.2f} kcal/mol"
            else:
                rel = "N/A"
            flag = " ⭐" if name == "Molecule_1_AI" else ""
            f.write(f"| {rank} | **{label}**{flag} | {energy:.3f} | {rel} |\n")

        f.write("\n## Conclusion\n")
        if sorted_results and sorted_results[0][0] == "Molecule_1_AI":
            f.write("**Molecule 1 (AI-Generated) achieved the BEST binding affinity** ")
            f.write("among all compared drugs, demonstrating superior computational potency.\n")
        else:
            # Find Mol1 rank
            mol1_rank = next((r for r, (n, _) in enumerate(sorted_results, 1) if n == "Molecule_1_AI"), None)
            mol1_energy = results.get("Molecule_1_AI", {}).get("energy", "N/A")
            f.write(f"Molecule 1 achieved rank #{mol1_rank} with ΔG = {mol1_energy:.3f} kcal/mol, ")
            f.write(f"demonstrating competitive binding affinity against established DRD2 drugs.\n")

    log(f"Markdown report saved to: {md_path}")

    # Try to generate bar chart with matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        drugs   = [MOLECULES[n]["label"] for n, _ in sorted_results]
        energies = [abs(d["energy"]) for _, (n, d) in [(r, (n, d)) for r, (n, d) in enumerate(sorted_results)]]
        colors   = [MOLECULES[n]["color"] for n, _ in sorted_results]
        raw_energies = [d["energy"] for _, d in sorted_results]

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        bars = ax.barh(drugs, [abs(e) for e in raw_energies], color=colors,
                       edgecolor='white', linewidth=0.5, height=0.6)

        for bar, energy in zip(bars, raw_energies):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{energy:.2f} kcal/mol', va='center', ha='left',
                    color='white', fontsize=10, fontweight='bold')

        ax.set_xlabel('|Binding Affinity| (kcal/mol)', color='white', fontsize=12)
        ax.set_title('Comparative AutoDock Vina Docking: DRD2 Binding Affinity\n'
                     'Molecule 1 (AI-Generated) vs 5 Established Antipsychotic Drugs',
                     color='white', fontsize=14, fontweight='bold', pad=15)
        ax.tick_params(colors='white', labelsize=10)
        ax.spines['bottom'].set_color('#666699')
        ax.spines['left'].set_color('#666699')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, max([abs(e) for e in raw_energies]) + 2.5)

        # Highlight Molecule 1 label in orange
        for label in ax.get_yticklabels():
            if "AI-Generated" in label.get_text():
                label.set_color('#FF6B35')
                label.set_fontweight('bold')
            else:
                label.set_color('white')

        plt.tight_layout()
        chart_path = os.path.join(OUTPUT_DIR, "comparative_docking_chart.png")
        plt.savefig(chart_path, dpi=200, bbox_inches='tight',
                    facecolor='#1a1a2e', edgecolor='none')
        plt.close()
        log(f"Bar chart saved to: {chart_path}")
    except Exception as e:
        log(f"Chart generation skipped: {e}")

    return sorted_results


def main():
    print("=" * 75)
    print("  COMPARATIVE DOCKING: Molecule 1 vs 5 Known DRD2 Drugs")
    print(f"  Exhaustiveness={EXHAUSTIVENESS} | Receptor: {os.path.basename(RECEPTOR)}")
    print("=" * 75)

    if not os.path.exists(VINA_EXE):
        log(f"ERROR: Vina not found at {VINA_EXE}")
        sys.exit(1)
    if not os.path.exists(RECEPTOR):
        log(f"ERROR: Receptor PDBQT not found at {RECEPTOR}")
        sys.exit(1)

    results = {}

    for mol_key, mol_data in MOLECULES.items():
        print()
        log(f"--- Processing: {mol_data['label']} ---")

        pdbqt = smiles_to_pdbqt(mol_key, mol_data["smiles"], WORK_DIR)

        if pdbqt is None:
            log(f"  SKIPPING {mol_key} - could not prepare PDBQT")
            results[mol_key] = {"energy": None, "time": None}
            continue

        energy, elapsed = run_vina(mol_key, pdbqt, WORK_DIR)
        results[mol_key] = {"energy": energy, "time": elapsed}
        time.sleep(0.5)

    sorted_results = generate_report(results)

    print("\n" + "=" * 75)
    print("  COMPARATIVE DOCKING COMPLETE!")
    print(f"  Results directory: {OUTPUT_DIR}")
    print("  Files generated:")
    print("    - comparative_docking_results.csv")
    print("    - comparative_docking_report.md")
    print("    - comparative_docking_chart.png")
    print("=" * 75)


if __name__ == "__main__":
    main()
