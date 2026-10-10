import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = r"C:\Users\devan\OneDrive\Desktop\final project\logs\hpc_max_run"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data collected from Vina output (Exhaustiveness 64)
results = [
    ("Molecule_1_AI", "Molecule 1 (AI-Generated)", -9.530, "#FF6B35", 69),
    ("Risperidone", "Risperidone", -9.457, "#96CEB4", 74),
    ("Clozapine", "Clozapine", -9.203, "#45B7D1", 8),
    ("Aripiprazole", "Aripiprazole", -9.156, "#FFEAA7", 104),
    ("Olanzapine", "Olanzapine", -8.500, "#DDA0DD", 8),
    ("Haloperidol", "Haloperidol (Reference)", -8.231, "#4ECDC4", 63),
]

def generate():
    # 1. Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "comparative_docking_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Drug_Key", "Drug_Label", "Best_dG_kcal_mol", "Time_seconds"])
        for rank, (key, label, energy, color, time) in enumerate(results, 1):
            writer.writerow([rank, key, label, energy, time])
    print(f"Saved CSV: {csv_path}")

    # 2. Save Markdown Report
    md_path = os.path.join(OUTPUT_DIR, "comparative_docking_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comparative Docking Report: Molecule 1 vs Clinical DRD2 Drugs\n\n")
        f.write("**Receptor:** DRD2 (PDB 6LUQ, Asp114 binding pocket)  \n")
        f.write("**Exhaustiveness:** 64  \n\n")
        f.write("| Rank | Compound | \u0394G (kcal/mol) | Relative to Haloperidol |\n")
        f.write("|------|----------|---------------|-------------------------|\n")
        
        halo_e = -8.231
        for rank, (key, label, energy, color, time) in enumerate(results, 1):
            diff = energy - halo_e
            rel = f"{diff:+.2f} kcal/mol"
            flag = " (AI LEAD)" if key == "Molecule_1_AI" else ""
            f.write(f"| {rank} | **{label}**{flag} | {energy:.3f} | {rel} |\n")
            
        f.write("\n## Conclusion\n")
        f.write("**Molecule 1 (AI-Generated) achieved the BEST binding affinity (-9.530 kcal/mol)** ")
        f.write("among all compared drugs, demonstrating superior computational potency against clinical standards.\n")
    print(f"Saved Markdown: {md_path}")

    # 3. Save Chart
    drugs = [r[1] for r in results]
    energies = [abs(r[2]) for r in results]
    raw_energies = [r[2] for r in results]
    colors = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.barh(drugs[::-1], energies[::-1], color=colors[::-1],
                   edgecolor='white', linewidth=0.5, height=0.6)

    for bar, energy in zip(bars, raw_energies[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{energy:.3f} kcal/mol', va='center', ha='left',
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
    ax.set_xlim(0, max(energies) + 2.5)

    # Highlight Molecule 1 label in orange
    for label in ax.get_yticklabels():
        if "AI-Generated" in label.get_text():
            label.set_color('#FF6B35')
            label.set_fontweight('bold')
        else:
            label.set_color('white')

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "comparative_docking_chart.png")
    plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"Saved Chart: {chart_path}")

if __name__ == "__main__":
    generate()
