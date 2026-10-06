"""
Deep Analysis Script — Last Week's PPO Run
Analyses both molecule_log_gpu (1).csv and progress (2).csv
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── Load Data ────────────────────────────────────────────────────────────────
mol_path  = r"c:\Users\devan\OneDrive\Desktop\molecule_log_gpu (1).csv"
prog_path = r"c:\Users\devan\OneDrive\Desktop\final project\logs\progress (2).csv"

mol_df  = pd.read_csv(mol_path,  on_bad_lines='skip')
prog_df = pd.read_csv(prog_path, on_bad_lines='skip')

# Coerce numerics
num_mol_cols = ['reward','docking_score','qed','mw','diversity_penalty',
                'cumulative_validity','unique_count','topk_bonus','ifars_overlap',
                'critical_overlap','ifars_bonus','wall_clock_hours','episode','timestep']
for c in num_mol_cols:
    if c in mol_df.columns:
        mol_df[c] = pd.to_numeric(mol_df[c], errors='coerce')

valid_df  = mol_df[mol_df['valid'] == True].copy()
invalid_df = mol_df[mol_df['valid'] == False].copy()

# ─── 1. RUN OVERVIEW ──────────────────────────────────────────────────────────
print("=" * 70)
print("  SECTION 1: RUN OVERVIEW")
print("=" * 70)
total_mols    = len(mol_df)
total_valid   = len(valid_df)
total_invalid = len(invalid_df)
unique_smiles = valid_df['smiles'].nunique()
validity_pct  = total_valid / total_mols * 100

total_steps   = prog_df['time/total_timesteps'].max()
total_iters   = prog_df['time/iterations'].max()
wall_hours    = prog_df['time/time_elapsed'].max() / 3600.0
total_eps     = prog_df['molecules/total_episodes'].max()

print(f"  Total molecules generated   : {total_mols:>10,}")
print(f"  Total valid molecules       : {total_valid:>10,}  ({validity_pct:.2f}%)")
print(f"  Total invalid molecules     : {total_invalid:>10,}  ({100-validity_pct:.2f}%)")
print(f"  Unique valid SMILES         : {unique_smiles:>10,}")
print(f"  Duplication rate (valid)    : {(1 - unique_smiles/total_valid)*100:>9.2f}%")
print(f"  Total PPO iterations        : {total_iters:>10,}")
print(f"  Total timesteps             : {total_steps:>10,}")
print(f"  Total episodes              : {total_eps:>10,}")
print(f"  Wall-clock time             : {wall_hours:>9.1f} hours  ({wall_hours/24:.1f} days)")
avg_mols_per_iter = total_mols / total_iters
print(f"  Avg molecules/iteration     : {avg_mols_per_iter:>9.1f}")
print(f"  Avg wall-time/iteration     : {wall_hours*3600/total_iters:>9.1f} seconds")

# ─── 2. DOCKING SCORE ANALYSIS ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 2: DOCKING SCORE ANALYSIS")
print("=" * 70)
docked = valid_df.dropna(subset=['docking_score'])
undocked = valid_df[valid_df['docking_score'].isna()]
print(f"  Molecules successfully docked    : {len(docked):>8,}  ({len(docked)/total_valid*100:.1f}% of valid)")
print(f"  Molecules NOT docked (no score)  : {len(undocked):>8,}  ({len(undocked)/total_valid*100:.1f}% of valid)")
print()
# Filter out clearly anomalous positive docking scores (Vina errors)
docked_clean = docked[docked['docking_score'] < 0]
bad_dock     = docked[docked['docking_score'] >= 0]
print(f"  Anomalous positive scores (Vina parse errors): {len(bad_dock):>5,}")
print()
print("  Clean Docking Score Distribution (kcal/mol):")
percs = [10,25,50,75,90,95,99]
dock_vals = docked_clean['docking_score'].values
for p in percs:
    print(f"    P{p:<2}: {np.percentile(dock_vals, p):>8.3f}")
print(f"    Mean : {docked_clean['docking_score'].mean():>8.3f}")
print(f"    Std  : {docked_clean['docking_score'].std():>8.3f}")
print(f"    Best : {docked_clean['docking_score'].min():>8.3f}")
print(f"    Worst: {docked_clean['docking_score'].max():>8.3f}")
print()

# Binding tier breakdown
tiers = [(-99,-11,'Elite   (< -11)'),(-11,-10,'Strong  (-11 to -10)'),
         (-10,-8, 'Good    (-10 to  -8)'),(-8,-6, 'Moderate ( -8 to  -6)'),
         (-6, 0,  'Weak    (  -6 to   0)')]
print("  Binding Affinity Tier Breakdown:")
for lo,hi,label in tiers:
    count = ((docked_clean['docking_score'] >= lo) & (docked_clean['docking_score'] < hi)).sum()
    pct   = count / len(docked_clean) * 100
    print(f"    {label}  : {count:>6,}  ({pct:5.2f}%)")

# ─── 3. REWARD ANALYSIS ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 3: REWARD ANALYSIS")
print("=" * 70)
print("  All Valid Molecule Rewards:")
for p in [5,25,50,75,95]:
    print(f"    P{p:<2}: {np.percentile(valid_df['reward'], p):>8.3f}")
print(f"    Mean: {valid_df['reward'].mean():>8.3f}   Std: {valid_df['reward'].std():.3f}")
print(f"    Best: {valid_df['reward'].max():>8.3f}   Worst: {valid_df['reward'].min():.3f}")
print()

# Reward evolution over time — split run into quartiles
mol_df_sorted = mol_df.sort_values('timestep')
valid_sorted  = valid_df.sort_values('timestep')
qs = np.array_split(valid_sorted, 4)
print("  Reward Evolution Across Run Quartiles:")
print(f"  {'Quartile':<12} {'Timestep Range':<24} {'Mean Reward':>12} {'Mean Docking':>14} {'Validity%':>10}")
print(f"  {'-'*12} {'-'*24} {'-'*12} {'-'*14} {'-'*10}")
total_mols_qt = len(mol_df) // 4
for i, q in enumerate(qs):
    q_docked = q.dropna(subset=['docking_score'])
    q_docked_clean = q_docked[q_docked['docking_score'] < 0]
    ts_range = f"{q['timestep'].min():.0f}-{q['timestep'].max():.0f}"
    val_pct  = len(q) / total_mols_qt * 100 if i < 3 else len(q) / (len(mol_df) - 3*total_mols_qt) * 100
    # validity in this quarter = valid in Q / all mols in Q
    all_q = mol_df_sorted.iloc[i*total_mols_qt:(i+1)*total_mols_qt] if i < 3 else mol_df_sorted.iloc[3*total_mols_qt:]
    val_rate = len(q) / len(all_q) * 100
    mean_dock = q_docked_clean['docking_score'].mean() if len(q_docked_clean) > 0 else float('nan')
    print(f"  Q{i+1:<11} {ts_range:<24} {q['reward'].mean():>12.3f} {mean_dock:>14.3f} {val_rate:>9.2f}%")

# ─── 4. QED (DRUG-LIKENESS) ANALYSIS ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 4: QED (DRUG-LIKENESS) ANALYSIS")
print("=" * 70)
print(f"  Mean QED        : {valid_df['qed'].mean():.4f}")
print(f"  Median QED      : {valid_df['qed'].median():.4f}")
print(f"  Std QED         : {valid_df['qed'].std():.4f}")
print(f"  QED > 0.5 (drug-like)     : {(valid_df['qed'] > 0.5).sum():>7,}  ({(valid_df['qed']>0.5).mean()*100:.1f}%)")
print(f"  QED > 0.7 (highly drug-like): {(valid_df['qed'] > 0.7).sum():>5,}  ({(valid_df['qed']>0.7).mean()*100:.1f}%)")
print(f"  QED > 0.8 (excellent)       : {(valid_df['qed'] > 0.8).sum():>5,}  ({(valid_df['qed']>0.8).mean()*100:.1f}%)")
print(f"  Max QED         : {valid_df['qed'].max():.4f}")

# ─── 5. MOLECULAR WEIGHT ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 5: MOLECULAR WEIGHT DISTRIBUTION")
print("=" * 70)
valid_mw = valid_df[valid_df['mw'] > 0]
lipinski_ok = valid_mw[(valid_mw['mw'] >= 160) & (valid_mw['mw'] <= 550)]
print(f"  Mean MW                          : {valid_mw['mw'].mean():.1f} Da")
print(f"  Median MW                        : {valid_mw['mw'].median():.1f} Da")
print(f"  In Lipinski MW range (160-550 Da): {len(lipinski_ok):>7,}  ({len(lipinski_ok)/len(valid_mw)*100:.1f}%)")
print(f"  Too small (< 160 Da)             : {(valid_mw['mw'] < 160).sum():>7,}  ({(valid_mw['mw']<160).mean()*100:.1f}%)")
print(f"  Too large (> 550 Da)             : {(valid_mw['mw'] > 550).sum():>7,}  ({(valid_mw['mw']>550).mean()*100:.1f}%)")

# ─── 6. DIVERSITY & EXPLORATION ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 6: DIVERSITY & EXPLORATION")
print("=" * 70)
div_pen_fired = valid_df[valid_df['diversity_penalty'] > 0] if 'diversity_penalty' in valid_df.columns else pd.DataFrame()
topk_bonus_fired = valid_df[valid_df['topk_bonus'] > 0]   if 'topk_bonus' in valid_df.columns else pd.DataFrame()

print(f"  Total unique valid SMILES       : {unique_smiles:>7,}")
print(f"  Unique / Total valid ratio      : {unique_smiles/total_valid:.4f}  (1.0 = perfect diversity)")
if len(div_pen_fired) > 0:
    print(f"  Diversity penalty fired         : {len(div_pen_fired):>7,}  times ({len(div_pen_fired)/total_valid*100:.1f}% of valid)")
if len(topk_bonus_fired) > 0:
    print(f"  Top-K scaffold bonus fired      : {len(topk_bonus_fired):>7,}  times ({len(topk_bonus_fired)/total_valid*100:.1f}% of valid)")

# Unique molecule discovery rate over time
prog_df['molecules/unique_count'] = pd.to_numeric(prog_df['molecules/unique_count'], errors='coerce')
unique_growth = prog_df[['time/iterations','molecules/unique_count']].dropna()
first_half  = unique_growth[unique_growth['time/iterations'] <= total_iters/2]
second_half = unique_growth[unique_growth['time/iterations'] >  total_iters/2]
if len(first_half) > 1 and len(second_half) > 1:
    rate1 = (first_half['molecules/unique_count'].max() - first_half['molecules/unique_count'].min()) / len(first_half)
    rate2 = (second_half['molecules/unique_count'].max() - second_half['molecules/unique_count'].min()) / len(second_half)
    print(f"  New unique mol discovery rate:")
    print(f"    First half of run   : {rate1:.2f} new unique mols/iteration")
    print(f"    Second half of run  : {rate2:.2f} new unique mols/iteration")
    decay = (rate1 - rate2) / rate1 * 100 if rate1 > 0 else 0
    print(f"    Discovery slowdown  : {decay:.1f}% — {'significant mode narrowing' if decay > 50 else 'healthy exploration'}")

# ─── 7. PPO TRAINING DYNAMICS ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 7: PPO TRAINING DYNAMICS")
print("=" * 70)
for col, label in [
    ('train/policy_gradient_loss', 'Policy Loss      '),
    ('train/value_loss',           'Value Loss       '),
    ('train/entropy_loss',         'Entropy Loss     '),
    ('train/approx_kl',            'Approx KL Div    '),
    ('train/clip_fraction',        'Clip Fraction    '),
    ('train/explained_variance',   'Explained Var    '),
    ('train/learning_rate',        'Learning Rate    '),
]:
    if col in prog_df.columns:
        s = pd.to_numeric(prog_df[col], errors='coerce').dropna()
        print(f"  {label}: start={s.iloc[0]:.5f}  end={s.iloc[-1]:.5f}  "
              f"mean={s.mean():.5f}  std={s.std():.5f}")

# Early stopping (KL divergence breaches)
if 'train/approx_kl' in prog_df.columns:
    kl_s = pd.to_numeric(prog_df['train/approx_kl'], errors='coerce').dropna()
    kl_breaches = (kl_s > 0.015).sum()
    print(f"\n  Target KL (0.015) breaches: {kl_breaches} / {len(kl_s)} iterations  ({kl_breaches/len(kl_s)*100:.1f}%)")

# ─── 8. TOP MOLECULES ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 8: TOP 10 MOLECULES BY REWARD")
print("=" * 70)
top_reward = valid_df.nlargest(10, 'reward')[['smiles','reward','docking_score','qed','mw','topk_bonus','diversity_penalty']]
for i, row in top_reward.reset_index(drop=True).iterrows():
    print(f"\n  #{i+1}  Reward: {row['reward']:.3f}  Docking: {row['docking_score']:.3f}  QED: {row['qed']:.3f}  MW: {row['mw']:.1f}")
    print(f"       SMILES: {row['smiles'][:90]}")

print("\n" + "=" * 70)
print("  TOP 10 MOLECULES BY DOCKING SCORE (Best Binders)")
print("=" * 70)
top_dock = docked_clean.nsmallest(10, 'docking_score')[['smiles','docking_score','reward','qed','mw']]
for i, row in top_dock.reset_index(drop=True).iterrows():
    print(f"\n  #{i+1}  Docking: {row['docking_score']:.3f}  Reward: {row['reward']:.3f}  QED: {row['qed']:.3f}  MW: {row['mw']:.1f}")
    print(f"       SMILES: {row['smiles'][:90]}")

# ─── 9. IF-ARS AUDIT ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 9: IF-ARS / PLIP AUDIT")
print("=" * 70)
ifars_cols = ['ifars_overlap','critical_overlap','ifars_bonus']
for col in ifars_cols:
    if col in mol_df.columns:
        s = pd.to_numeric(mol_df[col], errors='coerce').fillna(0)
        nonzero = (s != 0).sum()
        print(f"  {col:<25}: max={s.max():.4f}  nonzero_rows={nonzero}  (should be >0 in next run)")

asp114 = mol_df['asp114_hit'] if 'asp114_hit' in mol_df.columns else pd.Series([False]*len(mol_df))
asp114_true = (asp114 == True).sum()
print(f"  asp114_hit == True          : {asp114_true}  (was 0 — fixed for next run)")

# How many molecules qualified for PLIP analysis (docking <= -5.0)?
qualified = docked_clean[docked_clean['docking_score'] <= -5.0]
print(f"\n  Molecules that SHOULD have triggered PLIP (dock <= -5.0): {len(qualified):,}")
print(f"  → These all got ifars_bonus=0.0 due to the subprocess/race-condition bugs")
print(f"  → In the next run, these {len(qualified):,} evaluations will feed real PLIP signal")

# ─── 10. CURRICULUM ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 10: CURRICULUM PHASE ANALYSIS")
print("=" * 70)
if 'curriculum_phase' in mol_df.columns:
    phase_counts = mol_df['curriculum_phase'].value_counts()
    print("  Molecule counts per curriculum phase:")
    for phase, count in phase_counts.items():
        print(f"    Phase '{phase}': {count:>8,}  mols  ({count/len(mol_df)*100:.1f}%)")
elif 'curriculum/phase' in prog_df.columns:
    phases = prog_df['curriculum/phase'].value_counts()
    print("  Iterations per curriculum phase (from progress log):")
    for phase, count in phases.items():
        print(f"    Phase {phase}: {count:>5} iterations")

# ─── 11. WALL-CLOCK EFFICIENCY ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 11: WALL-CLOCK EFFICIENCY")
print("=" * 70)
fps = pd.to_numeric(prog_df['time/fps'], errors='coerce').dropna()
print(f"  Throughput (fps = mols/sec):")
print(f"    Mean : {fps.mean():.2f}  |  Median: {fps.median():.2f}  |  Peak: {fps.max():.2f}  |  Min: {fps.min():.2f}")
print(f"  Slowest 5% of iterations (fps < {np.percentile(fps,5):.2f}):")
slow = prog_df[pd.to_numeric(prog_df['time/fps'], errors='coerce') < np.percentile(fps, 5)]
print(f"    {len(slow)} slow iterations detected — likely VRAM OOM pressure or docking queue stalls")
total_wall_sec = prog_df['time/time_elapsed'].max()
print(f"  Total wall time   : {total_wall_sec/3600:.2f} hours")
print(f"  Avg per iteration : {total_wall_sec/total_iters:.1f} seconds")
print(f"  Efficiency ratio  : {total_mols / (total_wall_sec/3600):.0f} molecules/hour")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)
