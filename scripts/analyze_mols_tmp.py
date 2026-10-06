import pandas as pd
import numpy as np

file_path = r"c:\Users\devan\OneDrive\Desktop\molecule_log_gpu (1).csv"

try:
    df = pd.read_csv(file_path, on_bad_lines='skip') # just in case some smiles have commas
    print(f"Total rows read: {len(df)}")
    
    valid_df = df[df['valid'] == True]
    print(f"Total valid molecules: {len(valid_df)}")
    print(f"Overall validity rate: {len(valid_df)/len(df)*100:.2f}%")
    print(f"Unique valid smiles: {valid_df['smiles'].nunique()}")
    
    if len(valid_df) > 0:
        # Convert numeric columns safely
        for col in ['reward', 'docking_score', 'qed', 'mw', 'ifars_overlap']:
            valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce')
            
        print("\n--- Summary of Valid Molecules ---")
        print(valid_df[['reward', 'docking_score', 'qed', 'mw']].describe().to_string())
        
        print("\n--- Top 5 by Reward ---")
        top_reward = valid_df.nlargest(5, 'reward')[['smiles', 'reward', 'docking_score', 'qed', 'ifars_overlap', 'asp114_hit']]
        print(top_reward.to_string())
        
        print("\n--- Top 5 by Docking Score (lowest is best) ---")
        top_docking = valid_df.nsmallest(5, 'docking_score')[['smiles', 'docking_score', 'reward', 'qed', 'ifars_overlap', 'asp114_hit']]
        print(top_docking.to_string())
        
        if 'asp114_hit' in valid_df.columns:
            hits = valid_df[valid_df['asp114_hit'] == True]
            print(f"\nTotal ASP114 hits: {len(hits)}")
            if len(hits) > 0:
                print("Average docking score of ASP114 hits:", hits['docking_score'].mean())
                
        print("\n--- Correlation of Reward with other metrics ---")
        print(valid_df[['reward', 'docking_score', 'qed', 'mw', 'ifars_overlap']].corr()['reward'].to_string())

except Exception as e:
    print(f"Error analyzing CSV: {e}")
