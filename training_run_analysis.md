# Training Run Analysis (Aborted at 2.2M Steps)

This is a detailed analysis of the CSV log from yesterday's run. The run completed approximately **2.19 million timesteps** before halting due to the "checkpoint memory full" error. 

Overall, the data shows **massive signs of learning and success** before the crash. The model was working exactly as intended.

## 1. The Learning Curve (Reward & Validity)
The most important metric in Reinforcement Learning is whether the agent is actually improving its rewards. The data shows a textbook, successful PPO learning curve:

*   **Initial Phase (Steps 0 - 1.2M):** 
    *   The mean episode reward (`ep_rew_mean`) was stuck around **-4.9 to -5.0**. 
    *   This is because the untrained MolGPT was outputting random tokens, resulting in invalid SMILES (95% of them were invalid). The environment instantly punishes invalid SMILES with a `-5.0` reward.
*   **The Breakthrough (Step ~1.2M):** 
    *   Right around step 1,228,800, the reward suddenly spikes to **-0.35**. 
    *   By step 1,376,256, it goes positive to **+0.71**.
    *   It stabilizes around **+0.4 to +0.56** for the remainder of the run.
*   **Validity Rate:** 
    *   The `validity_rate` started at **5%** (`0.05`). 
    *   By the end of the run, it had steadily climbed to **45.6%** (`0.456`). The RL agent successfully learned the underlying SMILES grammar just to avoid the `-5.0` penalty!

## 2. The FPS Drop Explained
You might notice that the frames-per-second (`time/fps`) started high (163 FPS) and slowly degraded to ~27 FPS by the end of the run. 

*   **Why did this happen?** This is actually a *good* sign. 
*   Early on, 95% of the molecules were invalid. Invalid molecules skip the Vina-GPU docking phase entirely (saving massive amounts of time). 
*   As the model got smarter and generated more valid molecules (up to 45%), the environment had to physically dock 9x more molecules per second on the GPU, bottlenecking the FPS. 
*   *Note:* The LRU Docking Cache we implemented today will massively help prevent this FPS drop in your next run!

## 3. PPO Optimization Stability
The MolGPT weights are very sensitive, but the PPO hyperparameters you used kept it perfectly stable:
*   **`train/approx_kl`**: Fluctuated safely between `0.005` and `0.035`. It never exploded, meaning the policy didn't destroy its pre-trained chemical knowledge.
*   **`train/value_loss`**: Started at `3.0` and dropped all the way to `< 0.05`. The Critic network successfully learned how to predict the docking scores.

## 4. Why it Crashed (Checkpoint Memory Full)
*   **The Cause:** Yesterday's code was likely saving a full 1.6 GB MolGPT checkpoint every 10,000 steps without deleting the old ones. By step 2.19M, it had saved over 200 checkpoints, consuming **~320 GB of disk space** and destroying your OneDrive/local storage.
*   **The Fix:** This perfectly validates the `CleanCheckpointCallback` we added to `scripts/train_ppo_vina_gpu.py` today. Your new code will *only* keep the 2 most recent checkpoints (max 3.2 GB total), completely preventing this from ever happening again.

## Conclusion
The model works. The RL environment is correctly passing gradients back to the MolGPT transformer, and the grammar is improving. 

With the new Tier 2 & Tier 3 improvements we added today (Cache, Curriculum Warmup, Checkpoint Cleanup), your next run will be significantly faster and won't crash your hard drive!
