#!/bin/bash
# =============================================================================
# Standalone Linux Server Runner (No SLURM required)
# 
# Usage:
#   1. Start a terminal multiplexer so it runs when you disconnect:
#      tmux new -s mol_training
#   2. Run this script:
#      bash run_standalone.sh
# =============================================================================

# 1. Activate Environment (Change 'miniconda3' to 'anaconda3' if needed)
source ~/.bashrc
conda activate mol_linux

echo "================================================================="
echo " Starting Standalone Linux Training (Auto-Resume Enabled)"
echo "================================================================="

# Infinite loop: if the script crashes or the system reboots (and you run this again),
# it will automatically find the latest checkpoint and resume.
while true; do
    # Find the latest valid atomic checkpoint
    LATEST_CHECKPOINT=$(ls -t checkpoints/linux_hpc_run/*.pt 2>/dev/null | head -n 1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "[$(date)] No checkpoint found. Starting fresh training..."
        # If you have 1 GPU, use standard python. If you have multiple, you can use torchrun.
        python scripts/train_ppo_linux_hpc.py
    else
        echo "[$(date)] Resuming training from: $LATEST_CHECKPOINT"
        python scripts/train_ppo_linux_hpc.py --resume "$LATEST_CHECKPOINT"
    fi

    # Capture the exit code of the Python script
    EXIT_CODE=$?

    # If the exit code is 0, training finished naturally (e.g., reached total_timesteps)
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully! Exiting loop."
        break
    fi

    # If the exit code is not 0, it crashed (OOM, random error, etc.)
    echo "[$(date)] WARNING: Python script crashed with exit code $EXIT_CODE."
    echo "Restarting automatically in 10 seconds (Press Ctrl+C twice to stop completely)..."
    sleep 10
done
