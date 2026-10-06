#!/bin/bash
# =============================================================================
# Linux Desktop Auto-Start Setup
# This creates a cron job that automatically runs your training script 
# the moment the PC boots up.
# =============================================================================

PROJECT_DIR=$(pwd)
LOG_FILE="$PROJECT_DIR/logs/autostart.log"
CRON_CMD="@reboot cd $PROJECT_DIR && /bin/bash run_standalone.sh >> $LOG_FILE 2>&1"

# Check if the cron job already exists
crontab -l 2>/dev/null | grep -q "run_standalone.sh"

if [ $? -eq 0 ]; then
    echo "Autostart is already configured!"
else
    # Add the cron job
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "================================================================="
    echo " SUCCESS! Auto-start configured."
    echo "================================================================="
    echo "If the power goes out, the moment this PC turns back on and boots"
    echo "into Linux, it will automatically resume your training in the background."
    echo "You can view the live output anytime by typing:"
    echo "tail -f $LOG_FILE"
fi
