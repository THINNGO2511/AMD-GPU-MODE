#!/bin/bash
# MLA Ratcheting Loop — submit mla_fp8q_all.py to leaderboard every 65 minutes
# Run this in background: bash auto_mla_ratchet.sh &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POPCORN="$(which popcorn-cli 2>/dev/null || echo "$HOME/.local/bin/popcorn-cli")"
FILE="$SCRIPT_DIR/mixed-mla/mla_fp8q_all.py"
LOG="$SCRIPT_DIR/auto_mla_ratchet.log"

echo "Starting MLA ratchet loop at $(date)" >> "$LOG"
echo "File: $FILE" >> "$LOG"

while true; do
    echo "--- $(date) ---" >> "$LOG"
    $POPCORN submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard "$FILE" --no-tui 2>&1 | tee -a "$LOG"
    echo "Sleeping 65 minutes..." >> "$LOG"
    sleep 3900  # 65 minutes (1hr rate limit + 5min buffer)
done
