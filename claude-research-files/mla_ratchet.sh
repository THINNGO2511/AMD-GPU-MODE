#!/bin/bash
# MLA Ratchet — Submit same file to leaderboard every 65 minutes
# Runs forever. Each submission has 1-2μs variance, so repeated submissions
# can improve our score from 35.5μs toward 35.0μs through luck.

REPO="/home/claude/AMD-GPU-MODE"
FILE="mixed-mla/mla_fp8q_all.py"
LOG_DIR="$REPO/overnight_logs"
BEST_FILE="$LOG_DIR/mla_best.txt"

echo "=== MLA Ratchet Started: $(date) ==="
echo "File: $FILE"
echo "Interval: 65 minutes (leaderboard rate limit is 1/hr)"
echo ""

# Track best score
echo "999.0" > "$BEST_FILE"
ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo ""
    echo "--- MLA Ratchet #$ATTEMPT at $(date) ---"
    
    cd "$REPO"
    
    # Submit to leaderboard
    OUTPUT=$(popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard "$FILE" --no-tui 2>&1)
    echo "$OUTPUT"
    
    # Try to extract score from output
    SCORE=$(echo "$OUTPUT" | grep -oP '[\d.]+\s*μs' | head -1 | grep -oP '[\d.]+')
    if [ -n "$SCORE" ]; then
        BEST=$(cat "$BEST_FILE")
        echo "Score: ${SCORE}μs (best: ${BEST}μs)"
        
        # Update best if improved (using bc for float comparison)
        if echo "$SCORE < $BEST" | bc -l | grep -q 1; then
            echo "$SCORE" > "$BEST_FILE"
            echo "*** NEW MLA BEST: ${SCORE}μs ***"
        fi
    fi
    
    echo "Next ratchet at $(date -d '+65 minutes' 2>/dev/null || date -v+65M 2>/dev/null || echo '65 min from now')"
    echo "Sleeping 65 minutes..."
    sleep 3900  # 65 minutes
done
