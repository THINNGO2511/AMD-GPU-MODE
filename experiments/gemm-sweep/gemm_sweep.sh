#!/bin/bash
# GEMM Config Sweep Runner
# Generates sweep configs, submits as benchmark, logs results
# Rate limit: 6 benchmarks/hr = 1 every 10 minutes
# Alternates between k7168 and k2048 shapes

REPO="/home/claude/AMD-GPU-MODE"
SWEEP_DIR="$REPO/mxfp4-mm/sweep_configs"
LOG_DIR="$REPO/overnight_logs"
RESULTS_FILE="$LOG_DIR/gemm_sweep_results.csv"
BEST_FILE="$LOG_DIR/gemm_best.txt"
STATE_FILE="$LOG_DIR/gemm_sweep_state.txt"

mkdir -p "$LOG_DIR" "$SWEEP_DIR"

echo "=== GEMM Config Sweep Started: $(date) ==="

# Initialize results CSV
if [ ! -f "$RESULTS_FILE" ]; then
    echo "timestamp,shape,index,tag,benchmark_us,raw_output" > "$RESULTS_FILE"
fi

# Initialize best tracker
if [ ! -f "$BEST_FILE" ]; then
    echo "999.0" > "$BEST_FILE"
fi

# Generate configs if not already done
if [ ! -f "$SWEEP_DIR/sweep_k7168_0000.py" ]; then
    echo "Generating sweep configs..."
    cd "$REPO"
    python3 scripts/gemm_sweep_gen.py --shape both --count 150
    echo "Done. $(ls $SWEEP_DIR/*.py 2>/dev/null | wc -l) files generated."
fi

# Resume from last state
SHAPE_IDX=0  # 0 = k7168, 1 = k2048
CONFIG_K7168_IDX=0
CONFIG_K2048_IDX=0

if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    echo "Resuming: k7168 idx=$CONFIG_K7168_IDX, k2048 idx=$CONFIG_K2048_IDX"
fi

TOTAL_SUBMITTED=0
HOUR_COUNT=0

while true; do
    # Alternate shapes: 3 k7168 configs, then 3 k2048 configs (fills 6 slots/hr)
    for SLOT in 1 2 3 4 5 6; do
        if [ $((SLOT % 2)) -eq 1 ]; then
            SHAPE="k7168"
            IDX=$CONFIG_K7168_IDX
        else
            SHAPE="k2048"
            IDX=$CONFIG_K2048_IDX
        fi

        FNAME="sweep_${SHAPE}_$(printf '%04d' $IDX).py"
        FPATH="$SWEEP_DIR/$FNAME"

        if [ ! -f "$FPATH" ]; then
            echo "[$SHAPE] No more configs at index $IDX. Generating more..."
            cd "$REPO"
            python3 scripts/gemm_sweep_gen.py --shape "$SHAPE" --start "$IDX" --count 50
        fi

        if [ ! -f "$FPATH" ]; then
            echo "[$SHAPE] Exhausted all configs. Skipping."
            continue
        fi

        echo ""
        echo "--- GEMM Sweep #$TOTAL_SUBMITTED | $FNAME | $(date) ---"

        # Extract tag from the file
        TAG=$(grep "SWEEP_TAG:" "$FPATH" | head -1 | sed 's/.*SWEEP_TAG: //')
        echo "Config: $TAG"

        # Submit as benchmark
        cd "$REPO"
        OUTPUT=$(popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark "$FPATH" --no-tui 2>&1)
        echo "$OUTPUT" | tail -5

        # Extract benchmark score
        SCORE=$(echo "$OUTPUT" | grep -oP '[\d.]+\s*μs' | head -1 | grep -oP '[\d.]+')
        if [ -z "$SCORE" ]; then
            SCORE=$(echo "$OUTPUT" | grep -oP 'score[:\s]+[\d.]+' | grep -oP '[\d.]+')
        fi

        if [ -n "$SCORE" ]; then
            echo "Benchmark: ${SCORE}μs"
            # Log to CSV (escape commas in output)
            CLEAN_OUTPUT=$(echo "$OUTPUT" | tr '\n' ' ' | tr ',' ';' | cut -c1-200)
            echo "$(date -Iseconds),$SHAPE,$IDX,$TAG,$SCORE,$CLEAN_OUTPUT" >> "$RESULTS_FILE"

            # Check if this is our overall best
            BEST=$(cat "$BEST_FILE")
            if echo "$SCORE < $BEST" | bc -l 2>/dev/null | grep -q 1; then
                echo "$SCORE" > "$BEST_FILE"
                echo "*** NEW GEMM BEST: ${SCORE}μs (was ${BEST}μs) ***"
                echo "*** Config: $TAG ***"
                echo "*** File: $FPATH ***"

                # Copy best file for easy leaderboard submission
                cp "$FPATH" "$REPO/mxfp4-mm/sweep_best.py"
                echo "Copied to mxfp4-mm/sweep_best.py"
            fi
        else
            echo "No score extracted. Output:"
            echo "$OUTPUT" | tail -10
            echo "$(date -Iseconds),$SHAPE,$IDX,$TAG,FAILED,$OUTPUT" >> "$RESULTS_FILE"
        fi

        # Update index
        if [ $((SLOT % 2)) -eq 1 ]; then
            CONFIG_K7168_IDX=$((CONFIG_K7168_IDX + 1))
        else
            CONFIG_K2048_IDX=$((CONFIG_K2048_IDX + 1))
        fi

        # Save state
        cat > "$STATE_FILE" << EOF
CONFIG_K7168_IDX=$CONFIG_K7168_IDX
CONFIG_K2048_IDX=$CONFIG_K2048_IDX
EOF

        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))

        # Rate limit: 6 benchmarks per hour = 10 min between submissions
        echo "Sleeping 10 minutes (rate limit)..."
        sleep 600
    done

    HOUR_COUNT=$((HOUR_COUNT + 1))
    echo ""
    echo "=== Hour $HOUR_COUNT complete. Total submitted: $TOTAL_SUBMITTED ==="
    echo "Best so far: $(cat $BEST_FILE)μs"
    echo "Results: $RESULTS_FILE"
    echo ""

    # Every 3 hours, also submit current best to leaderboard
    if [ $((HOUR_COUNT % 3)) -eq 0 ]; then
        if [ -f "$REPO/mxfp4-mm/sweep_best.py" ]; then
            echo "--- Submitting sweep_best.py to LEADERBOARD ---"
            cd "$REPO"
            OUTPUT=$(popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode leaderboard mxfp4-mm/sweep_best.py --no-tui 2>&1)
            echo "$OUTPUT" | tail -5
        fi
    fi
done
