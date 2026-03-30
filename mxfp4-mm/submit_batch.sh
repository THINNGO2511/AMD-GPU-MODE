#!/bin/bash
# Submit multiple sweep files in sequence, waiting for rate limit
cd /Users/edwardngo/Downloads/code/AMD-GPU-MODE

FILES=(
    "mxfp4-mm/submission_try_a8wfp4.py"
    "mxfp4-mm/submission_sweep_k2048_v2.py"
    "mxfp4-mm/submission_sweep_batch2.py"
    "mxfp4-mm/submission_try_fused.py"
    "mxfp4-mm/submission_probe_compact.py"
)

for f in "${FILES[@]}"; do
    echo "=========================================="
    echo "SUBMITTING: $f"
    echo "TIME: $(date '+%H:%M:%S')"
    echo "=========================================="

    while true; do
        cp "$f" mxfp4-mm/submission.py
        OUTPUT=$(popcorn submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark mxfp4-mm/submission.py --no-tui 2>&1)
        echo "$OUTPUT"

        if echo "$OUTPUT" | grep -q "Rate limit"; then
            WAIT=$(echo "$OUTPUT" | sed -n 's/.*Try again in \([0-9]*\)s.*/\1/p')
            if [ -n "$WAIT" ]; then
                echo "Rate limited, waiting ${WAIT}s + 10s buffer..."
                sleep $((WAIT + 10))
            else
                echo "Rate limited, waiting 300s..."
                sleep 300
            fi
        else
            echo "SUBMITTED SUCCESSFULLY"
            break
        fi
    done

    echo ""
done

echo "ALL DONE at $(date '+%H:%M:%S')"
