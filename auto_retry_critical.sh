#!/bin/bash
# Auto-retry 2 critical experiments + leaderboard ratchets when runner recovers
# Checks every 5 min. Stops after all succeed.
cd "$(dirname "$0")"
POPCORN="$HOME/.local/bin/popcorn-cli"

echo "=== $(date) === Waiting for runner recovery ==="

# 1. MoE zhubenzhu v2 benchmark
MOE_DONE=0
while [ $MOE_DONE -eq 0 ]; do
    r=$($POPCORN submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark moe-mxfp4/sub_zhubenzhu_v2.py --no-tui 2>&1)
    if echo "$r" | grep -q "No HIP GPUs"; then
        echo "$(date +%H:%M) MoE: No GPU, retry in 5min..."
        sleep 300
    elif echo "$r" | grep -q "Rate limit"; then
        echo "$(date +%H:%M) MoE: Rate limited, retry in 2min..."
        sleep 120
    else
        echo "=== $(date) MoE ZHUBENZHU V2 RESULT ==="
        echo "$r" | grep -E "⏱|✅|❌|error" | head -15
        MOE_DONE=1
    fi
done

# 2. MLA pg2 subsample test
MLA_DONE=0
while [ $MLA_DONE -eq 0 ]; do
    r=$($POPCORN submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/sub_kv_subsample_pg2.py --no-tui 2>&1)
    if echo "$r" | grep -q "No HIP GPUs"; then
        echo "$(date +%H:%M) MLA: No GPU, retry in 5min..."
        sleep 300
    elif echo "$r" | grep -q "Rate limit"; then
        echo "$(date +%H:%M) MLA: Rate limited, retry in 2min..."
        sleep 120
    else
        echo "=== $(date) MLA PG2 SUBSAMPLE RESULT ==="
        echo "$r" | grep -E "⏱|✅|❌|mismatch|error|seed" | head -15
        MLA_DONE=1
    fi
done

# 3. Leaderboard ratchets
for lb_file in "amd-mxfp4-mm mxfp4-mm/sub_v6_cg_all.py" "amd-mixed-mla mixed-mla/sub_pg2_bf16q.py" "amd-moe-mxfp4 moe-mxfp4/submission_optimized_v2.py"; do
    lb=$(echo $lb_file | cut -d' ' -f1)
    f=$(echo $lb_file | cut -d' ' -f2)
    while true; do
        r=$($POPCORN submit --gpu MI355X --leaderboard $lb --mode leaderboard $f --no-tui 2>&1)
        if echo "$r" | grep -q "No HIP GPUs\|Rate limit"; then
            sleep 120
        else
            echo "=== $(date) $lb LEADERBOARD ==="
            echo "$r" | grep -E "⏱|✅|❌" | head -5
            break
        fi
    done
done

echo "=== $(date) ALL DONE ==="
