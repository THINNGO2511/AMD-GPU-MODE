#!/bin/bash
# Auto-fire 3 critical submissions when rate limits clear
# Run: nohup bash auto_fire_critical.sh > auto_fire.log 2>&1 &

cd "$(dirname "$0")"
POPCORN="$HOME/.local/bin/popcorn-cli"

echo "=== $(date) === Starting auto-fire for 3 critical submissions ==="

# 1. GEMM fast unshuffle leaderboard (HIGHEST PRIORITY)
echo "Waiting for GEMM leaderboard slot..."
while true; do
    r=$($POPCORN submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode leaderboard mxfp4-mm/submission_fast_unshuffle.py --no-tui 2>&1)
    if echo "$r" | grep -q "Rate limit"; then
        echo "  $(date +%H:%M:%S) GEMM rate limited, retrying in 60s..."
        sleep 60
    else
        echo "=== $(date) === GEMM FAST UNSHUFFLE SUBMITTED ==="
        echo "$r" | grep -E "⏱|✅|❌|Leaderboard|fail|success" | head -20
        break
    fi
done

# 2. MLA zero-copy subsample test
echo "Submitting MLA zero-copy subsample test..."
while true; do
    r=$($POPCORN submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/sub_kv_approx_safe.py --no-tui 2>&1)
    if echo "$r" | grep -q "Rate limit"; then
        echo "  $(date +%H:%M:%S) MLA test rate limited, retrying in 60s..."
        sleep 60
    else
        echo "=== $(date) === MLA ZERO-COPY SUBSAMPLE SUBMITTED ==="
        echo "$r" | grep -E "⏱|✅|❌|Passed|fail|success|mismatch" | head -20
        break
    fi
done

# 3. MLA leaderboard ratchet
echo "Submitting MLA leaderboard ratchet..."
while true; do
    r=$($POPCORN submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/sub_pg2_bf16q.py --no-tui 2>&1)
    if echo "$r" | grep -q "Rate limit"; then
        echo "  $(date +%H:%M:%S) MLA leaderboard rate limited, retrying in 60s..."
        sleep 60
    else
        echo "=== $(date) === MLA LEADERBOARD SUBMITTED ==="
        echo "$r" | grep -E "⏱|✅|❌|Leaderboard|fail|success" | head -20
        break
    fi
done

# 4. MoE leaderboard ratchet
echo "Submitting MoE leaderboard ratchet..."
while true; do
    r=$($POPCORN submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode leaderboard moe-mxfp4/submission_optimized_v2.py --no-tui 2>&1)
    if echo "$r" | grep -q "Rate limit"; then
        echo "  $(date +%H:%M:%S) MoE leaderboard rate limited, retrying in 60s..."
        sleep 60
    else
        echo "=== $(date) === MoE LEADERBOARD SUBMITTED ==="
        echo "$r" | grep -E "⏱|✅|❌|Leaderboard|fail|success" | head -20
        break
    fi
done

echo "=== $(date) === ALL CRITICAL SUBMISSIONS DONE ==="
echo "Check results above. The GEMM fast unshuffle ranked score is the key number."
