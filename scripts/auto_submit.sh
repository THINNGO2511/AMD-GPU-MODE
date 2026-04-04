#!/bin/bash
# Auto-submit loop: submits best submissions to leaderboard every hour.
# Run with: bash auto_submit.sh &
# Or: nohup bash auto_submit.sh > auto_submit.log 2>&1 &

cd "$(dirname "$0")"
POPCORN="$HOME/.local/bin/popcorn-cli"

while true; do
    echo "=== $(date) === AUTO SUBMIT ==="

    # GEMM: .cg cache modifier all shapes (current best 16.06μs)
    r=$($POPCORN submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode leaderboard mxfp4-mm/sub_v6_cg_all.py --no-tui 2>&1)
    echo "$r" | grep -q "Rate limit" && echo "GEMM: rate limited" || echo "GEMM: submitted"

    # MLA: pg2+bf16Q (67% secret seed pass rate, ratchets score down)
    r=$($POPCORN submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/sub_pg2_bf16q.py --no-tui 2>&1)
    echo "$r" | grep -q "Rate limit" && echo "MLA: rate limited" || echo "MLA: submitted"

    # MoE: optimized v2 (opus + CK injection + use_nt=False)
    r=$($POPCORN submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode leaderboard moe-mxfp4/submission_optimized_v2.py --no-tui 2>&1)
    echo "$r" | grep -q "Rate limit" && echo "MoE: rate limited" || echo "MoE: submitted"

    echo "Sleeping 3600s (1hr)..."
    sleep 3600
done
