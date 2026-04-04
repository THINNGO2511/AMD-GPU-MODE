#!/bin/bash
# AMD GPU MODE — Overnight Autonomous Runner
# Runs MLA ratchet + GEMM sweep + MoE probes in parallel via tmux
# Usage: ./overnight_runner.sh

set -e
REPO="/home/claude/AMD-GPU-MODE"
LOG_DIR="$REPO/overnight_logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  AMD GPU MODE — Overnight Autonomous Runner"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"

# Kill existing tmux session if any
tmux kill-session -t overnight 2>/dev/null || true

# Create new tmux session with 3 panes
tmux new-session -d -s overnight -n main

# ─── PANE 0: MLA Ratchet ───
tmux send-keys -t overnight "cd $REPO && bash scripts/mla_ratchet.sh 2>&1 | tee $LOG_DIR/mla_ratchet.log" C-m

# ─── PANE 1: GEMM Config Sweep ───
tmux split-window -h -t overnight
tmux send-keys -t overnight "cd $REPO && bash scripts/gemm_sweep.sh 2>&1 | tee $LOG_DIR/gemm_sweep.log" C-m

# ─── PANE 2: MoE Probes + Loop ───
tmux split-window -v -t overnight
tmux send-keys -t overnight "cd $REPO && bash scripts/moe_runner.sh 2>&1 | tee $LOG_DIR/moe_runner.log" C-m

echo ""
echo "Tmux session 'overnight' started with 3 panes."
echo "Attach with: tmux attach -t overnight"
echo ""
echo "Pane 0: MLA ratchet (every 65 min)"
echo "Pane 1: GEMM config sweep (6 benchmarks/hr)"
echo "Pane 2: MoE probes + experiments"
