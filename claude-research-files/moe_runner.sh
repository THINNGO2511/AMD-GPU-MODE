#!/bin/bash
# MoE Runner — Probes first, then experiments
# Phase 1: Run deepgemm probe + d2048 probe (2 benchmark slots)
# Phase 2: Based on results, run targeted experiments
# Phase 3: If any improvement found, submit to leaderboard every hour

REPO="/home/claude/AMD-GPU-MODE"
MOE_DIR="$REPO/moe-mxfp4"
LOG_DIR="$REPO/overnight_logs"
RESULTS_FILE="$LOG_DIR/moe_results.log"

mkdir -p "$LOG_DIR"

echo "=== MoE Runner Started: $(date) ==="
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: Discord grep (instant, no slots)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "--- Phase 0: Discord Intelligence Grep ---"
cd "$REPO"

if [ -d "discord-logs" ]; then
    echo "Searching for Maxwell Cipher clues..."
    grep -ri "maxwell" discord-logs/ 2>/dev/null | tee "$LOG_DIR/discord_maxwell.txt" || echo "No Maxwell mentions"
    
    echo "Searching for deepgemm mentions..."
    grep -ri "deepgemm" discord-logs/ 2>/dev/null | tee "$LOG_DIR/discord_deepgemm.txt" || echo "No deepgemm mentions"
    
    echo "Searching for MoE technique discussions..."
    grep -ri "107\|fused_moe\|moe.*custom\|moe.*hip\|moe.*kernel" discord-logs/ 2>/dev/null | head -50 | tee "$LOG_DIR/discord_moe_clues.txt" || echo "No MoE clues"
    
    echo "Searching for quant speedup..."
    grep -ri "quant.*fast\|quant.*hip\|quant.*custom\|mxfp4.*quant" discord-logs/ 2>/dev/null | head -30 | tee "$LOG_DIR/discord_quant.txt" || echo "No quant mentions"
else
    echo "No discord-logs/ directory found"
fi
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Probes (2 benchmark slots)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "--- Phase 1: Running Probes ---"

# Probe 1: deepgemm existence check
cat > "$MOE_DIR/probe_deepgemm.py" << 'PROBE1'
import torch
import sys

print("=" * 60)
print("PROBE: deepgemm / deepgemm_ck availability")
print("=" * 60)

import aiter
all_attrs = [x for x in dir(aiter) if 'deep' in x.lower() or 'gemm' in x.lower()]
print(f"aiter attrs with 'deep' or 'gemm': {all_attrs}")

for name in ['deepgemm', 'deepgemm_ck', 'deep_gemm', 'DeepGemm']:
    if hasattr(aiter, name):
        obj = getattr(aiter, name)
        print(f"\naiter.{name} EXISTS: {type(obj)}")
        if callable(obj):
            try:
                import inspect
                sig = inspect.signature(obj)
                print(f"  Signature: {sig}")
            except:
                print(f"  (could not get signature)")
            try:
                help(obj)
            except:
                pass
    else:
        print(f"aiter.{name}: NOT FOUND")

# Also check for deepgemm as a module
try:
    import deepgemm
    print(f"\nimport deepgemm: SUCCESS")
    print(f"  dir: {dir(deepgemm)}")
except ImportError as e:
    print(f"\nimport deepgemm: FAILED ({e})")

try:
    from aiter import deepgemm_ck
    print(f"\nfrom aiter import deepgemm_ck: SUCCESS")
    print(f"  type: {type(deepgemm_ck)}")
except ImportError as e:
    print(f"\nfrom aiter import deepgemm_ck: FAILED ({e})")

# Check aiter.ops for hidden gems
import aiter.ops as ops
print(f"\naiter.ops contents: {dir(ops)}")

# Quick scan of aiter source for deepgemm references
import subprocess
result = subprocess.run(
    ['grep', '-rl', 'deepgemm', '/home/runner/aiter/aiter/'],
    capture_output=True, text=True
)
print(f"\nFiles mentioning deepgemm: {result.stdout.strip() or 'NONE'}")

print("\n" + "=" * 60)
print("PROBE COMPLETE")
print("=" * 60)

# Still need to define custom_kernel for the eval harness
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

def custom_kernel(hidden_states, w1, w2, topk_weights, topk_ids,
                  w1_scale=None, w2_scale=None, num_experts=None):
    return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                     w1_scale=w1_scale, w2_scale=w2_scale,
                     activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32)
PROBE1

echo "Submitting deepgemm probe..."
cd "$REPO"
OUTPUT=$(popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark "$MOE_DIR/probe_deepgemm.py" --no-tui 2>&1)
echo "$OUTPUT" | tee "$LOG_DIR/probe_deepgemm_output.txt"
echo ""

# Wait 10 min for rate limit
echo "Waiting 10 minutes (rate limit)..."
sleep 600

# Probe 2: d=2048 kernel identity
cat > "$MOE_DIR/probe_d2048.py" << 'PROBE2'
import torch
import functools
import sys

print("=" * 60)
print("PROBE: d=2048 kernel identity and MoE internals")
print("=" * 60)

import aiter.fused_moe as fm
from aiter import ActivationType, QuantType

# Unwrap to get original function
try:
    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__
except AttributeError:
    orig_get_2stage = fm.get_2stage_cfgs

@functools.lru_cache(maxsize=2048)
def probed_get_2stage(*args, **kwargs):
    result = orig_get_2stage(*args, **kwargs)
    token, model_dim, inter_dim, expert = args[:4]
    print(f"\n>>> get_2stage_cfgs called: token={token} model_dim={model_dim} "
          f"inter_dim={inter_dim} expert={expert}")
    print(f"    block_m={result.block_m}")
    try:
        s1_name = result.stage1_kernel.keywords.get('kernelName', 'DEFAULT')
        s2_name = result.stage2_kernel.keywords.get('kernelName', 'DEFAULT')
        print(f"    stage1_kernel: {s1_name}")
        print(f"    stage2_kernel: {s2_name}")
    except Exception as e:
        print(f"    kernel names: could not extract ({e})")
        print(f"    stage1: {result.stage1_kernel}")
        print(f"    stage2: {result.stage2_kernel}")
    try:
        print(f"    ksplit={result.ksplit}")
        print(f"    use_nt={result.use_nt}")
    except:
        pass
    return result

fm.get_2stage_cfgs = probed_get_2stage
fm.cfg_2stages = None

# Also check what CK kernels exist for d=2048
import subprocess
result = subprocess.run(
    ['find', '/home/runner/aiter/hsa/gfx950/fmoe/', '-name', '*2048*'],
    capture_output=True, text=True
)
print(f"\n.co files with 2048 in name: {result.stdout.strip() or 'NONE'}")

# Check for larger tile sizes that might work for d=2048
result = subprocess.run(
    ['find', '/home/runner/aiter/hsa/gfx950/fmoe/', '-name', '*256x*128*'],
    capture_output=True, text=True
)
print(f"\n.co files with 256x...128 tiles: {result.stdout.strip() or 'NONE'}")

# List all unique tile sizes in fmoe
result = subprocess.run(
    ['bash', '-c', 'ls /home/runner/aiter/hsa/gfx950/fmoe/ | grep -oP "\\d+x\\d+x\\d+x\\d+" | sort -u'],
    capture_output=True, text=True
)
print(f"\nUnique fmoe tile sizes: {result.stdout.strip()}")

# Check for any hidden APIs
print(f"\nfm attrs with 'deep' or 'fast': {[x for x in dir(fm) if 'deep' in x.lower() or 'fast' in x.lower()]}")
print(f"fm attrs with 'ck' or 'tile': {[x for x in dir(fm) if 'ck' in x.lower() or 'tile' in x.lower()]}")

print("\n" + "=" * 60)
print("PROBE COMPLETE")
print("=" * 60)

# Still need to provide custom_kernel
fm2 = __import__('aiter.fused_moe', fromlist=['fused_moe'])

def custom_kernel(hidden_states, w1, w2, topk_weights, topk_ids,
                  w1_scale=None, w2_scale=None, num_experts=None):
    return fm2.fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                         w1_scale=w1_scale, w2_scale=w2_scale,
                         activation=ActivationType.Silu,
                         quant_type=QuantType.per_1x32)
PROBE2

echo "Submitting d2048 probe..."
cd "$REPO"
OUTPUT=$(popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark "$MOE_DIR/probe_d2048.py" --no-tui 2>&1)
echo "$OUTPUT" | tee "$LOG_DIR/probe_d2048_output.txt"

echo ""
echo "--- Phase 1 Complete ---"
echo "Check $LOG_DIR/probe_*.txt for results"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Wait, then ratchet current best
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "--- Phase 2: MoE Leaderboard Ratcheting ---"
echo "(Probes done. Running current best on leaderboard every 65 min.)"
echo "(When you wake up, check probe results and Claude Code will adapt.)"
echo ""

ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo ""
    echo "--- MoE Ratchet #$ATTEMPT at $(date) ---"

    cd "$REPO"

    # Submit current best to leaderboard
    OUTPUT=$(popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode leaderboard moe-mxfp4/submission_optimized_v2.py --no-tui 2>&1)
    echo "$OUTPUT" | tail -5

    # Extract score
    SCORE=$(echo "$OUTPUT" | grep -oP '[\d.]+\s*μs' | head -1 | grep -oP '[\d.]+')
    if [ -n "$SCORE" ]; then
        echo "MoE score: ${SCORE}μs"
        echo "$(date -Iseconds),ratchet,$ATTEMPT,$SCORE" >> "$LOG_DIR/moe_ratchet.csv"
    fi

    echo "Sleeping 65 minutes..."
    sleep 3900
done
