# ROUND 14: Write GEMM configs + Probe MoE kernels

## CRITICAL CONTEXT
We need top 20 on each problem to get ANY aggregate points:
- MLA: ~#14 (IN top 20, keep ratcheting)
- MoE: ~#65 (need 163→143μs for top 20)
- GEMM: ~#160 (need 15.7→9μs for top 20)

## PART 1: Submit experiments

Setup:
cp claude-research-files/gemm_write_configs.py mxfp4-mm/
cp claude-research-files/gemm_write_configs_ksplit4.py mxfp4-mm/
cp claude-research-files/moe_probe_kernels.py moe-mxfp4/

Submit:
1. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_write_configs.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_write_configs_ksplit4.py --no-tui
3. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_probe_kernels.py --no-tui
4. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui

ALSO: Check if mla_auto_splits leaderboard result came back. If it scored < 36.4μs, that's our new MLA best.

## PART 2: Continue source exploration (while waiting)

### 2A. List ALL CK 2-stage FP4 silu kernel names available
```bash
# Full list of gemm1 (stage 1) kernels
ls /home/runner/aiter/hsa/gfx950/fmoe/ | grep "ck2stages_gemm1.*FP4.*silu" | sed 's/.co$//' | sort
# Full list of gemm2 (stage 2) kernels  
ls /home/runner/aiter/hsa/gfx950/fmoe/ | grep "ck2stages_gemm2.*FP4" | sed 's/.co$//' | sort
```
REPORT: What tile sizes are available? Specifically for N>128 (d=2048 needs larger tiles).

### 2B. Read how fused_moe selects CK kernel names
```bash
# What function maps (M, N, K, expert) → kernel name?
grep -n "kernelName\|kernel_name\|select_kernel\|get_kernel" /home/runner/aiter/aiter/fused_moe.py | head -20
# Read the selection logic
sed -n '/def.*get_2stage_cfgs/,/^    return/p' /home/runner/aiter/aiter/fused_moe.py | head -80
```

### 2C. Read the Triton SPLITK reduce kernel
```bash
grep -n "reduce\|KSPLIT\|splitk" /home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py | head -30
# How much overhead does the reduce add?
```

### 2D. Can we monkey-patch the config loader to use our configs without writing files?
```bash
cat /home/runner/aiter/aiter/ops/triton/utils/gemm_config_utils.py
```

## After results:
- GEMM write_configs: if passes → BENCHMARK BOTH (KSPLIT=2 and KSPLIT=4). Report per-shape times.
  If K=7168 drops from 13.6μs to <8μs → MASSIVE WIN → leaderboard
  If K=2048 drops from 14.1μs to <8μs → leaderboard
- MoE probe_kernels: READ ALL [MOE] stdout lines — lists available kernel names.
  This is RESEARCH — tells us what kernels exist for d=2048.
- MLA: keep ratcheting

CRITICAL: Do NOT modify files. The GEMM experiments write JSON configs at runtime. Report if "Wrote config" or "FAILED" messages appear.
