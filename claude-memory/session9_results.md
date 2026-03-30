---
name: Session 9 Results
description: Mar 27 2026 session — MLA kv_splits improvement, GEMM ASM probe, MoE dead ends
type: project
---

## Session 9 Summary (Mar 27 2026)

### Infrastructure
- Updated popcorn-cli v0.1.0 → v1.3.6 (has --no-tui for non-interactive submission)
- Re-registered via Discord OAuth as noobmaster69_og
- popcorn-cli path: `~/.local/bin/popcorn-cli`
- Submission command: `popcorn-cli submit --gpu MI355X --leaderboard <name> --mode <mode> <file> --no-tui`

### MLA: 5% Improvement (42.5μs → ~40.4μs)
- **num_kv_splits sweep** found 16 optimal for most shapes, 8 for bs≤32+kv=1024
- Submitted `exp_optimal_splits.py` — rate limited, needs retry
- pg2 for kv=8192: TERRIBLE (193μs for bs=256) — too many pages
- a16w8 for all: TERRIBLE for kv=8192 (306μs) — bf16 Q bandwidth
- Hybrid a16w8+pg2/a8w8+pg8 with optimal splits is best approach

### GEMM: ASM Kernels Probed (No Improvement Yet)
- `deepgemm(XQ, WQ, Y, group_layout, x_scale, w_scale)` exists, wraps `deepgemm_ck`
- `gemm_a4w4_asm(A, B, A_scale, B_scale, out, kernelName, ...)` — direct ASM dispatch
- 35 pre-compiled .co files: `f4gemm_bf16_per1x32Fp4_BpreShuffle_{tile_M}x{tile_N}`
- CSV format: tile_M, tile_N, splitK, bpreshuffle, knl_name, co_name
- `get_padded_m(M,N,K,0)`: M=4→16, M=16→16, M=32→32, M=64→64, M=256→256
- `get_GEMM_config` found tuned configs for only 2/6 shapes
- **gemm_a4w4 (CK/ASM path) is 2x SLOWER than Triton** due to separate A quant + dispatch overhead
- **Hybrid Triton+ASM**: ASM for K=2048/1536 was WORSE (24.5/23μs vs 14.2/16μs)
- Triton gemm_a16wfp4 remains fastest path

### MoE: All Injection Attempts Failed
- d=2048 large-tile injection (256x128x128x128): no improvement (348μs vs 337μs)
- Custom CSV via AITER_CONFIG_FMOE: WORSE (188μs vs 167μs) — overrides good defaults
- block_m override for E=257: caused regressions (+10-70μs)
- `submission_opus_sort.py` remains the best at 167-169μs

### Experiments Run (15 submissions)
- 10 test mode, 7 benchmark mode across all 3 problems
- 1 leaderboard (rate limited before completion)
