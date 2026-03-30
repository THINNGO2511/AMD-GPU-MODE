---
name: Session 13 Results
description: Mar 28 — GEMM custom kernel BEATS aiter 1.29-1.56x, split-K leaderboard submitted
type: project
---

## Session 13 — HISTORIC BREAKTHROUGH

### Custom Triton GEMM Kernel Performance
| Shape | Custom | aiter | Speedup |
|-------|--------|-------|---------|
| M=4  N=2880 K=512 | 32.5μs | 32.3μs | 0.99x (tied) |
| M=32 N=4096 K=512 | 22.9μs | 29.4μs | **1.29x FASTER** |
| M=32 N=2880 K=512 | 18.8μs | 29.4μs | **1.56x FASTER** |
| M=16 N=2112 K=7168 | split-K | - | test passed |
| M=64 N=7168 K=2048 | split-K | - | test passed |
| M=256 N=3072 K=1536 | aiter fallback | - | proven |

### Leaderboard Submission
- **submission_splitk_v1.py**: ALL 3 CHECKS PASSED (test+benchmark+leaderboard)
- Handles all 6 shapes: custom for K=512, split-K for K=7168/2048, aiter for K=1536
- KSPLIT=7 for K=7168 (7168/7=1024, clean division by BK=256)
- KSPLIT=2 for K=2048 (2048/2=1024)
- Config: BM=32, BN=64, BK=256, num_warps=4, num_stages=2, mfma_nonkdim=16

### Key Technical Fixes
- M<32 padding: A_scale and A_fp4 must be padded to multiple of 32 rows for shuffle
- Scale shuffle output is (M//32, K) — not (M, K//32)
- Print to STDOUT (not stderr) for timing to survive popcorn truncation
- JIT compilation takes ~50-60s, eating into benchmark time budget

### Dead Ends Confirmed
- FlyDSL MoE: 0 binaries on runner
- CU_NUM=304: triggers slow cktile path
- pg2_fix: seed-dependent gamble
