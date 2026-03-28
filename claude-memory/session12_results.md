---
name: Session 12 Results
description: Mar 27 late — BREAKTHROUGH custom Triton GEMM works, MLA leaderboard, FlyDSL dead end
type: project
---

## Session 12 Summary (Mar 27 2026, late evening)

### BREAKTHROUGH: Custom Triton MXFP4 GEMM Kernel VALIDATED
- **v2**: 0/131072 mismatches, PERFECT accuracy on M=32 N=4096 K=512
- **Speed**: 43μs custom vs 29.8μs aiter = 0.69x (needs optimization)
- **v3**: XCD swizzle + fused A quant + grouped tiles — test PASSED, benchmark PASSED
- **Key discoveries**:
  - `tl.dot_scaled` works with proper multi-line syntax (semicolons cause CompilationError)
  - `shuffle_scales_cdna4()` is REQUIRED for CDNA4 MFMA — raw scales give 196K mismatches
  - B accessed as (K,N) via stride swap: stride_bk=B_q.stride(1), stride_bn=B_q.stride(0)
  - `_mxfp4_quant_op` importable from `aiter.ops.triton.quant` for fused A quant
  - Triton tutorial at triton-lang.org has complete CDNA4 kernel template

### MLA pg8_v3 Leaderboard Submitted
- All 6 runs passed (test + secret + benchmark + leaderboard)
- splits=16 for all shapes, pg1+bf16Q for kv≤1024, pg8+fp8Q for kv≥8192
- Estimated ~41μs (was 45.5μs = ~10% improvement)

### Confirmed Dead Ends
- FlyDSL MoE: 0 binaries on runner, env vars have NO effect
- CU_NUM=304: triggers slow cktile path with 155s JIT, WORSE
- CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3: no visible effect
- pg2_fix: seed-dependent gamble (4-6% mismatch inherent, ASM kernel has kPageSize=1)

### Submissions (Session 12)
- GEMM: ~8 submissions (v1 test, v2 test x2, v2 bench, diag bench, v3 test, v3 bench, sig probe)
- MoE: 2 submissions (FlyDSL test, CK_TILE bench)
- MLA: 0 new (pg8_v3 was Session 11)
