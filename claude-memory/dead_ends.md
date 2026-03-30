---
name: Dead Ends Comprehensive
description: All approaches tried and confirmed not to work — avoid re-trying these
type: feedback
---

DO NOT retry these approaches. They have been tested and confirmed to fail or regress.

**Why:** Each of these consumed rate-limited submissions and time. Re-trying wastes both.
**How to apply:** Before generating any new experiment, check this list. If the approach matches, skip it.

## GEMM Dead Ends
- **tl.dot_scaled compact syntax**: Semicolons in one-line statements cause CompilationError. Use proper multi-line format. Also need shuffle_scales_cdna4() for correct scale format.
- CK gemm_a4w4: 19-34μs (3 kernel launches, separate A quant kills it)
- gemm_a4w4 via gemm_a4w4_asm: still 2x slower due to A quant overhead
- Hybrid Triton+ASM (ASM for K=2048/1536): ASM path 24.5/23μs vs Triton 14.2/16μs
- HIP MFMA kernel: correct but 17.8μs = worse than Triton
- gemm_a16wfp4_preshuffle: Triton e8m0 dtype KeyError
- gemm_a8wfp4: eval framework scale shape assertion bug
- a8wfp4 configs on a16wfp4: 35-110% worse
- Custom scalar HIP: 5-10% quant mismatch
- load_inline: ~90s compile eats timeout
- num_stages=3 globally: K=2048 +34% regression
- Per-shape custom configs: ALL worse than defaults
- Aggressive split-K: 26-70% worse
- CUDA/HIP graphs: 2x worse
- a16wfp4 for K=1536: 25.9μs (vs afp4wfp4 ~16μs)
- a16wfp4 without K7168 config: 32.8μs for K=7168

## MoE Dead Ends
- **FlyDSL env vars** (AITER_USE_FLYDSL_MOE=1 etc): 0 FlyDSL binaries on runner. Env vars have NO effect.
- **CU_NUM=304**: Triggers cktile path (ksplit=2) instead of CK kernels. 155s+ JIT compile, likely timeout. WORSE.
- **CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3**: No visible effect.
- d=2048 large-tile injection (256x128x128x128): no improvement (348 vs 337μs)
- Custom CSV via AITER_CONFIG_FMOE: 188μs (WORSE, overrides good defaults)
- block_m override that breaks E=257: caused +10-70μs regression
- Direct ck_moe_stage1/stage2 calls: GPU memory fault
- Buffer reuse: GPU crash
- 1-stage kernel (fmoe_g1u1): 182μs, slower
- ksplit=2 for d=2048: triggers cktile path, 2x SLOWER
- dispatch_policy=1: 80% slower
- doweight_stage1=True: wrong results
- block_m override via monkey-patch: JIT timeout (old issue)
- CK injection for d=2048 with SMALL tiles (64x32): 17% WORSE
- inject_v2 with `_none_` stage2: 8-21% worse
- sepqsort (threshold=0): no improvement
- NO opus sorting for E=257: 170μs vs 169μs baseline — no difference (Session 10)
- NO tuned CSV configs exist for E=33 cu_num=256 — all shapes use DEFAULT kernel heuristic

## MLA Dead Ends
- pg2 for kv=1024 ALL sizes: FAILS leaderboard (Session 10: 6.1% mismatch on bs=64 kv=1024 secret seed). 4% in test but varies with seed. NEVER USE pg2 for kv<=1024.
- pg2 for kv=8192 (all approaches): 193μs for bs=256 — too many pages
- a16w8 for kv=8192: 306μs for bs=256 — bf16 Q bandwidth
- pg8 for kv=1024: FAILS accuracy
- pg4/pg16: FAIL accuracy
- fast_mode=True: 5-10% worse
- MXFP4 Triton attention: 6-291x slower
- HIP MXFP4 MLA: linker issues
- kv_granularity=max(PAGE_SIZE, 16): WRONG formula, causes accuracy issues
- deepgemm_ck: "Unsupported scales/output dtype" for all fp4 inputs — NOT for fp4 GEMM
- GEMM Triton env vars (BLOCK_PINGPONG, ASYNC_COPY, etc): already default in Triton 3.6 on gfx950
- MoE quant threshold (token_num_quant_moe_sort_switch=8192): 168.8μs = no improvement
- batched_gemm_a16wfp4: only batches SAME weight matrix (different batch, same N,K) — not for different shapes
- MoE PR2261 wider stage2 (S2_256x32x128x128): 175μs = 3.8% WORSE than baseline. d=2048 went 333→348μs.
- GEMM num_stages=3 partial config: KeyError (must pass ALL config fields, not just override)
