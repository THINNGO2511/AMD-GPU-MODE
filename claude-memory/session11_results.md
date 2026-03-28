---
name: Session 11 Results
description: Mar 27 evening — custom Triton GEMM kernel attempts, source probes, MLA pg8_v3
type: project
---

## Session 11 Summary (Mar 27 2026 evening)

### Custom Triton GEMM Kernel (BLOCKED)
- Wrote `submission_custom_triton.py` with fused A-quant + tl.dot_scaled GEMM
- `_mxfp4_quant_op` import FAILS from `aiter.ops.triton._triton_kernels.quant.quant` — path doesn't exist
- Wrote `submission_custom_afp4.py` with external dynamic_mxfp4_quant + custom tl.dot_scaled
- Custom tl.dot_scaled kernel FAILS SILENTLY (test passes via aiter fallback)
- **BLOCKER**: Need correct import path for quant op, AND need to debug tl.dot_scaled data layout
- Targeted probe `exp_quant_path.py` written to search for `_mxfp4_quant_op` in all aiter files

### Source Dump Probes (TRUNCATED)
- All source dumps severely truncated by popcorn output limit (~30 last lines shown)
- GEMM: Found `_USE_GEMM_SPLITK_BF16` flag, `get_splitk()` function (4822 lines omitted)
- MoE: Found `AITER_ONLINE_TUNE` env var, `use_cfg()` bypass logic (716 lines omitted)
- MoE kernel: Full fused_moe source (27329 lines omitted)
- MLA: Reduce kernel internals (3841 lines omitted)
- **Lesson**: Targeted probes with small output are essential, not full source dumps

### MLA pg8_v3 (LEADERBOARD SUBMITTED)
- Uses num_kv_splits=16 for ALL shapes (was heuristic in v2)
- Test PASSED: 3.15% mismatch on kv=8192 bs=256 (safe)
- Leaderboard SUBMITTED: per-shape 33.9-95.4μs best times
- bs=32 kv=1024: 34.8μs, bs=64 kv=8192: 45.4μs, bs=256 kv=8192: 95.4μs
- Estimated geomean ~41μs (need to see leaderboard ranking)

### POPCORN Header Discovery
- `#!POPCORN benchmark <name>` is INVALID directive — must use `#!POPCORN leaderboard <name>`
- The `--mode benchmark/test/leaderboard` flag controls the mode, NOT the header
- Wasted 5 submissions on this error (may have consumed rate limits)

### Rate Limit Usage (Session 11)
- GEMM: ~5 submissions (3 test + 2 benchmark including probes)
- MoE: ~4 submissions (2 benchmark probes)
- MLA: ~4 submissions (1 test + 2 benchmark + 1 pending)
- Failed header submissions may have counted against limits

### SESSION 12: tl.dot_scaled WORKS! Custom GEMM kernel validated!
- Custom Triton GEMM with tl.dot_scaled: PERFECT ACCURACY (0 mismatch)
- Scale shuffle (shuffle_scales_cdna4) is REQUIRED and WORKING
- Current speed: 43μs vs aiter 29.8μs = 0.69x (31% slower, needs optimization)
- Key: unshuffle inside kernel, pre-shuffle on host
- FlyDSL MoE: DEAD END (0 binaries on runner)
- MLA pg8_v3: leaderboard submitted and passed

### tl.dot_scaled CompilationError (LIKELY SYNTAX ISSUE, NOT DEAD END)
- Our minimal test kernel had CompilationError — but used compact semicolon syntax
- aiter's gemm_a16wfp4 DOES use tl.dot_scaled and works fine
- **ROOT CAUSE**: likely semicolons in one-line statements OR wrong scale format
- Triton tutorial shows `shuffle_scales_cdna4()` is REQUIRED for CDNA4
- Raw E8M0 scales won't work — must shuffle for MFMA instruction format
- **Next step**: Rewrite test kernel with proper multi-line syntax + shuffled scales

### Competitor Research BREAKTHROUGH Findings
- **Triton tutorial**: triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
  - Has complete `block_scaled_matmul_kernel_cdna4` with `shuffle_scales_cdna4()`
  - Scale shuffle needed: reorganizes E8M0 for MFMA nonkdim=16 or 32
  - BLOCK_K=256 for FP4, B transposed (KxN)
- **FlyDSL MoE env vars** (untested!):
  - AITER_USE_FLYDSL_MOE=1, AITER_USE_FLYDSL_MOE_STAGE1=1, AITER_USE_FLYDSL_MOE_STAGE2=1
  - Kimi K2.5 blog: FlyDSL delivers significant speedups over CK
- **ZainHaider20**: 1μs GEMM (101 subs) — likely benchmark exploit (caching/precompute)
- **Danishlynx**: 27μs MLA (490 subs) — leads MLA
- **CU_NUM**: MI355X has 304 CUs, but CSV uses 256 — might affect MoE config matching
- **CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3**: untested env var for MoE

### New Dead Ends
- `tl.dot_scaled` in custom Triton kernels: CompilationError on runner
- Custom tl.dot_scaled kernel with external quant: same error
- Full source dump probes via popcorn: truncated by output limit
- `_mxfp4_quant_op` import works but useless without tl.dot_scaled
