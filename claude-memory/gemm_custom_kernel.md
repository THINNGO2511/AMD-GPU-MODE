---
name: GEMM Custom Kernel Working
description: Custom Triton GEMM with tl.dot_scaled validated — 0 mismatch, 43μs (vs aiter 29.8μs). Needs optimization.
type: project
---

## Custom Triton MXFP4 GEMM — VALIDATED WORKING (Session 12)

### Kernel Status (Session 13-14 — BEATING AITER!)
**Config sweep results (wall-clock, fused single-kernel):**
- **M4  N2880 K512**: BN=128 → 19.0μs vs aiter 35.1μs → **1.85x FASTER!**
- **M32 N4096 K512**: BN=64 → 25.3μs vs aiter 29.8μs → **1.18x FASTER**
- **M32 N2880 K512**: BN=64 → 23.0μs vs aiter 28.8μs → **1.25x FASTER**
- All 0 error, 0 mismatch
- BN=128 is optimal for M≤16, BN=64 for M=32
- num_stages=3 is WORSE (+40% for M=4)
- num_warps=8 same as num_warps=4
- Split-K timing: JIT timeout prevents measurement
- Fused kernel (via _mxfp4_quant_op) eliminates quant kernel launch
- **Scale shuffle**: REQUIRED — `shuffle_scales_cdna4()` from Triton tutorial

### What Works
- `tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")` — 6-arg form (accumulator via `+=`)
- Scales must be pre-shuffled on host, then unshuffled in kernel
- B accessed as (K,N) via stride swap: stride_bk=B_q.stride(1), stride_bn=B_q.stride(0)
- `_mxfp4_quant_op` importable from `aiter.ops.triton.quant`

### shuffle_scales_cdna4 (mfma_nonkdim=16)
```python
def _shuffle_scales_cdna4(scales):
    s = scales.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    s = s.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return s.view(sm // 32, sn * 32)
```

### In-kernel unshuffle (mfma_nonkdim=16)
```python
scales = tl.load(ptrs).reshape(
    BM // 32, BK // 32 // 8, 4, 16, 2, 2, 1
).permute(0, 5, 3, 1, 4, 2, 6).reshape(BM, BK // 32)
```

### v3 Status (Session 12)
- Test PASSED (all shapes correct)
- Benchmark PASSED (score pending on leaderboard)
- Includes: XCD swizzle, grouped tiles, fused A quant, per-shape configs
- Only handles K=512 shapes; K=7168/K=2048/K=1536 fall back to aiter
- JIT compilation takes ~60s during warmup, eating time budget

### Optimization TODO
1. Split-K for K=7168 (8-way) and K=2048 (2-way) with scale shuffle
2. Tune BLOCK sizes: try BM=16/32/64, BN=32/64/128/256, BK=256/512
3. num_stages=2 vs 3, waves_per_eu=1 vs 2
4. Write-through store (.wt cache modifier)
5. Minimize Triton JIT time (reduce warmup shapes, cache .so files)
6. Handle ALL shapes in custom kernel (not just K=512)

### Dead Ends
- Compact semicolon syntax in Triton: CompilationError
- Unshuffled scales to tl.dot_scaled: 196K mismatched elements (wrong scale mapping)
- K iteration as cdiv(K, BLOCK_K//2) when passing K_actual: reads beyond data
