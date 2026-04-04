# MXFP4 GEMM — Matrix Multiplication

**Best score:** 9.72us benchmark, 15.6us ranked (#160)
**Starting point:** 16.5us
**Improvement:** 41% (benchmark), 5% (ranked — L2 cache gap)

## Problem

Quantize bf16 `A` to MXFP4, then GEMM with pre-quantized MXFP4 `B` to produce bf16 `C`. Tolerance: `rtol=1e-2, atol=1e-2`.

Benchmark shapes: 6 shapes varying M=4-256, N=2112-7168, K=512-7168 (geometric mean).

## Approach

### Kernel Selection
- K=7168, K=2048: `gemm_a16wfp4` (fused bf16-to-FP4 quant + GEMM in one Triton kernel)
- K=1536: `gemm_afp4wfp4` (separate quant + FP4xFP4 GEMM)
- K=512: `gemm_a16wfp4` with default configs

### Per-Shape Config Tuning
Each shape has a tuned Triton config:

| Shape | BM | BN | BK | KSPLIT | stages | waves | cache |
|-------|----|----|-----|--------|--------|-------|-------|
| K=7168 | 16 | 64 | 512 | 8 | 2 | 2 | .cg |
| K=2048 | 16 | 128 | 512 | 1 | 2 | 4 | .cg |
| K=512 | 4 | 128 | 512 | 1 | 1 | 2 | .cg |

### Optimizations
- **Fast E8M0 unshuffle**: `torch.take` with precomputed gather indices
- **Full JIT prewarm**: All 6 shapes compiled before first benchmark
- **Output tensor caching**: Pre-allocated per-shape output buffers

### Custom HIP MFMA Kernel (Experimental)

Built a custom FP4 GEMM kernel using the `v_mfma_scale_f32_16x16x128_f8f6f4` MFMA intrinsic via `load_inline`:

- **Phase 1**: Single MFMA tile — confirmed instruction works on gfx950, mapped output registers
- **Phase 2**: Added K-loop + M/N tiling — works on all benchmark shapes
- **Blocker**: 0.93 correlation with reference — internal K-position permutation is unknown

See [experiments/hip-kernels/](../../experiments/hip-kernels/) for all 8 iterations.

## Key Discovery: Memory-Bound Bottleneck

Kernel compute time: **~0.56us** (warm cache, 50 reps). Benchmark time: **~6.18us**. The 91% gap is L2 cache clearing between iterations. No kernel optimization can fix this — the bottleneck is memory, not compute.

## Files

| File | Description |
|------|-------------|
| `submission_prewarm.py` | Best submission — per-K kernel selection, full JIT prewarm, tuned configs. Leaderboard version. |
| `sub_ultimate_v1.py` | Alternate approach — different config tuning strategy, used for benchmarking |

## Dead Ends (18 total)

- hipBLASLt FP4 — accumulation order mismatch, 38% relative error (14 attempts)
- CK ASM (`gemm_a4w4`) — 3-launch overhead (quant + shuffle + GEMM) negates kernel speed
- All Triton config sweep variations — 9 tested, 0 improvements over current optimum
- Custom HIP MFMA kernel — 8 iterations, 0.93 correlation ceiling
- See [docs/DEAD_ENDS.md](../../docs/DEAD_ENDS.md) for full list
