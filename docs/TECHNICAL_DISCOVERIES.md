# Technical Discoveries

Key technical findings from 22+ sessions of MI355X kernel optimization.

## 1. MFMA FP4 Intrinsic on gfx950

The `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` instruction works on gfx950 via `load_inline`:

```cpp
typedef int operand_t __attribute__((ext_vector_type(8)));
typedef float c_result_t __attribute__((ext_vector_type(4)));

c_acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    a_reg, b_reg, c_acc, 4, 4, 0, sa, 0, sb);
// cbsz=4 (FP4 for A), blgp=4 (FP4 for B)
```

**Output mapping** (verified by probe):
- `col = lane % 16`
- `row = (lane / 16) * 4 + reg_idx` (for `reg_idx` 0-3)
- 64 threads x 4 regs = 256 values = full 16x16 tile

**Unsolved**: The internal K-position-to-thread-register mapping has an unknown permutation. Data loaded in row-major order achieves 0.93 correlation with the Triton reference — close but not matching.

## 2. GEMM is 91% Memory-Bound

Measured on MI355X with `torch.cuda.Event`:
- **Kernel compute**: ~0.56us (warm L2, 50 repetitions)
- **Benchmark timing**: ~6.18us (with L2 clearing)
- **L2 cache contribution**: 91% of benchmark time

No kernel optimization (custom or Triton config) can fix this. The leaderboard measures with L2 clearing, making it a memory access pattern problem, not a compute problem.

## 3. hipBLASLt FP4 Accumulation Order

hipBLASLt produces numerically different results than Triton for FP4 GEMM — not due to precision, but **accumulation order**. The K-dimension reduction sums in a different order, producing up to 38% relative error. After 14 attempts with various configurations, this was confirmed as unfixable.

## 4. E8M0 Scale Computation

aiter's MXFP4 quantization uses a specific bit-manipulation for E8M0 scales:

```python
# Pseudocode matching aiter's _mxfp4_quant_op
amax_int = reinterpret_float_as_uint32(amax)
rounded = (amax_int + 0x200000) & 0xFF800000  # round up to next fp32 quantum
e8m0 = (rounded >> 23) - 2  # extract exponent, subtract 2
```

This is NOT equivalent to `ceil(log2(amax/6)) + 127` — the bit manipulation handles edge cases differently. Our custom HIP kernel only achieved 100% scale match after implementing this exact formula.

## 5. E8M0 Shuffle Permutation

The `e8m0_shuffle` function rearranges scale values for CK ASM kernel consumption:

```python
# Input: [M, K/32] uint8
# Pads to multiples of (256, 8)
scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
# Output: [padded_M, K/32] with internal 6D permutation
```

The inverse (unshuffle) is:
```python
s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
```

## 6. MoE `use_nt` Impact

`fm.use_nt = lambda token, topk, expert: False` hurts E=257 shapes by ~11us because:
- E=257 with d=256: weight matrices are small (~256 FP4 elements per expert)
- These fit in L2 cache and benefit from temporal caching
- Non-temporal loads bypass L2, causing unnecessary HBM fetches

The library's default `use_nt` function is already optimal — it enables NT loads only for shapes that don't fit in L2.

## 7. Ephemeral Pod Constraints

Every submission starts on a fresh pod:
- Triton JIT cache is destroyed — all kernels recompile
- `load_inline` HIP compilation takes ~10-30s depending on complexity
- No pip install — must use pre-installed libraries
- The word "stream" is grep-filtered from submissions (causes build failure)
- Use `hipLaunchKernelGGL(kernel, grid, block, 0, 0, args)` — the `0` replaces what would normally be a launch configuration parameter

## 8. `deepgemm_ck` API

Exists in aiter at `/home/runner/aiter/aiter/ops/deepgemm.py` but only supports gfx942 (MI300X):

```python
deepgemm_ck(XQ, WQ, Y, group_layout, x_scale, w_scale)
# Grouped FP8 GEMM with variable M per group
# Test: if get_gfx() not in ["gfx942"]: return
```

## 9. MoE Quantization Pipeline

`fused_dynamic_mxfp4_quant_moe_sort` is a Triton kernel (not C++) at:
`/home/runner/aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py`

It's called when `token_num <= 1024`. For `token_num > 1024`, a separate `dynamic_mxfp4_quant` + `moe_mxfp4_sort` is used. Both paths are monkey-patchable from Python.

## 10. CK Kernel Binary Names

The CK kernel names encode tile configuration:
```
moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16
│                   │              │    │                   │  │         │      │                │
│                   tile_M×N×K×K2  │    │                   │  │         │      │                output_dtype
│                                  warps│                   │  │         │      activation
│                                       variant             │  quant_type      weight_mode
│                                                           swizzle
stage (gemm1=stage1, gemm2=stage2)
```
