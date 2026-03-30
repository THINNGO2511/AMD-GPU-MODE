---
name: GEMM Technical Findings
description: All GEMM kernel APIs, configs, and performance data discovered through Session 9
type: project
---

## Best Path: gemm_a16wfp4 (Triton, fuses A quant)
- ~10μs benchmark, ~16μs ranked (L2 cache clearing adds ~6μs)
- K=7168: custom config (BM=8 BN=64 BK=512 KS=8) — only config that beats defaults
- K=1536: use gemm_afp4wfp4 (separate quant + GEMM) — a16wfp4 is 25.9μs vs ~16μs
- All other K: default config is optimal

## ASM Kernel Inventory (Session 9 Probe)
- 35 pre-compiled .co files at `/home/runner/aiter/hsa/gfx950/f4gemm/`
- Format: `f4gemm_bf16_per1x32Fp4_BpreShuffle_{tile_M}x{tile_N}.co`
- Available tile_M: 32, 64, 96, 128, 160, 192, 224, 256
- Available tile_N: 128, 256, 384, 512, 640, 768, 896, 1024
- CSV config at same dir: columns = tile_M, tile_N, splitK, bpreshuffle, knl_name, co_name

## API Inventory
```
gemm_a16wfp4(A, w, w_scales, dtype, y=None, config=None)     # Triton, fused A quant — FASTEST
gemm_afp4wfp4(A_fp4, w, A_scale, w_scales, dtype)            # Triton, pre-quantized A
gemm_a4w4(A_fp4, B_shuf, A_scale_sh, B_scale_sh, dtype, bpreshuffle=True)  # CK/ASM dispatch
gemm_a4w4_asm(A, B, A_scale, B_scale, out, kernelName, bias, alpha, beta, bpreshuffle, log2_k_split)
deepgemm(XQ, WQ, Y, group_layout, x_scale, w_scale)          # wraps deepgemm_ck
get_padded_m(M, N, K, gl) -> int                              # M padding for ASM kernels
get_GEMM_config(M, N, K) -> dict or None                      # tuned ASM config lookup
```

## Tuned ASM Configs (only 2 of 6 benchmark shapes have configs)
- (64, 7168, 2048) → kernel `_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E`, splitK=0, 6.8μs claimed
- (256, 3072, 1536) → kernel `_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E`, splitK=0, 6.2μs claimed
- Other 4 shapes: no tuned config (returns None)

## Why ASM is Slower in Practice
- gemm_a4w4 adds ~10μs overhead from separate dynamic_mxfp4_quant + e8m0_shuffle
- Even with direct gemm_a4w4_asm call, A quant overhead makes it 2x slower than gemm_a16wfp4
- The 6.8μs claimed time likely measures ONLY the ASM kernel, not including A quant

## Session 10 New Findings (GEMM Probe)
- deepgemm(XQ, WQ, Y, group_layout: Tensor, x_scale=None, w_scale=None) — group_layout MUST be a Tensor
- deepgemm_ck is a C++ torch op at torch.ops.aiter.deepgemm_ck
- Passing int for group_layout fails: "Expected a value of type 'Tensor'"
- Need to discover correct tensor format (probably group layout for blockscale grouping)

### NEW APIs Discovered
- `batched_gemm_a16wfp4.py` — batched GEMM (could process multiple shapes in one call!)
- `batched_gemm_afp4wfp4.py` — batched fp4 GEMM
- `batched_gemm_afp4wfp4_pre_quant.py` — batched with pre-quant
- `fused_gemm_afp4wfp4_a16w16.py` — fused fp4+bf16 GEMM
- `fused_gemm_afp4wfp4_mul_add.py` — fused with multiply+add
- `fused_gemm_afp4wfp4_split_cat.py` — fused with split/concat
- `gemm_afp4wfp4_pre_quant_atomic.py` — atomic pre-quant
- Batched GEMM configs exist: `gfx950-BATCHED_GEMM-AFP4WFP4-N=128-K=512.json`

### Total .co File Inventory
1314 total: 35 f4gemm, 24 bf16gemm, 6 fp8gemm_blockscale, 182 fmoe_2stages, 27 mla, 41 pa, 419 gelu, 421 silu

## Remaining Unexplored Leads
- deepgemm with correct tensor group_layout — need to probe format
- batched_gemm_a16wfp4 — could batch multiple shapes
- fused_gemm_afp4wfp4_a16w16 — could be faster fused path
- gemm_a8wfp4 — fp8 A + fp4 B (previously blocked by eval assertion)
- Custom Triton persistent kernel combining all 6 shapes
