# Dead Ends — 53 Documented Approaches That Didn't Work

Every failed approach is documented here with the reason it failed. This is a reference for anyone working on MI355X FP4 optimization.

## GEMM (18 dead ends)

| # | Approach | Result | Why |
|---|----------|--------|-----|
| 1 | hipBLASLt FP4 | 38% relative error | Accumulation order mismatch with eval reference. 14 attempts. |
| 2 | CK ASM `gemm_a4w4` | 19-34us | 3-launch overhead (quant 12us + shuffle 1us + GEMM 3-8us) |
| 3 | `gemm_a16wfp4_preshuffle` | KeyError | Triton `float8_e8m0fnu` type not supported in this aiter version |
| 4 | `gemm_a8wfp4` | Assertion | Eval harness `B_scale_sh` shape doesn't match kernel expectation |
| 5 | Custom Triton `tl.dot_scaled` | 3x slower | Library kernel already optimized for this hardware |
| 6 | Custom HIP MFMA via `load_inline` | 0.93 correlation | Internal K-position permutation unknown. 8 iterations. |
| 7 | KSPLIT values 2,4,12,14 | All worse | Reduce kernel adds ~19us overhead at benchmark (L2 cold) |
| 8 | `num_stages=3` for K=2048 | +34% regression | Too much pipeline pressure for this K dimension |
| 9 | SPLITK_BLOCK_SIZE sweep (512, 2048) | No change | Parameter has no effect on actual timing |
| 10 | GROUP_SIZE_M=2,4 | No change | Tile group scheduling doesn't matter for small M |
| 11 | `matrix_instr_nonkdim=32` | Catastrophic (158us K=2048) | Wrong instruction variant for these shapes |
| 12 | `waves_per_eu=1` | No change | Already at optimal occupancy |
| 13 | `gemm_afp4wfp4` for all shapes | Slower | a16wfp4 fused quant is faster for K!=1536 |
| 14 | L2 cache pre-warming | Adds overhead | Warming wrong addresses doesn't help |
| 15 | XCD remap monkey-patch | Accuracy broken | Remapping changes reduction order |
| 16 | CUDA/HIP graphs | 2x worse | Copy + clone overhead |
| 17 | `deepgemm_ck` | gfx942 only | Only supports MI300X, not MI355X |
| 18 | All env vars (HIP_FORCE_DEV_KERNARG, etc.) | No effect | Already at defaults |

## MLA (12 dead ends)

| # | Approach | Result | Why |
|---|----------|--------|-----|
| 1 | `pg2` for `kv=1024` | ~4% mismatch | Fails secret seed 33% of the time |
| 2 | `qseqlen2` dispatch | GPU memory fault | Kernel not compatible with our metadata |
| 3 | MXFP4 KV cache | Dimension error | `dim=288` not divisible by 512 |
| 4 | `auto_splits` | Fails secret seed | Seed-dependent accuracy |
| 5 | `pg4`, `pg16` | Fail accuracy | Too much approximation |
| 6 | `fast_mode=True` | 5-10% worse | Skips too much computation |
| 7 | `a16w8` for `kv=8192` | 2x slower | bf16 Q = 2x bandwidth vs fp8 |
| 8 | Custom HIP flash-decoding | 561us (14x slower) | Simple kernel can't compete with ASM |
| 9 | Custom Triton flash-decoding | JIT timeout | Compilation exceeds runner time limit |
| 10 | `mla_tuned_v2`, `mla_splits8` | Slower | Sub-optimal split configurations |
| 11 | `kv_granularity` tuning | Zero effect | Only affects metadata, not kernel |
| 12 | `splits=1` for small bs | Underutilizes CUs | 256 CUs need parallelism |

## MoE (23 dead ends)

| # | Approach | Result | Why |
|---|----------|--------|-----|
| 1 | `ksplit=2` | Timeout | Triggers cktile path, JIT recompilation |
| 2 | `block_m=16` | Assertion error | CK kernel doesn't support this tile size |
| 3 | `AITER_USE_OPUS_MOE_SORTING=0` env var | Crash | Must use monkey-patch `fm._USE_OPUS_MOE_SORTING = False` |
| 4 | 1-stage kernel (`fmoe_g1u1`) | 182us (slower) | 2-stage pipeline is more efficient |
| 5 | CK injection for d=2048 (large tiles) | No improvement | Default kernels already optimal for large shapes |
| 6 | Direct `fused_moe_2stages` call | GPU memory fault | C++ wrapper does essential memory management |
| 7 | Direct `ck_moe_stage1_fwd`/`stage2_fwd` | GPU memory fault | Even with fresh allocations |
| 8 | Buffer reuse (`sorted_ids`, `moe_out`) | GPU crash | C++ must allocate fresh each call |
| 9 | `doweight_stage1=True` | Wrong results | Misapplies expert weights |
| 10 | FlyDSL | Zero binaries on runner | `is_flydsl_available()=True` but no .co files |
| 11 | Sort caching | No value | `moe_sorting` called once already |
| 12 | `torch.compile` | Dead | Not compatible with CK pipeline |
| 13 | `dispatch_policy=1` | 80% slower | Wrong dispatch strategy |
| 14 | `ksplit=2` for d=2048 | 2x slower | Triggers slow cktile recompilation |
| 15 | Custom Triton MoE | JIT timeout | Too much compilation for ephemeral pod |
| 16 | `block_m=32` for d=2048 | 410us (worse) | Tile too small for 2048-wide GEMM |
| 17 | `block_m=256` for d=2048 | Crash | CK kernel doesn't support this tile |
| 18 | CSV override with 128x128 tiles for E=33 | GPU crash | Incompatible tile configuration |
| 19 | OPUS off | 175us (worse) | OPUS sorting helps with expert load balancing |
| 20 | Removing E=33 CK injection | 184us (worse) | Default kernels are slower for E=33 d=512 |
| 21 | `use_nt=True` for d=2048 | 185us (worse) | Non-temporal helps for weight-heavy shapes but hurts globally |
| 22 | Quant threshold=8192 (force fused path) | All shapes worse | Fused quant+sort slower for large token counts |
| 23 | HIP quant kernel injection | All shapes worse | 10s `load_inline` compile overhead on ephemeral pods |
