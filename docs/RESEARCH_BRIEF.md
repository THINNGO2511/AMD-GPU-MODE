# AMD x GPU MODE Hackathon — Complete Research Brief

**Competitor**: noobmaster69_og | **Deadline**: April 6, 2026 11:59 PM PST | **Target GPU**: AMD Instinct MI355X (gfx950, CDNA4)

## Current Scores (March 31, 2026)
| Problem | Our Score | Our Rank | Top 10 Cutoff | Gap |
|---------|-----------|----------|---------------|-----|
| MXFP4 GEMM | 15.7μs | ~#160/320 | 8.2μs | 2x |
| MLA Decode | 42.16μs | #17/185 | 34.9μs | 17% |
| MXFP4 MoE | 169μs | #64/225 | 129μs | 24% |

## Leaderboard Intel
| Rank | User | GEMM | MLA | MoE | Notes |
|------|------|------|-----|-----|-------|
| 1 | josusanmartin | 7.65μs | 31.16μs | 127μs | 3861 submissions, automated sweeping |
| 1 | Ananda Sai A | 8.1μs | 33μs | 110μs | Top 5 all 3, 293 subs |
| 1 | Maxwell Cipher | — | — | 107μs | MoE specialist |
| exploit | HorizonLiang | 4.36μs | — | — | tinygrad comgr timing artifact |
| exploit | Borui Xu | — | 12.7μs | — | Returned zeros for kv=8192 (self-reported) |
| ~10 | threshold | 8.2μs | 34.9μs | 129μs | Top 10 qualification |

---

## Problem 1: MXFP4 GEMM (leaderboard: amd-mxfp4-mm)

### Task
- Input: A[M,K] bf16, B[N,K] bf16, B_q[N,K/2] fp4x2, B_shuffle[N,K/2] fp4x2 (tile-shuffled), B_scale_sh E8M0 (shuffled)
- Flow: Quantize bf16 A → MXFP4, then GEMM with pre-quantized fp4 B → bf16 C
- Tolerance: rtol=1e-2, atol=1e-2
- Benchmark shapes: (M=4,N=2880,K=512), (M=16,N=2112,K=7168), (M=32,N=4096,K=512), (M=32,N=2880,K=512), (M=64,N=7168,K=2048), (M=256,N=3072,K=1536)
- Ranking: geometric mean of benchmark latencies

### Current Best Approach (15.7μs ranked)
- `gemm_a16wfp4` for K=512/2048/7168 (fused bf16→fp4 quant inside Triton kernel)
- `dynamic_mxfp4_quant` + `gemm_afp4wfp4` for K=1536 (separate quant + GEMM)
- `torch.take` fast unshuffle of E8M0 scales (eliminates permute+contiguous overhead)
- `.cg` cache modifier on all configs
- BM=16 for K=7168 (sweeper-found, 3% better than BM=8)
- Output tensor caching, pre-warmup for all 6 shapes
- `OPTIMIZE_EPILOGUE=1` env var
- `HIP_FORCE_DEV_KERNARG=1` env var

### Key Configs
```python
K7168: BM=16, BN=64, BK=512, GSM=1, warps=4, stages=2, wpe=2, .cg, KSPLIT=8, SPLITK_BLOCK=1024
K512:  BM=4, BN=128, BK=512, GSM=1, warps=4, stages=1, wpe=2, .cg, KSPLIT=1
K2048: BM=16, BN=128, BK=512, GSM=1, warps=8, stages=2, wpe=4, .cg, KSPLIT=1
K1536: separate quant + afp4wfp4 (library defaults)
```

### Benchmark vs Ranked Gap
- Benchmark geomean: 9.6μs
- Ranked geomean: 15.7μs
- Gap: ~6μs per shape from L2 cold cache (eval.py allocates 64GB tensor to flush L2 between calls)
- This gap is the #1 bottleneck — no code-level fix found

### Available Kernel Paths on Runner
| Kernel | A input | B input | Launches | Timing | Status |
|--------|---------|---------|----------|--------|--------|
| `gemm_a16wfp4` | bf16 (fused fp4 quant) | fp4 raw uint8 | 1 | 6-14μs | **BEST** |
| `gemm_afp4wfp4` | fp4 pre-quant uint8 | fp4 raw uint8 | 2 | 16μs | K=1536 only |
| `gemm_a16wfp4_preshuffle` | bf16 (fused) | fp4 shuffled (reshaped) | 1 | Runs but WRONG results | Dead: shuffle format mismatch |
| `gemm_a4w4` | fp4 pre-quant | fp4 shuffled | 3 | 19-34μs | Dead: quant=44μs, API changed |
| `gemm_a8wfp4` | fp8 per-token | fp4 raw uint8 | 2 | 117μs, 94% mismatch | Dead: wrong precision + slow |
| Custom Triton (tl.dot_scaled) | bf16 (fused) | fp4 raw uint8 | 1 | 19-25μs K=512 | Dead: 3x slower than library |
| Custom HIP (MFMA intrinsic) | bf16 (manual quant) | fp4 raw | 1 | 13% accuracy error | Dead: data layout wrong |

### CK ASM Kernels Available
35 pre-compiled `.co` files: `f4gemm_bf16_per1x32Fp4_BpreShuffle_{tile_M}x{tile_N}.co`
- Tiles: 32×{128-1024}, 64×{128-1024}, 96×{128-640}, 128×{128-512}, 160×{128-384}, 192×{128-256}, 224×{128-256}, 256×{128-256}
- CSV: `f4gemm_bf16_per1x32Fp4.csv` with tile_M, tile_N, splitK, bpreshuffle, kernelName columns
- `gemm_a4w4` no longer accepts `kernelName` parameter — API changed to `(A, B, A_scale, B_scale, bias, dtype, alpha, beta, bpreshuffle)`

### Triton Config Files on Runner (gfx950, our shapes)
- `gfx950-GEMM-A16WFP4.json` — default configs
- `gfx950-GEMM-A16WFP4-N=7168-K=2048.json` — BM=8, BN=128, BK=512, warps=8, stages=2
- `gfx950-GEMM-A16WFP4-N=512-K=7168.json` — NOT our N=2112
- `gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json` — BM=8, BN=128, BK=512, wpe=1, stages=1
- `gfx950-GEMM-AFP4WFP4-N=2112-K=7168.json` — BM=16, BN=32, BK=256
- No shape-specific A16WFP4 configs for N=2880 K=512 or N=4096 K=512

### Dead Ends (GEMM)
1. CK ASM gemm_a4w4: 19-34μs, 3 kernel launches, quant alone is 44μs
2. gemm_a16wfp4_preshuffle: eval harness B_shuffle format ≠ aiter internal preshuffle format
3. gemm_a8wfp4: 94% mismatch, 117μs, fp8 per-token quant doesn't match fp4 block-scale reference
4. Custom Triton tl.dot_scaled: 3x slower than library for K=512
5. Custom HIP MFMA: 13% accuracy error from data layout, never resolved
6. KSPLIT=14 for K=7168: 24% SLOWER (more reduction overhead)
7. KSPLIT=4 for K=2048: 39% SLOWER
8. XCD remap monkey-patch: accuracy broken (sign-flipped values)
9. L2 prefetch HIP kernel: +5μs overhead (launch cost > L2 miss savings)
10. JSON-derived configs (wpe=1, stages=1): 6-7% SLOWER
11. num_stages=3: WORSE for K=2048 (+34%), marginal for K=512
12. CUDA/HIP graphs: 2x WORSE (copy + clone overhead)
13. Hand-tuned configs for K=512/2048: WORSE than library defaults
14. OPTIMIZE_EPILOGUE env var: marginal, doesn't help split-K shapes
15. 200+ config sweep: no improvement beyond 9.74μs baseline

---

## Problem 2: MLA Decode (leaderboard: amd-mixed-mla)

### Task
- DeepSeek R1 MLA: 16 query heads, 1 KV head, qk_dim=576, v_dim=512
- Input: q (total_q, 16, 576) bf16, kv_data dict with bf16/fp8/mxfp4 formats, indptr tensors
- KV cache: bf16 (total_kv, 1, 576), fp8 (total_kv, 1, 576) + scalar scale, mxfp4 (total_kv, 1, 288) fp4x2 + (total_kv, 24) e8m0
- Tolerance: rtol=0.1, atol=0.1 + 5% mismatch bypass
- Benchmark: bs=4/32/64/256 × kv=1024/8192 (8 shapes)
- Output: (total_q, 16, 512) bf16

### Current Best Approach (42.16μs ranked)
- `sub_pg2_bf16q.py` — pg2 + bf16 Q for kv≤1024, pg8 + fp8 Q for kv≥8192
- Uses fp8 KV data (kv_data["fp8"])
- Fused Triton Q fp8 quantization (2 kernels: amax + cast)
- Metadata caching, output tensor caching
- Pre-compiled ASM kernels: `mla_a16w8` for kv≤1024, `mla_a8w8` for kv≥8192
- 67% leaderboard pass rate (pg2 has ~4% mismatch that varies by secret seed)

### ASM Kernels Available (28 .co files)
- `mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co` — fp8 Q + fp8 KV (used for kv≥8192)
- `mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps.co` — bf16 Q + fp8 KV (used for kv≤1024)
- `mla_a8w8_qh16_qseqlen2_gqaratio16_ps.co` — EXISTS but dispatch code not on runner (needs PR #2440)
- Various a16w16 variants — bf16 Q + bf16 KV
- NO fp4 KV kernels exist

### Page Size Analysis
| pg | kv=1024 accuracy | kv=8192 accuracy | kv=1024 speed | kv=8192 speed |
|----|-------------------|-------------------|---------------|---------------|
| pg1 | 0% mismatch ✅ | 0% mismatch ✅ | 30-89μs (slow for bs=256) | Fast |
| pg2 | **~4% mismatch** (risky) | ~3% mismatch | **32-52μs** | Fast |
| pg4 | FAILS (7673 mismatches) | FAILS | — | — |
| pg8 | N/A | ~3% mismatch | N/A | **Fastest** |
| pg16 | N/A | FAILS (71596) | N/A | — |

### qseqlen2 Dispatch
- `max_seqlen_hint=2` tells metadata to dispatch to qseqlen2 ASM kernel
- Tested for bs=256 kv=8192: 96μs benchmark (was 189μs) — **49% speedup**
- BUT: fails accuracy on secret leaderboard seeds for bs=256 kv=8192
- Only validated on specific test seeds, not reliable for leaderboard

### Dead Ends (MLA)
1. KV stride-2 subsample: 71596 mismatches on kv=8192 — too many tokens skipped
2. pg2 for kv=1024 on leaderboard: ~33% failure rate on secret seeds
3. qseqlen2 for bs=256 kv=8192 on leaderboard: fails on secret seeds
4. bf16 KV (a16w16) for kv≤1024: same ~4% mismatch as fp8 — pg2 mismatch is from page boundaries, NOT KV format
5. pg4 for kv=1024: 7673 mismatches
6. pg16 for kv=8192: 71596 mismatches
7. pg8 for kv=1024: not tested (probably worse accuracy)
8. num_kv_splits tuning: <1% difference
9. kv_granularity tuning: zero effect on accuracy
10. fast_mode=True: 5-10% WORSE
11. Custom HIP flash-decoding: 561μs (14x slower than aiter)
12. Custom Triton flash-decoding: JIT timeout on ephemeral runner
13. MXFP4 KV: no ASM kernel exists, 1.9x bandwidth savings unusable
14. pg1 + qseqlen2: 44μs benchmark, worse than pg2's 42μs
15. a16w8 for kv=8192: 2x slower (bf16 Q = 2x bandwidth)

---

## Problem 3: MXFP4 MoE (leaderboard: amd-moe-mxfp4)

### Task
- DeepSeek-R1 fused MoE: 256 routed + 1 shared expert, top-8+1=9 active, SwiGLU
- 2-stage: Stage1 (gate+up MXFP4 GEMM + SiLU), Stage2 (down MXFP4 GEMM + weighted sum)
- Tolerance: rtol=2e-2, atol=2e-2
- Benchmark: E=257 d=256 bs=16/128/512, E=33 d=512 bs=16/128/512, E=33 d=2048 bs=512

### Current Best Approach (169μs ranked)
- `submission_optimized_v2.py`
- `use_nt=False` for all shapes
- CK kernel injection for E≤64 d<2048 (STAGE1_64/256 + STAGE2_V1)
- block_m tuning: 32 for est_m<50, 64 for est_m≥50, 128 for d≥2048 est_m≥100
- Uses aiter fused_moe (C++ torch op essential — direct calls crash)

### Profiling Breakdown (bs=16, E=257, d=256)
- Stage1 CK GEMM: 41% (biggest target)
- fused_dynamic_mxfp4_quant_moe_sort: 28% (called 2x)
- Stage2 CK GEMM: 23%
- moe_sorting: 8%

### CK Kernel Names (proven working)
```
Stage1 small: moe_ck2stages_gemm1_64x32x32x128_1x1_..._silu_FP4X2_FP4X2_B16
Stage1 large: moe_ck2stages_gemm1_256x32x128x128_1x4_..._silu_FP4X2_FP4X2_B16
Stage2 v1:    moe_ck2stages_gemm2_64x32x32x128_1x1_..._v1_..._FP4X2_FP4X2_B16
Stage2 v3:    moe_ck2stages_gemm2_256x32x128x128_1x4_..._v3_..._FP4X2_FP4X2_B16 (exists but same speed)
```

### Competitor Techniques (from research)
| Competitor | Score | Technique |
|-----------|-------|-----------|
| Ryan Mathieu | ~151μs | ksplit=2 for E≥257, block_m=16 for small batches, sepqsort |
| Bortlesboat | ~150.9μs | ksplit=2 for E≥257, sepqsort |
| "noopus" | ~144μs | Disable opus sorting, CK injection for E=33 |

### Dead Ends (MoE)
1. ksplit=2 for E≥257: triggers cktile path → TIMEOUT on ephemeral runner
2. ksplit=2 MOEMetadata surgery: same timeout
3. ksplit=2 via get_ksplit override: doesn't propagate to C++ layer
4. block_m=16 for E≥257: internal error / assertion failure on test shapes
5. AITER_USE_OPUS_MOE_SORTING=0: internal error crash
6. sepqsort (token_num_quant_moe_sort_switch=0): adds overhead, 179μs (WORSE)
7. 1-stage kernel: 182μs (WORSE than 2-stage 169μs)
8. Split shared expert: FAILS accuracy
9. ksplit=2 for d=2048: triggers slow cktile path, 2x SLOWER
10. dispatch_policy=1: 80% slower
11. doweight_stage1=True: wrong results
12. Direct fused_moe_2stages call: GPU memory fault
13. Direct ck_moe_stage1_fwd/stage2_fwd: GPU memory fault
14. Reusing moe_out/sorting buffers: GPU crash
15. Custom Triton MoE: JIT timeout on ephemeral runner
16. Stage2 v3 for E=33: same speed as v1
17. 256x64x128x128 stage1 for E=33: 30% WORSE
18. FlyDSL: available but 0 binaries on runner
19. torch.compile: dead
20. All env vars (AITER_USE_NT, AITER_KSPLIT, AITER_CONFIG_FMOE, CU_NUM): exhausted

---

## Runner Environment
- GPU: AMD Instinct MI355X (gfx950, CDNA4, 304 CUs, 8 XCDs, 5300 GB/s HBM, 32MB L2, 64MB MALL)
- CPU: AMD EPYC 9575F
- OS: Linux 6.8.0-60, Ubuntu
- PyTorch: 2.10.0+rocm7.1
- aiter: commit f3be04a12 (NOT updated, no PRs #2261/#2440/#2497)
- Ephemeral Kubernetes pods — Triton JIT cache destroyed per submission
- pip install BLOCKED
- Internet works (wget/curl)
- "stream" word is grep-filtered from submissions
- Rate limits: 6 benchmark/hr, 1 leaderboard/hr per problem

### aiter Key APIs
```python
# GEMM
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4, gemm_a16wfp4_preshuffle
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter import gemm_a4w4  # CK ASM, API changed — no kernelName param

# MLA
from aiter.mla import mla_decode_fwd
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# MoE
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType
```

### Quant Functions Available
```python
dynamic_mxfp4_quant, dynamic_per_tensor_quant_fp8_i8, dynamic_per_token_quant_fp8_i8,
fused_dynamic_mxfp4_quant_moe_sort, fused_flatten_fp8_group_quant, fused_flatten_mxfp4_quant,
fused_fp8_quant, fused_mxfp4_quant, fused_reduce_act_mul_and_mxfp4_quant,
fused_reduce_act_mul_fp8_group_quant, fused_reduce_rms_fp8_group_quant,
fused_reduce_rms_mxfp4_quant, fused_rms_fp8_group_quant,
fused_rms_fp8_per_tensor_static_quant, fused_rms_mxfp4_quant,
static_per_tensor_quant_fp8_i8
```

---

## What Would Change Everything
1. **Runner aiter update** (PRs #2261, #2440, #2497) — new configs, qseqlen dispatch, new kernels
2. **Exploit scrub** — removes invalid top scores, improves relative ranking
3. **Finding aiter's internal preshuffle format** — could eliminate 5-6μs GEMM overhead
4. **MoE ksplit working without cktile** — competitors use ksplit=2 somehow
5. **pg2 accuracy fix in ASM kernel** — would make MLA 42μs reliable (100% pass)

## Open Questions for Research
1. How are competitors (josusanmartin, Ananda Sai A) achieving 7-8μs GEMM? What path are they using?
2. Why does ksplit=2 for MoE work for competitors but triggers cktile timeout for us? Different aiter version?
3. Is there a way to call gemm_a16wfp4_preshuffle with the eval harness's B_shuffle format? What's the exact shuffle permutation difference?
4. Can the pg2 ~4% mismatch be reduced by modifying how metadata is built? (kv_indptr rounding, page alignment)
5. Are there undocumented aiter APIs or env vars that top competitors are using?
6. Can load_inline HIP kernels achieve better MFMA utilization than Triton for small-M GEMV workloads?
