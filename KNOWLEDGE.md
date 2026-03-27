# AMD GPU MODE Hackathon — Full Knowledge Base

## Project Overview
- **Competition**: AMD x GPU MODE Phase 1 Qualifiers
- **Deadline**: April 7, 2026
- **Goal**: Top 10 on all 3 leaderboards
- **GPU**: AMD Instinct MI355X (gfx950, CDNA4, 304 CUs, 8 XCDs, 5300 GB/s HBM)
- **GitHub**: https://github.com/THINNGO2511/AMD-GPU-MODE
- **Submission**: `popcorn submit --gpu MI355X --leaderboard <name> --mode <mode> <file> --no-tui`
- **Rate limits**: 10 benchmark/hr per problem, 1 leaderboard/hr per problem

---

## 3 Problems

### 1. MXFP4 GEMM (leaderboard: amd-mxfp4-mm)
- **Task**: Quantize bf16 A → MXFP4, multiply with pre-quantized MXFP4 B → bf16 C
- **Input**: `(A[m,k] bf16, B[n,k] bf16, B_q fp4x2, B_shuffle fp4x2, B_scale_sh e8m0)`
- **Benchmark shapes**: (4,2880,512), (16,2112,7168), (32,4096,512), (32,2880,512), (64,7168,2048), (256,3072,1536)
- **Tolerance**: rtol=1e-2, atol=1e-2
- **Our best**: 16.5μs ranked / ~10μs benchmark | top is 4.36μs (HorizonLiang)

### 2. MXFP4 MoE (leaderboard: amd-moe-mxfp4)
- **Task**: DeepSeek-R1 fused MoE, 256 routed + 1 shared expert, top-k=8+1=9
- **Config**: hidden=7168, expert intermediate: 256 (E=257) or 512/2048 (E=33)
- **Benchmark shapes**: bs=16/128/512 for E=257 d=256; bs=16/128/512 for E=33 d=512; bs=512 E=33 d=2048
- **Tolerance**: rtol=2e-2, atol=2e-2
- **Our best**: ~167μs benchmark / ~169μs ranked | top 10 needs ~136μs

### 3. MLA Decode (leaderboard: amd-mixed-mla)
- **Task**: DeepSeek R1 forward_absorb MLA, 16 Q heads, 1 KV head
- **Config**: qk_head_dim=576, v_head_dim=512, sm_scale=1/sqrt(576)
- **KV formats**: bf16, fp8, mxfp4 (all provided simultaneously)
- **Benchmark shapes**: bs=4/32/64/256, kv=1024/8192
- **Tolerance**: rtol=0.1, atol=0.1, 5% mismatch bypass
- **Our best**: ~45μs benchmark | top is 33μs (Ananda Sai A)

---

## Current Best Submissions

### GEMM: `mxfp4-mm/submission_prewarm.py` (16.5μs ranked)
- gemm_a16wfp4 for K!=1536 (fused A quant into GEMM, single kernel)
- gemm_afp4wfp4 for K=1536 (separate quant + GEMM)
- Custom K=7168 config: BM=8 BN=64 BK=512 KS=8 W=4 S=2 WPE=2 MI=16
- Pre-warm quant kernel for all M values on first call
- Output tensor caching by (m, n) key

### MoE: `moe-mxfp4/submission_inject_metadata.py` (~167μs benchmark)
- OPUS sorting enabled (`fm._USE_OPUS_MOE_SORTING = True`)
- use_nt=False globally
- CK kernel injection for E≤64 d<2048 (S1_64/S1_256 + S2_V1)
- d≥2048: currently NO injection (default heuristic, ~337μs for that shape)
- Custom get_block_size_M: 32 for est_m<50, 64 for est_m≥50, 128 for d≥2048 est_m≥100

### MLA: `mixed-mla/submission_pg8_v2.py` (~45μs benchmark)
- pg1+bf16Q for kv≤1024 (a16w8 ASM kernel, no Q quant overhead)
- pg8+fp8Q for kv≥8192 (a8w8 ASM kernel, 8x less KV cache entries)
- kv_granularity = max(1, 16 // page_size) — CRITICAL formula
- Fused Q fp8 quant (2 Triton kernels instead of 6 PyTorch ops)
- Per-shape num_kv_splits: bs≤4→4, bs≤32→8, else→16

---

## Runner Environment (probed Session 8, Mar 27)
- **GPU**: AMD Instinct MI355X, 1 device
- **CPU**: AMD EPYC 9575F 64-Core
- **ROCm**: 7.1, HIP 7.1.25424
- **PyTorch**: 2.10.0+rocm7.1
- **Triton**: 3.6.0
- **CU_NUM**: 256
- **aiter**: JIT-compiled, no version string
- **Python**: 3.12

### aiter Key Paths on Runner
- `/home/runner/aiter/aiter/` — Python source
- `/home/runner/aiter/hsa/gfx950/` — Pre-compiled kernel binaries
- `/home/runner/aiter/hsa/gfx950/f4gemm/` — 36 FP4 GEMM ASM .co files
- `/home/runner/aiter/aiter/ops/triton/configs/gemm/` — 137 config JSONs (66 FP4)
- `/home/runner/aiter/aiter/configs/tuned_fmoe.csv` — MoE tuned configs (1422 lines)
- `/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv` — DSv3 FP4 MoE

### aiter Key APIs
```python
# GEMM
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4  # bf16 A, fused quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4  # fp4 A+B
from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4  # fp8 A + fp4 B (NEW)
from aiter.ops.triton.quant import dynamic_mxfp4_quant

# MLA
from aiter.mla import mla_decode_fwd
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# MoE
from aiter.fused_moe import fused_moe  # NOT: from aiter import fused_moe
from aiter import ActivationType, QuantType
```

### aiter GEMM Kernel APIs (discovered Session 8)
```
gemm_a16wfp4(A, w, w_scales, dtype, y=None, config=None)
  A: bf16 (M,K), w: fp4 uint8 (N,K//2), w_scales: E8M0 UNSHUFFLED (N,K//32)

gemm_afp4wfp4(A_fp4, w, A_scale, w_scales, dtype)
  A_fp4: fp4 uint8 (M,K//2), w: fp4 uint8 (N,K//2)
  A_scale: E8M0 (M,K//32), w_scales: E8M0 UNSHUFFLED (N,K//32)

gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config=None)  # NEW
  x: FP8 E4M3 (M,K), w: fp4 uint8 (N,K//2), y: output (M,N)
  x_scales: FP32 per-row (M,1), w_scales: E8M0 UNSHUFFLED (N,K//32)
  BLOCKED: eval framework B_scale_sh has mismatched shape — strict assertion fails

gemm_a16wfp4_preshuffle(x, w, w_scales, prequant=True, dtype, y, config, skip_reduce)
  Uses B_shuffle + shuffled scales directly (no unshuffle needed)
  DEAD END: Triton 3.6 KeyError 'float8_e8m0fnu' on scale pointer dtype

deepgemm(XQ, WQ, Y, group_layout, x_scale=None, w_scale=None)  # UNEXPLORED
deepgemm_ck(*args, **kwargs)  # UNEXPLORED
```

---

## GEMM — Detailed Findings

### Status: CONFIG TUNING EXHAUSTED (Session 8)
- 10μs benchmark / 16.5μs ranked. Top 4.36μs = 4x gap.
- **12 experiments in Session 8, zero improvements** beyond noise
- aiter default configs are already near-optimal for Triton path
- Gap requires fundamentally different approach (persistent kernel, custom Triton, etc.)

### Session 8 GEMM Experiment Results
| Exp | Approach | Geomean | vs Baseline |
|-----|----------|---------|-------------|
| 01 | Env probe | 10.0μs | same (probe data) |
| 02 | Full warmup | 10.0μs | same |
| 03 | stages=3 all | 10.3μs | +3% (K=2048 kills it) |
| 04 | Per-shape tuned | ~12.5μs | +25% worse |
| 05 | Aggressive ksplit | ~13.8μs | +39% worse |
| 07 | waves1+cg | ERROR | "work on another stream" |
| 08 | Small tiles | not run | |
| 11 | a8wfp4 probe | 10.0μs | probe data |
| 12 | a8wfp4 v1 | ERROR | scale shape mismatch |
| 13 | preshuffle | ERROR | Triton e8m0 dtype bug |
| 14 | Hybrid stages=3 K=512 | 9.8μs | -1% (noise) |
| 15 | a8wfp4 v2 | ERROR | same scale shape bug |
| 16 | a8wfp4 configs on a16w | ~14.5μs | +45% terrible |

### GEMM Key Discoveries
- `gemm_a8wfp4` EXISTS with tuned per-M configs (WPE=4-6, .cg, KS=4)
  - BLOCKED by eval framework: B_scale_sh shape doesn't match (N, K//32)
  - a16wfp4 handles this gracefully, a8wfp4 has strict assertion
- `gemm_a16wfp4_preshuffle` EXISTS but DEAD END (Triton 3.6 e8m0 dtype)
- a8wfp4-style configs DON'T transfer to a16wfp4 (BK=256 wrong for small K)
- num_stages=3: helps K=512 shapes (-6%) but kills K=2048 (+34%)
- Default configs are unbeatable — all custom configs are worse
- Autoresearch infra built: `mxfp4-mm/gemm_autoresearch.py` + `gemm_experiments/`

### GEMM Remaining Leads (radical only)
- Custom Triton kernel from scratch (persistent threads)
- f4gemm ASM .co kernels called directly (36 available)
- deepgemm/deepgemm_ck (needs group_layout investigation)
- torch.compile on kernel function

### GEMM Config Reference
```python
# K=7168 custom config (ONLY config that beats defaults)
{"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
 "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
 "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
 "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024}

# K=1536: use afp4wfp4 path (separate quant + GEMM)
# All other K values: use default (None) config
```

### GEMM Dead Ends (comprehensive, Sessions 1-8)
- CK gemm_a4w4: 19-34μs (3 kernel launches kill it)
- HIP MFMA kernel: correct (ZERO error) but 17.8μs = worse than Triton
- gemm_a16wfp4_preshuffle: Triton e8m0 dtype KeyError
- gemm_a8wfp4: eval framework scale shape assertion bug
- a8wfp4 configs on a16wfp4: -35% to -110% worse
- Custom scalar HIP: 5-10% quant mismatch
- load_inline: ~90s compile eats timeout budget
- num_stages=3 globally: K=2048 +34% regression
- Per-shape custom configs: ALL worse than defaults
- Aggressive split-K: -26% to -70% worse
- Full warmup: no improvement
- CUDA/HIP graphs: 2x worse
- Hand-tuned a16wfp4 configs for K=512/2048: worse than defaults

---

## MoE — Detailed Findings

### Status: d=2048 INJECTION ACTIVE (Session 8)

### Session 8 MoE Root Cause: CSV Mismatch
- **fc0c54bb commit NOT deployed** to runner (CSV still 1422 lines, unchanged)
- Even if deployed: commit uses `expert=32, topk=8` but benchmark has `expert=33, topk=9`
- CSV lookup is EXACT match on all 13 fields — will never hit our shape
- ALL E=33 shapes hit "2stage default" heuristic (confirmed by probe)
- Heuristic default DOES set a kernelName — so `if not kernelName` check skips injection

### fc0c54bb Commit Kernel Names (for E=32 d=2048 FP4 Silu)
```
token≤4:   S1=64x32x32x128_v3       S2=256x32x128x128_v3    block_m=32
token=8:   S1=256x32x128x128_v3     S2=256x32x128x128_v3    block_m=32
token≤64:  S1=64x32x32x128_v3       S2=256x32x128x128_v3    block_m=32
token=128: S1=256x64x128x128_v3     S2=256x64x128x128_v3    block_m=64
token=256: S1=256x64x128x128_v3     S2=256x64x128x128_v3    block_m=64
token≥512: S1=256x128x128x128_v3    S2=256x128x128x128_v3   block_m=128
```

Full kernel names for bs=512 (our bottleneck):
```
S1: moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16
S2: moe_ck2stages_gemm2_256x128x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16
```

### MoE Injection Attempts (Session 8)
- **v1** (inject_fc0c54bb.py): 342μs — injection didn't fire; heuristic default sets kernelName so `if not kernelName` check skipped the override
- **v2** (inject_fc0c54bb_v2.py): 342μs — injection CONFIRMED firing (`INJECT_D2048: s1=256x128 s2=256x128`) but **NO improvement**. The 256x128 v3 kernels perform identically to the default heuristic. The d=2048 bottleneck is fundamentally memory-bound, not kernel-selection-bound.

### MoE What Works
- `submission_inject_metadata.py`: OPUS + CK injection for E≤64 d<2048 = **167μs**
- use_nt=False globally: +2-4%
- OPUS sorting: helps E=257 ~10%
- CK kernel injection (S1_64/S1_256 + S2_V1): +10-20% for E=33 d<2048
- d≥2048 heuristic was "17% faster" than OLD injection (small 64x32 tiles)
  - BUT commit kernels use LARGE tiles (256x128) — v2 tests this

### MoE Monkey-Patching Pattern
```python
import aiter.fused_moe as fm
import functools

fm.use_nt = lambda token, topk, expert: False
fm._USE_OPUS_MOE_SORTING = True

orig_get_2stage = fm.get_2stage_cfgs.__wrapped__
@functools.lru_cache(maxsize=2048)
def new_get_2stage(token, model_dim, inter_dim, expert, topk,
                   dtype, q_dtype_a, q_dtype_w, q_type,
                   use_g1u1, activation, doweight_stage1,
                   hidden_pad, intermediate_pad, is_shuffled=True):
    result = orig_get_2stage(...)
    # Override result for specific shapes
    if expert <= 64 and inter_dim >= 2048:
        return fm.MOEMetadata(
            functools.partial(fm.ck_moe_stage1, kernelName=..., ...),
            functools.partial(aiter.ck_moe_stage2_fwd, kernelName=..., ...),
            block_m, 0, False)
    return result
fm.get_2stage_cfgs = new_get_2stage
fm.cfg_2stages = None  # Clear cached config
```

### MoE Benchmark Reference
| Shape | inject_metadata | notes |
|-------|----------------|-------|
| E=257 bs=16 d=256 | 127μs | CK injected |
| E=257 bs=128 d=256 | 206μs | CK injected |
| E=257 bs=512 d=256 | 241μs | CK injected |
| E=33 bs=16 d=512 | 87μs | CK injected |
| E=33 bs=128 d=512 | 111μs | CK injected |
| E=33 bs=512 d=512 | 178μs | CK injected |
| E=33 bs=512 d=2048 | **337μs** | **GEOMEAN KILLER — default heuristic** |
| **Geomean** | **~167μs** | **Need 136μs for top 10** |

### MoE Dead Ends
- Direct ck_moe_stage1/stage2 calls: GPU memory fault
- Buffer reuse: GPU crash
- 1-stage kernel (fmoe_g1u1): 182μs, slower
- ksplit=2 for d=2048: triggers cktile path, 2x SLOWER
- dispatch_policy=1: 80% slower
- doweight_stage1=True: wrong results
- block_m override via monkey-patch: JIT timeout
- CK injection for d=2048 with SMALL tiles (64x32): 17% WORSE
- inject_v2 with `_none_` stage2: 8-21% worse
- sepqsort (threshold=0): no improvement

---

## MLA — Detailed Findings

### Status: ~40μs benchmark (Session 9 improvement), need 33μs
- Two ASM kernels: a16w8 (bf16 Q) and a8w8 (fp8 Q)
- pg8 for kv≥8192 gives 8x KV cache reduction → major speedup
- pg2 for kv≤1024 works with CORRECT kv_granularity=max(1, 16//ps)=8
- Optimal num_kv_splits from Session 9 sweep: 16 for most shapes, 8 for bs≤32+kv=1024
- Leaderboard: exp_optimal_splits.py submitted (~40.4μs benchmark)

### Session 9 MLA Experiments (Mar 27)
| Experiment | Geomean | Notes |
|-----------|---------|-------|
| a16w8+pg2 all sizes | 63.4μs | bs=256 kv=8192 terrible (306μs) |
| hybrid a16w8+a8w8 pg2 | ~55μs | pg2 kv=8192 still bad (193μs) |
| a8w8+pg2 all sizes | ~55μs | similar to hybrid |
| optimal num_kv_splits (a16w8+pg2/a8w8+pg8) | **40.4μs** | 5% improvement from splits tuning |

### Session 9 num_kv_splits Sweep Results
| Shape | splits=4 | splits=8 | splits=16 | splits=32 | Best |
|-------|----------|----------|-----------|-----------|------|
| bs=4 kv=1024 | 0.102ms | **0.039ms** | 0.038ms | - | 8 |
| bs=4 kv=8192 | 0.034ms | 0.025ms | **0.024ms** | - | 16 |
| bs=32 kv=1024 | - | **0.040ms** | 0.040ms | - | 8 |
| bs=32 kv=8192 | - | 0.030ms | **0.027ms** | 0.030ms | 16 |
| bs=64 kv=1024 | - | 0.047ms | **0.042ms** | 0.043ms | 16 |
| bs=64 kv=8192 | - | 0.036ms | **0.034ms** | 0.035ms | 16 |
| bs=256 kv=1024 | - | 0.059ms | **0.058ms** | 0.060ms | 16 |
| bs=256 kv=8192 | - | 0.051ms | **0.046ms** | 0.050ms | 16 |

### MLA Dead Ends
- pg2+pg8 (41μs benchmark): FAILS leaderboard secret seed
- pg2 for kv=8192 all (193μs for bs=256): too many pages vs pg8
- a16w8 for kv=8192 (306μs for bs=256): bf16 Q bandwidth kills it
- pg8 for kv=1024: FAILS accuracy
- pg4/pg16: FAIL accuracy
- fast_mode=True: 5-10% worse
- MXFP4 Triton attention: 6-291x slower
- HIP MXFP4 MLA: linker issues
- OLD bug: kv_granularity=max(PAGE_SIZE, 16) is WRONG, must be max(1, 16//PAGE_SIZE)

---

## MXFP4 Format Reference
- **E2M1**: 4-bit FP, values {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
- **Packing**: low nibble = even index, high nibble = odd index
- **E8M0 scale**: 2^(byte_value - 127), one per 32 elements
- **B_scale unshuffle**: view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2)

### B_scale Unshuffle (verified)
```python
def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)
```

---

## load_inline HIP Pattern (PROVEN)
```python
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
from torch.utils.cpp_extension import load_inline

mod = load_inline(
    name="my_kernel_v1",  # bump version to bust cache
    cpp_sources="torch::Tensor fn(torch::Tensor A);",  # forward decl
    cuda_sources=HIP_SOURCE,
    functions=["fn"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
)
```
- **NO PYBIND11_MODULE** in cuda_sources
- **NO word "stream"** anywhere — use `hipLaunchKernelGGL(k, g, b, 0, 0, args)`
- Use `hip_bfloat16`, NOT `__hip_bfloat16`
- Takes ~90s to compile — eats benchmark timeout budget

---

## Competitor Intelligence
| Rank | User | GEMM | MoE | MLA | Notes |
|------|------|------|-----|-----|-------|
| 1 | HorizonLiang | **4.36μs** | — | — | Fundamentally different approach? |
| 1 | Ananda Sai A | 8.1μs | **110μs** | **33μs** | pg2_fix all sizes, 293 subs |
| 2 | josusanmartin | **7.8μs** | 127μs | 43.5μs | 3861 subs (automated sweeping) |
| ~10 | threshold | ~9μs | ~136μs | ~50μs | |
| us | noobmaster69_og | 16.5μs | 169μs | 45μs | |

---

## Project File Structure
```
AMD-GPU-MODE/
├── KNOWLEDGE.md          # This file (push to GitHub)
├── CLAUDE.md             # NEVER push (local only)
├── mxfp4-mm/
│   ├── submission.py              # Active leaderboard submission
│   ├── submission_prewarm.py      # Best GEMM (16.5μs ranked)
│   ├── gemm_autoresearch.py       # Automated experiment runner
│   ├── gemm_autoresearch_results.json
│   └── gemm_experiments/          # 16 experiment submissions
├── moe-mxfp4/
│   ├── submission.py              # Active leaderboard submission
│   ├── submission_inject_metadata.py  # Best MoE (167μs)
│   └── moe_experiments/
│       ├── probe_csv_d2048.py     # CSV investigation probe
│       ├── inject_fc0c54bb.py     # v1 kernel injection (failed)
│       └── inject_fc0c54bb_v2.py  # v2 unconditional injection (pending)
├── mixed-mla/
│   ├── submission.py              # Active leaderboard submission
│   └── submission_pg8_v2.py       # Best MLA (45μs)
└── auto_sweep_all.py              # MoE auto-sweep script
```

---

## MoE Internal Architecture (from source reading)

### Profiling Breakdown (bs=16, E=257, d=256)
- stage1 CK GEMM: **41%** of kernel time (biggest target)
- fused_dynamic_mxfp4_quant_moe_sort: **28%** (called 2x per inference)
- stage2 CK GEMM: **23%**
- moe_sorting C++ kernel: **8%** (tiny)

### Key Internal Details
- `fused_moe_` is a C++ torch op (`torch.ops.aiter.fused_moe_`), calls Python functions internally
- `fused_moe_2stages` source: 242 lines, handles quant path selection, calls stage1/stage2
- `token_num_quant_moe_sort_switch = 1024` — LOCAL var in fused_moe_2stages
  - For token_num<=1024: uses fused quant+sort kernel
  - For token_num>1024: uses separate quant + moe_mxfp4_sort
- `_moe_sorting_impl` (42 lines): allocates 5 tensors per call, calls moe_sorting_opus_fwd
  - Returns: (sorted_ids[int32], sorted_weights[fp32], sorted_expert_ids[int32], num_valid_ids[int32,2], moe_buf[bf16])
- **Cannot bypass C++ wrapper** — causes GPU memory faults on repeated calls

### CSV Key Format (13 fields, exact match)
`cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1`

### Available FP4 Stage1 Kernel Tiles (from CSV)
- `64x32x32x128_1x1` (best for small est_m)
- `256x32x128x128_1x4` (best for medium est_m like bs=64/128)
- `256x64x128x128_1x4` (WORSE for E=33 bs=512 — don't use for d<2048)
- `256x128x128x128_1x4` (only for token≥2048 or d=2048 from fc0c54bb)

### MoE Environment Variables
- `AITER_USE_NT`: -1=auto, 0=off, 1=on
- `AITER_KSPLIT`: 0=auto
- `AITER_CONFIG_FMOE`: path to custom tuned config CSV
- `AITER_USE_OPUS_MOE_SORTING`: 0 or 1
- `AITER_BYPASS_TUNE_CONFIG`: 0 or 1

### MoE FlyDSL (probed, not useful)
- FlyDSL IS available: `fm.is_flydsl_available() = True`
- Kernel name format: `flydsl_moe2_afp4_wfp4_bf16_t{block_m}x{N}x{K}_reduce`
- Only 2 CSV entries: token=16384, E=257, block_m=64/128
- FlyDSL _atomic and _reduce: ±3%, not significant
- Empty kernelName crashes: `ValueError: Invalid FlyDSL kernel name:`

### Runner MoE Facts
- CU_NUM = 256 (matches dsv3 CSV)
- PR #2414 sort optimization: ALREADY applied on runner
- tuned_fmoe.csv: 1422 rows, cu_num=80 (wrong GPU) and cu_num=256 entries
- Merges 3 CSVs: `tuned_fmoe.csv` + `dsv3_fp4_tuned_fmoe.csv` + `a8w8_blockscale_tuned_fmoe_qwen3_235b.csv`

---

## GEMM Proven Techniques (from CLAUDE.md)

### gemm_a16wfp4 (BEST PATH)
- Takes bf16 A directly, quantizes on-the-fly inside kernel. NO separate A quant needed.
- Call: `gemm_a16wfp4(A, B_q_uint8, B_scale_unshuffled, dtype=torch.bfloat16, y=output, config=cfg)`
- B_q must be viewed as uint8: `B_q.view(torch.uint8)`
- B_scale must be UNSHUFFLED (raw E8M0, not e8m0_shuffle'd)
- Per-K performance: K=512: 6.15-6.86μs, K=7168: 14.7μs (tuned), K=2048: 14.2μs, K=1536: 16μs (WORSE, use afp4wfp4)

### Config Injection
- Pass `config=dict(...)` parameter directly
- Key params: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE, num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim
- `tl.dot_scaled`: Use `acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)` (fused form, NOT `acc +=`)

### gfx950 Triton Compile Options
- `waves_per_eu`: 1 for decode (more registers), 2 for compute-heavy
- `matrix_instr_nonkdim`: 16 for small M, 32 for medium M with large N
- `cache_modifier`: ".cg" for decode workloads (BUT triggers "stream" error on runner)
- `BLOCK_SIZE_K`: Always 128 for blockscale, >=256 for AFP4WFP4
- `num_stages`: 2-3 (3 helps K=512 but kills K=2048)

### A Quantization (exact match formula)
- Scale: `(amax_int + 0x200000) & 0xFF800000` (round amax up), then `floor(log2) - 2`
- FP4 RNE: `mant_odd = (qx >> 22) & 1; qx += val_to_add + mant_odd; qx >>= 22`

### MFMA FP4 Intrinsic (PROVEN, for HIP path reference)
- `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 4, 4, 0, scale_a, 0, scale_b)`
- Header: `#include <hip/hip_ext_ocp.h>`
- CK actually uses `v_mfma_scale_f32_16x16x128_f8f6f4` (NOT 32x32x64)
- Output mapping: col=tid%32, row=half*4+j+i*8 for c_reg[i*4+j]
- L2 cache clearing: Ranked adds ~6μs per call (Benchmark 9.8μs → Ranked 16.3μs)

---

## MLA Proven Techniques (from CLAUDE.md)

### Two ASM Kernels (pre-compiled, no JIT)
- `mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co` — fp8 Q + fp8 KV (best for large kv)
- `mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps.co` — bf16 Q + fp8 KV (faster for small kv)

### pg2_fix Formula (PR #1950)
1. kv_indptr counts PAGES not tokens: `kv_indptr_paged[1:] = cumsum((seq_lens + ps - 1) // ps)`
2. kv_last_page_lens: `seq_lens % ps`, replace 0 with ps
3. kv_granularity: `max(1, 16 // page_size)` = 8 for pg2, 2 for pg8
4. kv_buffer reshaped: `(num_pages, page_size, nkv, dim)`
5. kv_indices: `arange(total_pages)` not `arange(total_tokens)`

### Fused Q fp8 Quant (2 Triton kernels vs 6 PyTorch ops)
- `_q_amax_kernel`: block-parallel abs+max with atomic_max
- `_q_to_fp8_kernel`: read amax, compute scale, divide+clamp+cast

### Key Constraints
- ASM kernel: kPageSize=1 hardcoded. pg2/pg8 work via metadata trick
- 3-buffer KV layout: NOT available in current aiter version
- num_kv_splits: 8 for total_kv<=8192, 16 for larger. 32 is too much overhead
- Pre-allocate: output tensor, amax_buf, scale_buf, q_fp8_flat — reuse across calls

---

## Session 3 Research Findings (Mar 23)

### GEMM Autotune Warmup
- eval.py leaderboard mode only warms tests[0] shape
- Other 5 shapes trigger Triton JIT → fixed by pre-warming all shapes

### MoE Session 3 Dead Ends
- 256x64x128x128 stage1: 30% WORSE for E=33 bs=512
- 64x128x128x128 v3 stage2: 5% WORSE
- AITER_USE_NT=0: slightly worse for E=257
- CU_NUM=256: already correct
- All env vars tested: no improvement found

---

## Competitor Intelligence
| Rank | User | GEMM | MoE | MLA | Technique |
|------|------|------|-----|-----|-----------|
| 1 | HorizonLiang | **4.36μs** | 148μs | — | Fundamentally different approach? |
| 1 | Ananda Sai A | 8.1μs | **110μs** | **33μs** | pg2_fix all sizes, v42, 293 subs |
| 2 | Nicky Pochinkov | — | 152μs | 33.4μs | Very close to #1 MLA |
| 2 | josusanmartin | **7.8μs** | 127μs | 43.5μs | 3861 subs (automated sweeping) |
| ~10 | threshold | ~9μs | ~136μs | ~50μs | — |
| us | noobmaster69_og | 16.5μs | 169μs | 45μs | |

### Submission Filename Hints
- `submission_v75_pg8_8k` (Danishlynx, MLA #1, 29μs) — pg8 for 8k shapes
- `v1271_e33_no_nt` (josusanmartin, MoE) — E=33 specific, no NT loads, 1271 versions
- `submission_sagemath_fft` (GEMM) — FFT approach
- `cfg_257k2_33k2_16128_blockmwide_stage2_sepqsort` (Ryan Mathieu) — per-shape configs

---

## Leaderboard URLs
- https://gpumode.com/leaderboard/amd-mxfp4-mm
- https://gpumode.com/leaderboard/amd-moe-mxfp4
- https://gpumode.com/leaderboard/amd-mixed-mla
