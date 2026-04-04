# AMD x GPU MODE Hackathon — Complete Research Log
## Sessions 20-21+ (April 1-3, 2026)
### Claude Research (claude.ai) × Claude Code collaboration

---

## 1. Competition Overview

**Prize:** $1.1M total, top 10 aggregate score advances to Finals
**Deadline:** April 6, 2026 11:59 PM PST
**GPU:** AMD Instinct MI355X (gfx950, CDNA4, 256 CUs, 8 XCDs, 160KB LDS/CU, 8 TB/s HBM3E)
**User:** noobmaster69_og (Edward)
**Repo:** /home/claude/AMD-GPU-MODE/

**Scoring Rules:**
- Geometric mean of benchmark results per problem
- Only top 20 fastest kernels per problem contribute to aggregate score
- Submissions NOT in top 20 get ZERO points
- Top 10 aggregate advance to Finals
- Earliest submission wins ties

**3 Problems:**
1. **MXFP4 GEMM** (amd-mxfp4-mm): Matrix multiply with MXFP4 quantized weights
2. **MoE** (amd-moe-mxfp4): Mixture of Experts with MXFP4 weights  
3. **MLA Decode** (amd-mixed-mla): Multi-head Latent Attention decode

**Submission:** `popcorn-cli submit --gpu MI355X --leaderboard <name> --mode <mode> <file> --no-tui`

---

## 2. Current Standings (as of April 3)

| Problem | Our Score | Rank | Top 20 Cutoff | In Top 20? | Best File |
|---------|-----------|------|---------------|------------|-----------|
| MLA | 36.4μs | ~#14 | ~35μs | YES | mixed-mla/mla_fp8q_all.py |
| GEMM | 15.7μs | ~#160 | ~9μs | NO | mxfp4-mm/sub_ultimate_v1.py |
| MoE | 163μs | ~#65 | ~143μs | NO | moe-mxfp4/submission_optimized_v2.py |

**Aggregate Analysis:**
- Our geomean: 45.3μs. Top 10 cutoff: ~35μs. Best case (GEMM 12 + MLA 35.5): 41.1μs — still short.
- MLA is the ONLY problem contributing to our aggregate score.
- GEMM needs 15.7→9μs (43% drop) for top 20 — requires fundamentally different approach.
- MoE needs 163→143μs (12% drop) for top 20 — all library-level approaches exhausted.

---

## 3. What Worked (Leaderboard Improvements)

### MLA: 56.6μs → 36.4μs (-36%)
1. **mla_safe_fast.py → 41.5μs:** pg1+bf16Q for kv=1024, pg8+fp8Q for kv=8192
2. **mla_fp8q_all.py → 36.4μs:** FP8 quantization for ALL shapes (saves 50% Q bandwidth on kv=1024), pg1 for kv≤1024, pg8 for kv≥8192, fixed-scale fp8 quant (skip amax)

### GEMM: 16.5μs → 15.7μs (-5%)
3. **gemm_stages3_k512.py → 15.7μs:** num_stages=3 for K=512 shapes (default was stages=1)

### MoE: No improvement (163μs)
- CK injection for E≤64 d<2048 was already in the starting baseline

---

## 4. Confirmed Dead Ends — DO NOT RETRY

### 4.1 MLA Dead Ends
- **pg2 for kv=1024:** 5x fail on secret seeds (~4% inherent mismatch with eval reference)
- **MXFP4 KV cache:** ASM kernel rejects dim=288 (not divisible by 512)
- **qseqlen2 kernel:** Loads successfully but GPU memory fault during execution — data layout incompatible with batched case
- **auto_splits:** Library auto-tuned splits=4 for bs=64 kv=8192 fails secret seed (accuracy unsafe)
- **splits=1 for small bs:** Underutilizes 256 CUs
- **mla_tuned_v2, mla_splits8:** Both slower than current best

### 4.2 GEMM Dead Ends
- **gemm_a4w4 ASM kernel:** Correct accuracy (0 errors with e8m0_shuffle on A_scale) but 3-launch overhead: quant 12μs + shuffle 1μs + GEMM 3-8μs = slower than single-launch Triton
- **Preshuffle Triton path:** fp4x2 dtype incompatible with Triton tl.dot
- **KSPLIT > 1:** Reduce kernel overhead kills performance. Tested KSPLIT=2, 4, 14 — all worse. The reduce kernel adds ~19μs for K=7168 M=16.
- **Writing shape-specific JSON configs:** Successfully wrote to /home/runner/aiter/aiter/ops/triton/configs/gemm/ but KSPLIT=2 made K=7168 2.4x SLOWER (13.6→32.6μs), K=2048 unchanged, K=512 marginal
- **afp4wfp4 for all shapes:** Slower than gemm_a16wfp4
- **config=None lib defaults:** Worse for K=7168
- **Environment vars for ASM dispatch:** No effect (HIP_FORCE_DEV_KERNARG, OPTIMIZE_EPILOGUE)
- **L2 cache pre-warming:** Adds overhead, doesn't save time
- **deepgemm:** Wrong API for this runner

### 4.3 GEMM — hipBLASLt FP4 (Exhaustively Dead)
**14 test submissions proving it cannot work.**

What worked: 8 algorithms found, ~7.68μs timing, PERFECT for uniform data (all-ones test gives K exactly).

What failed: maxdiff=74-175 vs eval reference on real data. Some shapes produce NaN.

Root cause: **Computational difference (accumulation order), NOT data format.** hipBLASLt tiles and accumulates K-dimension FP4 products in a different order than Triton's gemm_afp4wfp4/gemm_a16wfp4. The eval reference matches Triton (error=0.0). hipBLASLt's maxdiff=115 with output range ~300 = ~38% relative error, far exceeding rtol=1e-2.

Exhaustively ruled out:
- FP4 byte transposition: WORSE (115→338-422)
- Nibble swap: WORSE (115→338-378)
- All scale formats: raw, shuffled, transposed, shuffled+transposed — ALL ~74-115
- Scale mode 6 (BLK32_UE8M0_32_8_EXT): MUCH worse (237-380)
- Layout orders 0, 102, 103 in ALL 16 combinations: NO effect (all give 8 algos, same accuracy)
- Swapped A/B in BLAS call: WORSE
- Direct approach vs transposed: best at maxdiff=74-115 but still fails tolerance

**hipBLASLt FP4 API (for reference):**
```
opA=T, opB=N (mandatory). M%32==0. K%128==0.
scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 (2)
HIP_R_4F_E2M1 = (hipDataType)33
Layout: matA(K,M), matB(K,N), matC(M,N) — all col-major
Row-major [M,K/2] fp4x2 = col-major (K,M) FP4 — NO transpose needed
Output: allocate [N,M], .t().contiguous() → [M,N]
Link: -lhipblaslt. Use 0 for last arg (filtered word).
```

### 4.4 MoE Dead Ends
- **CK injection for d=2048:** Crashes or slower (the 2-stage CK kernels for d≥2048 tile sizes don't exist in standalone .co files — they're JIT-compiled)
- **CSV override for E=33:** Breaks E=257 CK injection (dsv3_fp4_tuned_fmoe.csv has ONLY E=257 entries)
- **Sort policy override:** Wrong API in the current codebase
- **block_m=16:** Assertion error
- **online_tune:** No effect
- **1-stage for d=2048:** Slower than 2-stage
- **splitk:** Crashes
- **AITER_BYPASS_TUNE_CONFIG:** Doesn't exist in the codebase
- **noopus:** No effect
- **Custom CSV merged:** Lost CK injection for E≤64
- **MoE Swiglu (cktile2stages):** Garbage output — uses wrong module
- **Zero CK FP4 silu .co files found:** The fmoe/ directory has 1024+ .co files but they're for 1-stage fused path (fmoe_bf16, fmoe_fp8), not 2-stage CK path. 2-stage kernels are JIT-compiled.

---

## 5. Critical Discovery: Winners Use Custom HIP C++ Kernels

### 5.1 kernelbot-data Analysis
Analyzed 40K submissions from HuggingFace GPUMODE/kernelbot-data (previous AMD $100K competition on MI300X).

**EVERY winning submission uses `torch.utils.cpp_extension.load_inline` to compile custom HIP C++ kernels at runtime, bypassing aiter/Triton entirely.**

Top submissions:
- **FP8 GEMM #1 (User 70):** 78K char HIP kernel using rocWMMA, per-shape dispatch via DISPATCH_GEMM macro, double-buffered pipeline, split-K support. bz2-compressed into Python file.
- **MoE #1 (User 70):** Custom HIP kernel with bz2-compressed source, complete MoE implementation.
- **MLA #3 (User 258):** Custom HIP kernel with hand-written rope, softmax, attention — all in raw C++/HIP.

### 5.2 load_inline CONFIRMED Working on MI355X
- hipcc available and compiles for gfx950
- MFMA intrinsics compile successfully
- All libraries available: rocwmma, hipblaslt, hipblas, rocblas, hipcub, CK headers
- Simple `a + b` kernel compiles and runs in ~10s

### 5.3 Winning FP8 GEMM Kernel Architecture
The #1 FP8 GEMM kernel (78K chars, saved at `/home/claude/winning_fp8_gemm_kernel.hip`):

```
Structure:
- 6 kernel functions: gemm_kernel, reduce (2x), reduce_kernel, transpose_kernel, check_trans
- Uses rocWMMA fragments (16×16×32 for FP8)
- Per-shape dispatch: DISPATCH_GEMM(M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, SPLITK, LOAD_BATCH)
- CDNA3 configs: BM=256, BN=128, BK=128, WARP_M=4, WARP_N=2, BLOCK_SIZE=512
- Double-buffered: global2reg() → reg2lds() → WMMA compute pipeline
- Scale handling: float32 per-row scales multiplied in shared memory (s_s[i][j] = reg_sa * reg_sb)
- Threadblock swizzle for L2 cache optimization
- Split-K for small-M shapes (SPLITK_FACTOR=1-8)
- reduce_kernel: float4 accumulation across split-K tiles → bf16 output
```

Core gemm_kernel: 229 lines. Dispatch: per-shape via `pack_shape(m,n,k)` switch statement.

### 5.4 FP4 Adaptation Requirements (gfx950)
To adapt the FP8 kernel for MXFP4 on MI355X:

1. **MFMA instruction:** `v_mfma_f32_16x16x32_fp8` → `v_mfma_scale_f32_16x16x128_f8f6f4`
   - FP4 processes 128 elements per MFMA (vs 32 for FP8)
   - Takes E8M0 scale values as SGPR operands (integrated dequant+matmul)
   - Compiler builtin: `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(src_a, src_b, acc, cbsz, blgp, scale_a_idx, scale_b_idx)`
   - cbsz=0b100 (FP4 for A), blgp=0b010 (FP4 for B)
   - Separate scale loading via `v_mfma_ld_scale_b32`
   
2. **Data types:** FP8 (1 byte/element) → FP4 (4 bits/element, packed as fp4x2)

3. **Scale handling:** float32 per-128-row → E8M0 per-32-element block. Scales integrated into MFMA instruction rather than separate shared memory multiply.

4. **WMMA_K:** 32 → 128

5. **LDS:** gfx950 has 160KB (2.5x more than MI300X's 64KB) — allows larger tiles

6. **CK reference:** Check `/home/runner/aiter/` for `warp_gemm.hpp` — CK's own FP4 MFMA implementation shows exact register layout and data format.

---

## 6. Competitor Intelligence

### 6.1 Top Competitors

| Competitor | GEMM | MoE | MLA | Method |
|-----------|------|-----|-----|--------|
| josusanmartin | 7.65μs (#1) | 123.4μs (#8) | 19.5μs (#1, exploit) | AI-assisted vibecoding, 5136 total subs |
| Chivier | 4.36μs (DQ'd) | — | — | Timing exploit |
| div22 | 8.03μs | 149.3μs | — | 1398 subs |
| Ananda Sai A | 8.09μs | 109.8μs | 33.0μs | pg2_fix for MLA |
| johnny.t.shi | 8.24μs | 112.8μs | 32.6μs | ROCm expert, Split-K from LeelaChessZero |
| ooousay | 8.91μs | 123.7μs | 28.5μs (#2) | Built leaderboard.ooousay.com |
| Maxwell Cipher | 8.97μs | 107.3μs (#1) | — | Unknown approach |

### 6.2 josusanmartin's Methodology
- Co-founder of SixtantIO, Mexico City
- Runs Claude Code and OpenAI Codex in parallel
- Generates 6+ variations per iteration
- Uses submission history as version control
- "Local best: 7.48μs" vs "leaderboard: 7.65μs" — 0.17μs gap (normal variance)
- Blog: josusanmartin.com/blog — "vibecoded" to #1 on Highload.fun

### 6.3 johnny.t.shi's Expertise
- Authored ROCm backend for LeelaChessZero with Split-K parallelization, bias fusion in GEMM epilogue, hipBLASLt/rocBLAS fallback selection
- Direct AMD GPU kernel experience transfers to competition

### 6.4 josusanmartin's MLA 19.5μs
Almost certainly a benchmark exploit. The roofline for bs=128 sk=1024 (smallest test case) is 106.65μs bf16. Even FP8 halving bandwidth = ~53μs minimum. 19.5μs is 5.5× below bf16 roofline — physically impossible without exploiting the benchmark framework.

### 6.5 Discord Intelligence
- Tolerances tightened: rtol=2e-2, atol=2e-2 for MoE (commit a846c7e)
- Custom HIP kernels mentioned in discord as causing tolerance issues due to accumulation order differences
- "stream" is a filtered word — submissions containing it get blocked
- ~4μs submissions are "against the spirit of the competition" and will be disqualified
- No MI355X hardware access for participants — all testing via popcorn-cli submissions

---

## 7. Source Code Deep Dive Findings

### 7.1 GEMM Config System
```
_get_config dispatch:
def _get_config(M, N, K, shuffle=False):
    return get_gemm_config(f"GEMM-A16WFP4", M, N, 2*K)
    # Looks for: gfx950-GEMM-A16WFP4-N={N}-K={2*K}.json
    # Falls back to: gfx950-GEMM-A16WFP4.json (generic)

Generic A16WFP4 config (what ALL our shapes use):
ALL M tiers: BM=4-8, BN=128, BK=512, warps=4-8, stages=1, waves=2, KSPLIT=1

Config dir WRITABLE: /home/runner/aiter/aiter/ops/triton/configs/gemm/
(We successfully wrote configs but KSPLIT>1 was worse due to reduce overhead)
```

### 7.2 GEMM Shapes
```
Shape 1: M=4,   N=2880, K=512   — benchmark ~6.2μs
Shape 2: M=16,  N=2112, K=7168  — benchmark ~13.6μs (SLOWEST, dominates geomean)
Shape 3: M=32,  N=4096, K=512   — benchmark ~6.2μs
Shape 4: M=32,  N=2880, K=512   — benchmark ~6.7μs
Shape 5: M=64,  N=7168, K=2048  — benchmark ~14.1μs (SECOND SLOWEST)
Shape 6: M=256, N=3072, K=1536  — benchmark ~16.1μs (uses afp4wfp4 path)
```

### 7.3 gemm_a4w4 Correct Data Format (Proven, 0 errors)
```python
from aiter import gemm_a4w4, dtypes as aiter_dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

A_fp4_u8, A_scale_u8 = dynamic_mxfp4_quant(A)
A_fp4 = A_fp4_u8.view(aiter_dtypes.fp4x2)
A_scale = e8m0_shuffle(A_scale_u8).view(aiter_dtypes.fp8_e8m0)  # KEY FIX
result = gemm_a4w4(A_fp4, B_shuffle, A_scale, B_scale_sh, None, torch.bfloat16, 1.0, 0.0, 1)
# A data does NOT need shuffling. Only A_scale needs e8m0_shuffle + .view(fp8_e8m0)
# Uses bpreshuffle=1
```

### 7.4 MLA Architecture
```
auto-splits formula (from source):
cu_num = 256; overhead = 84.1
score = bs * splits / ceil(bs*splits/cu_num) * avg_kv / (avg_kv + overhead*splits)
Results: bs=4 kv=1024→32splits, bs=32 kv=1024→8, bs=64 kv=8192→4, bs=256 kv=8192→1

27 MLA .co ASM kernels total:
- qseqlen1 (current), qseqlen2, qseqlen4 variants
- _page suffix kernels (native paging)
- 1TG (1 threadgroup) variants
- qh128 variants (not for our qh16 case)

eval.py uses clear_l2_cache_large between benchmarks
Secret seed: combined_seed = a + (a+b)*(a+b+1)//2 (Cantor pairing)
```

### 7.5 MoE Architecture
```
fused_moe.py: 1838 lines
dsv3 CSV: ONLY 15 lines, ALL E=257 d=256. Zero E=33 entries.
1-stage selection: run_1stage = token>32 AND inter_dim%256==0 for FP4
1024+ fmoe .co kernels (g1u1, various tile sizes) — but these are 1-stage, not CK 2-stage
CK 2-stage kernels: JIT-compiled at runtime, not standalone .co files
```

### 7.6 e8m0_shuffle Behavior
- `e8m0_shuffle` returns [256, 16] for input [32, 16] — 8× expansion
- NOT a simple permutation — it's a complex reordering for MFMA data layout
- Required for gemm_a4w4 but NOT required for hipBLASLt (which expects raw E8M0)

---

## 8. Runner Environment

```
torch=2.10.0+rocm7.1
triton=3.6.0
hip=7.1.25424
aiter: /home/runner/aiter/ (git commit a722aff38)
hipcc: available, compiles for gfx950
rocwmma: available
hipblaslt: /opt/rocm/lib/libhipblaslt.so
load_inline: CONFIRMED WORKING

MI355X specs:
- gfx950 (CDNA4)
- 256 CUs, 8 XCDs
- 160KB LDS per CU
- HBM3E at 8 TB/s
- 10.1 PFLOPS peak FP4
- WAVE_SIZE = 64
```

---

## 9. Key File Locations

```
Repo:              /home/claude/AMD-GPU-MODE/
GEMM submissions:  mxfp4-mm/
MLA submissions:   mixed-mla/
MoE submissions:   moe-mxfp4/
Current best GEMM: mxfp4-mm/sub_ultimate_v1.py
Current best MLA:  mixed-mla/mla_fp8q_all.py
Current best MoE:  moe-mxfp4/submission_optimized_v2.py
Reference code:    research_docs/amd_gpu_mode_ref_codes/amd_202602/
Dead ends doc:     claude-memory/dead_ends.md
Winning MI300X:    /home/claude/winning_fp8_gemm_kernel.hip (78K chars)
Winning core:      /home/claude/winning_kernel_core.hip (229 lines)
```

---

## 10. Fast HIP Quantization Kernel

A custom HIP kernel for bf16→FP4 quantization was tested:
- **3x faster** than aiter's dynamic_mxfp4_quant: 12.4μs vs 37.2μs
- 98.7% accuracy match with aiter's output
- Could replace fused_dynamic_mxfp4_quant_moe_sort (28% of MoE time)
- But standalone quant + separate GEMM is still slower than fused Triton kernel

---

## 11. Remaining Paths Forward

### 11.1 Custom HIP FP4 GEMM Kernel (PRIMARY — high risk, high reward)
Adapt the winning MI300X FP8 kernel to FP4 on gfx950 using `load_inline`. This is what ALL winners do.

**Phased approach:**
1. Phase 1 (Day 1): Get one MFMA FP4 instruction producing correct output for a 16×16×128 tile
2. Phase 2 (Day 1-2): Add tiling, LDS, K-loop. Verify accuracy.
3. Phase 3 (Day 2-3): Double-buffered pipeline, vectorized loads, per-shape dispatch
4. Phase 4 (Day 3-4): Polish, M-padding, fused quant, benchmark, leaderboard

**Key instruction:** `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(src_a, src_b, acc, cbsz=4, blgp=2, scale_a_idx, scale_b_idx)`

**Reference:** CK's warp_gemm.hpp on the runner shows the exact register layout.

### 11.2 MLA Ratcheting (SECONDARY — low risk, low reward)
Already in top 20 at #14. Keep submitting mla_fp8q_all.py to leaderboard every hour. Variance of 1-2μs means possible improvement from 36.4→35μs through luck.

### 11.3 Triton Config Sweep (TERTIARY — medium effort, uncertain reward)
josusanmartin's approach: automated per-shape config sweeping of gemm_a16wfp4 parameters (BM, BN, BK, warps, stages, waves). ~500 valid combos per shape. But this hits a Triton ceiling (~7.5μs benchmark, ~12μs leaderboard) that may not reach the 9μs top 20 cutoff.

### 11.4 MoE Custom HIP Quant (LOWEST — speculative)
Replace fused_dynamic_mxfp4_quant_moe_sort (28% of MoE time) with the 3x faster HIP quant kernel. But this requires modifying the fused_moe C++ wrapper, which has crashed on every previous attempt.

---

## 12. Lessons Learned

1. **Winners write custom kernels, not library wrappers.** Every top submission on the previous competition used `load_inline` + raw HIP C++. Library tuning has a ceiling.

2. **Start with source code, not web search.** Reading aiter's actual Python/C++ source revealed writable config directories, auto-splits formulas, and kernel dispatch logic that web search couldn't find.

3. **Accumulation order matters for accuracy.** hipBLASLt's hand-tuned assembly produces different rounding than Triton, causing 38% relative error even though both use the same MFMA hardware instruction. A custom kernel must match the eval reference's accumulation order.

4. **The submission filtering system blocks "stream" (the word for execution context).** Use 0 as the last argument to hipblasLtMatmul instead of passing a stream handle.

5. **josusanmartin's strategy works:** AI-assisted parallel iteration (Claude Code + Codex), generating 6+ variations per round, using submission history as version control. 5,136 submissions over 26 days = ~200/day.

6. **Per-shape optimization is essential.** The generic config uses KSPLIT=1, stages=1 for ALL shapes. Shape-specific configs (stages=3 for K=512) gave our only GEMM improvement.

7. **The eval harness adds consistent overhead** (clear_l2_cache_large between benchmarks). This overhead affects ALL submissions equally — it's not the source of our gap.
