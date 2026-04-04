# AMD x GPU MODE Hackathon — Session Briefing

## Competition
- 3 kernel optimization problems on AMD MI355X (gfx950, CDNA4, 256 CUs)
- Deadline: April 6, 2026 11:59 PM PST (~5 days)
- Top 10 aggregate score advances to Finals ($1.1M prize pool)
- Scoring: geometric mean per problem, only top 20 per problem get aggregate points
- Submit via: `popcorn-cli submit --gpu MI355X --leaderboard <name> --mode <mode> <file> --no-tui`
- Leaderboards: `amd-mxfp4-mm` (GEMM), `amd-moe-mxfp4` (MoE), `amd-mixed-mla` (MLA)
- User: noobmaster69_og
- Repo: /home/claude/AMD-GPU-MODE/

## Current Standings
| Problem | Score | Rank | Top 20 Cutoff | File |
|---------|-------|------|---------------|------|
| MLA | 36.4μs | ~#14 | ~35μs | mixed-mla/mla_fp8q_all.py |
| GEMM | 15.7μs | ~#160 | ~9μs | mxfp4-mm/sub_ultimate_v1.py |
| MoE | 163μs | ~#65 | ~143μs | moe-mxfp4/submission_optimized_v2.py |

MLA is IN top 20. GEMM and MoE are NOT — meaning we get ZERO aggregate points from them.

## CRITICAL DISCOVERY: Top competitors write custom HIP C++ kernels

We analyzed 40K submissions from the previous AMD competition (HuggingFace GPUMODE/kernelbot-data).
**EVERY winning submission uses `torch.utils.cpp_extension.load_inline` to compile custom HIP C++ kernels at runtime.** They bypass aiter/Triton entirely with hand-written GPU code using MFMA instructions.

This explains why we couldn't close the gap — we were tuning library parameters while competitors wrote custom kernels.

## IMMEDIATE TASK: Test if load_inline works on MI355X runner

Submit this file as TEST (already in claude-research-files/):
```
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_probe_loadinline.py --no-tui
```

This probes whether the runner supports:
- torch.utils.cpp_extension.load_inline compilation
- hipcc compiler availability
- MFMA intrinsic compilation for gfx950
- Available libraries (hipBLAS, hipBLASLt, rocBLAS)

READ ALL [PROBE] lines from stdout. This determines our entire remaining strategy.

Also explore the compilation environment directly:
```bash
which hipcc 2>/dev/null && hipcc --version
which amdclang++ 2>/dev/null
ls /opt/rocm/lib/libhipblas* 2>/dev/null
ls /opt/rocm/lib/libhipblaslt* 2>/dev/null
python3 -c "from torch.utils.cpp_extension import load_inline; print('importable')" 2>&1
python3 -c "import torch; print(torch.cuda.get_device_properties(0))" 2>&1
```

## Also keep MLA ratcheting:
```
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui
```

## What We Know Works
- MLA: fp8 Q quantization for ALL shapes + pg1 for kv≤1024 + pg8 for kv≥8192 = 36.4μs
- GEMM: gemm_a16wfp4 Triton kernel with per-shape configs (stages=3 for K=512) = 15.7μs
- MoE: fused_moe with use_nt=False + CK injection for E≤64 d<2048 = 163μs
- GEMM: gemm_a4w4 ASM kernel works with correct format (e8m0_shuffle on A_scale) but 3-launch overhead makes it slower

## Confirmed Dead Ends (DO NOT RETRY)
**MLA:** pg2 for kv=1024 (5x fail), MXFP4 KV (ASM rejects dim=288), qseqlen2 (GPU crash), auto_splits (secret seed fail), splits=1 for small bs (slower)
**GEMM:** gemm_a4w4 ASM (correct but slower due to quant overhead), preshuffle Triton (fp4x2 incompatible), KSPLIT>1 (reduce overhead kills it), afp4wfp4 for all shapes (slower), config=None lib defaults (worse for K=7168), env vars for ASM dispatch (no effect), writing JSON configs (KSPLIT=2/4 both worse)
**MoE:** ALL monkey-patches exhausted — CK injection d=2048 (crashes/slower), CSV override (breaks E=257), sort policy (wrong API), block_m=16 (assertion), online_tune (no effect), 1stage d=2048 (slower), splitk (crash), AITER_BYPASS_TUNE_CONFIG (doesn't exist)

## Runner Details
- torch=2.10.0+rocm7.1, triton=3.6.0, hip=7.1.25424
- aiter installed at /home/runner/aiter/
- Config dir WRITABLE: /home/runner/aiter/aiter/ops/triton/configs/gemm/
- 27 MLA .co ASM kernels, 35 f4gemm .co kernels, 1024+ fmoe .co kernels
- eval.py uses clear_l2_cache_large between benchmarks

## If load_inline WORKS — Next Steps
1. Write a custom HIP MXFP4 GEMM kernel using MFMA scale instructions (V_MFMA_SCALE_F32_16X16X128_F8F6F4)
2. Write a custom MoE kernel bypassing fused_moe entirely
3. Or call hipBLASLt directly for optimized GEMM
4. Iterate with AI-assisted code generation (josusanmartin's proven approach)

## If load_inline FAILS — Next Steps  
1. Focus on MLA ratcheting (already in top 20, variance can improve score)
2. Accept library ceiling on GEMM and MoE
3. Document learnings for next competition

CRITICAL: Report ALL stdout from the probe. Every [PROBE] line matters.
