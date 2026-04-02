# Pickup Prompt — Session 20

Continue GPU kernel optimization for AMD x GPU MODE hackathon. I'm noobmaster69_og. Deadline April 6, 2026 11:59 PM PST.

## CURRENT SCORES (Apr 1 end of session 19)
- **GEMM**: 15.7μs rank ~#160/320 → need <8.2μs for top 10
- **MLA**: 42.16μs rank #17/185 → need <34.9μs for top 10
- **MoE**: 169.13μs rank #64/225 → need <129.4μs for top 10

## SESSION 19 KEY FINDINGS

### MoE Internal Flow (PROBE-CONFIRMED)
Per fused_moe call:
- `moe_sorting` / `_moe_sorting_impl`: called **ONCE** (not twice!)
- `fused_dynamic_mxfp4_quant_moe_sort`: called **TWICE** (fused Triton kernel doing quant + rearrange, once per GEMM stage)
- Sort caching is COMPLETELY USELESS — the expensive 2x call is the Triton quant kernel, not the Python sort

### CSV Override WORKS (format confirmed)
- AITER_CONFIG_FMOE picks up custom CSV correctly
- E=33 entries found and used by runtime
- Column format: `cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw`
- Values must use EXACT strings: `ActivationType.Silu`, `torch.bfloat16`, `torch.float4_e2m1fn_x2`, `QuantType.per_1x32`, `use_g1u1=1`, `doweight_stage1=0`
- E=33 d=2048 with 256x128x128x128 + 64x128x128x128_v3 kernels → GPU CRASH. Only smaller tiles (64x32, 256x32) are safe for E=33.

### Block_m Mismatch (potential fix)
- E=33 d=512 bs=512: DEFAULT block_m=128, our override gives 64, CK injection hardcodes 32
- We might be HURTING ourselves on this shape — need to test default vs our overrides

### Dead Ends Confirmed (Session 19)
- MoE sort caching (moe_sorting called once, also crashed with 1.65M mismatches)
- FlyDSL env vars (AITER_ENFORCE_DSL, AITER_USE_FLYDSL_MOE — no effect on FP4)
- GEMM TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 (±2%, no real improvement)
- GEMM AMDGCN_USE_BUFFER_OPS=0 (forces Triton recompile → timeout)
- GEMM AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1 (same timeout)
- MLA pg4 for kv=8192 (timed out)
- CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3 (not submitted, FlyDSL dead)

### Timed Out (need retry)
- moe_minimal.py (just use_nt=False, no overrides — tests if our patches hurt)
- mla_fixed_scale.py (fixed fp8 Q scale, saves 1 kernel launch)
- gemm_exp_deep_probe.py (discovers ASM/deepgemm API paths)

## WHAT TO DO NEXT

### Priority 1: MoE — Fix block_m mismatch
Our CK injection hardcodes block_m=32 in MOEMetadata. The default for E=33 d=512 bs=512 is block_m=128. Test:
1. Remove CK injection for E=33 entirely (let defaults handle it)
2. Or fix MOEMetadata to use get_block_size_M() result instead of hardcoded 32

### Priority 2: MoE — Retry minimal test
Submit moe_minimal.py again (just use_nt=False). If it's faster than 169μs, our overrides are the problem.

### Priority 3: MLA — Fixed-scale quant retry
The fixed-scale fp8 Q quant (single kernel, hardcoded scale=16.0/FP8_MAX) timed out. Retry.

### Priority 4: MLA — Keep ratcheting pg2 leaderboard
67% pass rate per attempt. Need luck.

### Priority 5: GEMM — Deep probe retry
The ASM/deepgemm probe timed out. Need to discover if there are alternative kernel paths.

### Priority 6: MoE — CSV override for E=33 with SAFE kernels
CSV format works! But only use 64x32 or 256x32 kernels for E=33 (128x128 crashes).

## Claude Research Files
10 experiment files at `claude-research-files/`. Status:
- DEAD: moe_sort_cache.py, mla_pg2_all.py, gemm_stages3_k512.py, gemm_afp4_k2048.py
- TIMED OUT: moe_minimal.py, mla_fixed_scale.py, gemm_exp_deep_probe.py (retry these)
- UNTESTED: moe_custom_csv.py (replaced by our sub_moe_csv_fixed.py), mla_aggressive.py
- CRASHED: sub_moe_csv_fixed.py (128x128 kernels on E=33 d=2048)

## COMMANDS
```bash
popcorn submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark moe-mxfp4/FILE.py --no-tui
popcorn submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/sub_pg2_bf16q.py --no-tui
popcorn submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark mxfp4-mm/FILE.py --no-tui
```
