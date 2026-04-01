# Pickup Prompt — Session 19

Continue GPU kernel optimization for AMD x GPU MODE hackathon. I'm noobmaster69_og. Deadline April 6, 2026 11:59 PM PST.

## CURRENT SCORES (Mar 31 end of session 18)
- **GEMM**: 15.7μs rank ~#160/320 → need <8.2μs for top 10
- **MLA**: 42.16μs rank #17/185 → need <34.9μs for top 10
- **MoE**: 169.13μs rank #64/225 → need <129.4μs for top 10

## MoE CSV Format (PROBED — use this for AITER_CONFIG_FMOE)
**Actual columns** (from tuned_fmoe.csv, 1421 rows):
```
cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype,
q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1, block_m, ksplit,
us1, kernelName1, err1, us2, kernelName2, err2, us, run_1stage, tflops, bw
```
- dsv3 CSV has extra `_tag` column (14 rows, all E=257)
- tuned_fmoe.csv: 25 E=257 entries, 26 E=33 entries (cu_num=80, wrong GPU)
- dsv3 CSV: 14 E=257 entries (cu_num=256)
- **STILL NEED**: actual sample rows to see value formats (dtypes print was empty)
- Next session: re-probe with `df.head(3).to_csv()` to get raw CSV row format

## WHAT TO DO NEXT (from Claude Research intel)

### Priority 1: MoE Sort Caching (UNTRIED, highest impact)
The research confirmed `moe_sorting()` returns depend ONLY on routing (topk_ids). Sort results can be cached across GEMM stages. Monkey-patch pattern:
```python
_original = aiter_fmoe.moe_sorting
_cache = {}
def cached_moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, cache_key=None):
    if cache_key and cache_key in _cache:
        s_tok, s_wt, s_exp, n_valid = _cache[cache_key]
        out = torch.empty((topk_ids.shape[0], model_dim), dtype=dtype, device=topk_ids.device)
        return s_tok, s_wt, s_exp, n_valid, out
    result = _original(topk_ids, topk_weights, E, model_dim, dtype)
    if cache_key:
        _cache[cache_key] = result[:4]
    return result
```
NOTE: fused_moe C++ wrapper calls moe_sorting internally. Need to monkey-patch at the right level.

### Priority 2: MoE AITER_CONFIG_FMOE CSV (fix format)
Previous attempt crashed — used wrong column value format. The probe will reveal:
- Exact column names and dtypes
- What cu_num value the runner uses
- What q_type format (numeric vs string)
- Existing E=257 and E=33 entries
Use this to build correct CSV with different kernel combos for E=257.

### Priority 3: GEMM TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1
Confirmed DISABLED by default on gfx950 but EXISTS. Claude Research says it controls in-thread transpose optimization. Quick env var test.

### Priority 4: MLA pg2 Ratcheting
Keep submitting `sub_pg2_bf16q.py` to leaderboard (67% pass rate). Each attempt has independent secret seeds.

## CONFIRMED DEAD ENDS (Session 18, don't retry)
- GEMM preshuffle: shuffle format matches but reshape data layout wrong, need CK C++ source
- GEMM gemm_a8wfp4: 94% mismatch, 117μs
- GEMM L2 prefetch HIP: +5μs overhead
- GEMM KSPLIT=14: worse
- GEMM XCD remap: accuracy broken
- GEMM Triton PINGPONG/ASYNC env vars: auto-enabled, explicit setting worse
- GEMM OPTIMIZE_EPILOGUE: doesn't exist as env var
- MLA bf16 KV: same 4% mismatch as fp8
- MLA KV subsample: 71K mismatches
- MLA qseqlen2: fails secret seeds
- MLA HSA_HIGH_PRECISION_MODE: zero effect
- MoE ksplit=2: JIT timeout (kFFN_gemm1_split_k template >10 min)
- MoE block_m=16: assertion crash on test shapes
- MoE AITER_USE_OPUS_MOE_SORTING=0: internal error
- CK ASM gemm_a4w4: API changed, quant=44μs

## COMPREHENSIVE RESEARCH BRIEF
See `RESEARCH_BRIEF.md` for complete details on all approaches, timings, competitor intel, and open questions.

## WHAT WORKS (proven, current best)
- **GEMM**: `mxfp4-mm/sub_ultimate_v1.py` — 15.7μs (torch.take + BM=16 + .cg)
- **MLA**: `mixed-mla/sub_pg2_bf16q.py` — 42.16μs (pg2+bf16Q kv≤1024, pg8+fp8Q kv≥8192)
- **MoE**: `moe-mxfp4/submission_optimized_v2.py` — 169μs (CK injection E≤64, use_nt=False)

## COMMANDS
```bash
# MLA ratchet (67% pass rate dice roll)
popcorn submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/sub_pg2_bf16q.py --no-tui

# MoE test new approach
popcorn submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark moe-mxfp4/NEW_FILE.py --no-tui

# GEMM test new approach
popcorn submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark mxfp4-mm/NEW_FILE.py --no-tui
```
