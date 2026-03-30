#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
MoE Key Probe — Logs EXACT get_2stage_cfgs lookup keys & results for each benchmark shape.
Monkey-patches get_2stage_cfgs (bypassing @lru_cache) to see what keys are generated,
whether the CSV has a match, and what MOEMetadata is returned.
"""

import os
os.environ['CU_NUM'] = '256'
os.environ['AITER_USE_NT'] = '0'

import functools
import sys
import torch
from typing import Dict, Tuple

input_t = Tuple[torch.Tensor, ...]
output_t = torch.Tensor

# === MONKEY-PATCH get_2stage_cfgs ===
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType

# get_2stage_cfgs is @functools.lru_cache decorated.
# We need to get the underlying unwrapped function, replace the module-level
# reference, and re-wrap with our logging + fresh lru_cache.

# Step 1: Get the original unwrapped function
_orig_fn = fm.get_2stage_cfgs.__wrapped__

# Step 2: Also grab the original cfg_2stages dict loader for inspection
# We'll read the CSV ourselves to check what keys exist

_call_count = 0

@functools.lru_cache(maxsize=2048)
def _logged_get_2stage_cfgs(
    token, model_dim, inter_dim, expert, topk,
    dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
    activation, doweight_stage1, hidden_pad, intermediate_pad,
    is_shuffled=True,
):
    global _call_count
    _call_count += 1

    # === LOG THE KEY ===
    from aiter.jit.utils.chip_info import get_cu_num
    cu_num = get_cu_num()

    # This is the exact key tuple used for CSV dict lookup
    csv_key = (
        cu_num,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        str(activation),
        str(dtype),
        str(q_dtype_a),
        str(q_dtype_w),
        str(q_type),
        use_g1u1,
        doweight_stage1,
    )

    print(f"\n{'='*100}", flush=True)
    print(f"PROBE CALL #{_call_count}", flush=True)
    print(f"  FUNCTION ARGS:", flush=True)
    print(f"    token={token}, model_dim={model_dim}, inter_dim={inter_dim}", flush=True)
    print(f"    expert={expert}, topk={topk}", flush=True)
    print(f"    dtype={dtype}, q_dtype_a={q_dtype_a}, q_dtype_w={q_dtype_w}", flush=True)
    print(f"    q_type={q_type}, use_g1u1={use_g1u1}", flush=True)
    print(f"    activation={activation}, doweight_stage1={doweight_stage1}", flush=True)
    print(f"    hidden_pad={hidden_pad}, intermediate_pad={intermediate_pad}", flush=True)
    print(f"    is_shuffled={is_shuffled}", flush=True)
    print(f"  CSV LOOKUP KEY (13-tuple):", flush=True)
    print(f"    {csv_key}", flush=True)
    print(f"  KEY ELEMENTS:", flush=True)
    for i, (name, val) in enumerate(zip(
        ['cu_num','token','model_dim','inter_dim','expert','topk',
         'act_type','dtype','q_dtype_a','q_dtype_w','q_type','use_g1u1','doweight_stage1'],
        csv_key
    )):
        print(f"    [{i:2d}] {name:20s} = {val!r}  (type={type(val).__name__})", flush=True)

    # === Check CSV directly ===
    try:
        from aiter.jit.core import AITER_CONFIGS
        tune_file = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
        print(f"  TUNE FILE: {tune_file}", flush=True)

        if os.path.exists(tune_file):
            import pandas as pd
            df = pd.read_csv(tune_file)
            print(f"  CSV shape: {df.shape}, columns: {list(df.columns)}", flush=True)

            # Show rows that match model_dim and inter_dim
            mask = (df['model_dim'] == model_dim) & (df['inter_dim'] == inter_dim)
            matching = df[mask]
            if len(matching) > 0:
                print(f"  CSV rows matching model_dim={model_dim}, inter_dim={inter_dim}: {len(matching)}", flush=True)
                for idx, row in matching.iterrows():
                    print(f"    Row {idx}: token={row.get('token','?')} expert={row.get('expert','?')} "
                          f"topk={row.get('topk','?')} cu_num={row.get('cu_num','?')} "
                          f"act={row.get('act_type','?')} dtype={row.get('dtype','?')} "
                          f"q_a={row.get('q_dtype_a','?')} q_w={row.get('q_dtype_w','?')} "
                          f"q_type={row.get('q_type','?')} g1u1={row.get('use_g1u1','?')} "
                          f"dw={row.get('doweight_stage1','?')} "
                          f"bm={row.get('block_m','?')} ks={row.get('ksplit','?')} "
                          f"kn1={row.get('kernelName1','?')} kn2={row.get('kernelName2','?')}", flush=True)
            else:
                print(f"  CSV: NO rows match model_dim={model_dim}, inter_dim={inter_dim}", flush=True)

            # Also check the dict form (this is what get_2stage_cfgs does)
            _INDEX_COLS = [
                "cu_num","token","model_dim","inter_dim","expert","topk",
                "act_type","dtype","q_dtype_a","q_dtype_w","q_type","use_g1u1","doweight_stage1",
            ]
            if "_tag" in df.columns:
                df_clean = df[df["_tag"].fillna("") == ""]
            else:
                df_clean = df
            cfg_dict = df_clean.set_index(_INDEX_COLS).to_dict("index")
            found = cfg_dict.get(csv_key)
            print(f"  DICT LOOKUP RESULT: {'FOUND' if found else 'NOT FOUND (None)'}", flush=True)
            if found:
                print(f"    {found}", flush=True)

            # Show what keys ARE in the dict (first 10)
            all_keys = list(cfg_dict.keys())
            print(f"  TOTAL DICT KEYS: {len(all_keys)}", flush=True)
            for k in all_keys[:10]:
                print(f"    {k}", flush=True)
            if len(all_keys) > 10:
                print(f"    ... ({len(all_keys) - 10} more)", flush=True)
        else:
            print(f"  TUNE FILE NOT FOUND!", flush=True)
    except Exception as e:
        print(f"  CSV inspection error: {e}", flush=True)

    # === Now call the REAL function ===
    result = _orig_fn(
        token, model_dim, inter_dim, expert, topk,
        dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
        activation, doweight_stage1, hidden_pad, intermediate_pad,
        is_shuffled,
    )

    # === LOG THE RESULT ===
    print(f"  RESULT MOEMetadata:", flush=True)
    print(f"    block_m   = {result.block_m}", flush=True)
    print(f"    ksplit    = {result.ksplit}", flush=True)
    print(f"    run_1stage= {result.run_1stage}", flush=True)
    print(f"    has_bias  = {getattr(result, 'has_bias', 'N/A')}", flush=True)
    print(f"    use_nt    = {getattr(result, 'use_non_temporal_load', 'N/A')}", flush=True)
    print(f"    stage1    = {result.stage1}", flush=True)
    print(f"    stage2    = {result.stage2}", flush=True)

    # Try to extract kernelName from partial functions
    if result.stage1 and hasattr(result.stage1, 'keywords'):
        kn1 = result.stage1.keywords.get('kernelName', '')
        print(f"    stage1.kernelName = {kn1!r}", flush=True)
    if result.stage2 and hasattr(result.stage2, 'keywords'):
        kn2 = result.stage2.keywords.get('kernelName', '')
        print(f"    stage2.kernelName = {kn2!r}", flush=True)

    print(f"{'='*100}\n", flush=True)
    return result

# Step 3: Replace module-level reference.
# Also need to clear the original lru_cache since it might have been warmed.
fm.get_2stage_cfgs.cache_clear()
fm.get_2stage_cfgs = _logged_get_2stage_cfgs

# Also patch the global cfg_2stages to None so it re-reads the CSV
fm.cfg_2stages = None

print("=" * 100, flush=True)
print("MoE KEY PROBE — INSTALLED", flush=True)
print("  get_2stage_cfgs monkey-patched with logging", flush=True)
print("  Will print exact CSV lookup keys and results", flush=True)
print("=" * 100, flush=True)

# === Also log environment and helper function outputs ===
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
print(f"\nENVIRONMENT:", flush=True)
print(f"  CU_NUM env = {os.environ.get('CU_NUM', 'not set')}", flush=True)
print(f"  AITER_USE_NT env = {os.environ.get('AITER_USE_NT', 'not set')}", flush=True)
print(f"  AITER_BYPASS_TUNE_CONFIG = {os.environ.get('AITER_BYPASS_TUNE_CONFIG', 'not set')}", flush=True)


def custom_kernel(data):
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data

    from aiter.fused_moe import fused_moe

    # Log shape info before fused_moe call
    M, topk_k = topk_ids.shape
    E_local = w1_qw.shape[0]
    d_model = w2_qw.shape[1]
    d_inter = w2_qw.shape[2]

    # Check is_shuffled on w1
    is_shuf = getattr(w1_qw, 'is_shuffled', 'NOT SET')
    print(f"\n>>> custom_kernel called: M={M} topk={topk_k} E={E_local} "
          f"d_model={d_model} d_inter={d_inter} w1_qw.dtype={w1_qw.dtype} "
          f"is_shuffled={is_shuf}", flush=True)

    # get_padded_M(M) is what gets passed as 'token'
    padded_M = fm.get_padded_M(M)
    print(f"    get_padded_M({M}) = {padded_M}", flush=True)

    # What q_dtype_a will be for per_1x32 + Silu + gfx950:
    # q_dtype_a = dtypes.fp4x2 (since activation != Swiglu)
    from aiter import dtypes
    print(f"    Expected q_dtype_a for per_1x32+Silu = {dtypes.fp4x2}", flush=True)
    print(f"    Expected q_dtype_w = {w1_qw.dtype}", flush=True)

    # Run fused_moe — this will trigger our patched get_2stage_cfgs
    result = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                       expert_mask=None, activation=ActivationType.Silu,
                       quant_type=QuantType.per_1x32, doweight_stage1=False,
                       w1_scale=w1_qs, w2_scale=w2_qs)

    return result
