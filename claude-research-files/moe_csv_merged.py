#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Merge dsv3 CSV WITH E=33 entries, preserving E=257.

Previous AITER_CONFIG_FMOE replaced dsv3 CSV → lost E=257 CK entries.
This reads the EXISTING dsv3 CSV, appends E=33 entries, writes merged.
"""
import os
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_p = False
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

def _patch():
    global _p
    if _p: return
    _p = True
    try:
        dsv3 = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
        with open(dsv3, 'r') as f:
            existing = f.read()
        # dsv3 CSV has extra _tag column — add empty _tag to match
        # Count columns in header to match
        header_line = existing.split('\n')[0]
        n_cols = len(header_line.split(','))
        base = "256,{bs},7168,512,33,9,1,bf16,fp4x2,fp4x2,per_1x32,0,0,{bm},0,0.0,{s1},0,0.0,{s2},0,0.0,0,0.0,0.0"
        e33 = []
        for bs, bm, s1 in [(16, 32, S1_64), (128, 32, S1_256), (512, 32, S1_256)]:
            row = f"256,{bs},7168,512,33,9,1,bf16,fp4x2,fp4x2,per_1x32,0,0,{bm},0,0.0,{s1},0,0.0,{S2_V1},0,0.0,0,0.0,0.0"
            # Pad with empty fields if dsv3 has more columns
            row_cols = len(row.split(','))
            if row_cols < n_cols:
                row += ',' * (n_cols - row_cols)
            e33.append(row)
        merged = existing.rstrip() + "\n" + "\n".join(e33) + "\n"
        path = "/tmp/merged_fmoe.csv"
        with open(path, 'w') as f:
            f.write(merged)
        os.environ["AITER_CONFIG_FMOE"] = path
        fm.cfg_2stages = None
        try:
            orig = fm.get_2stage_cfgs.__wrapped__
            fm.get_2stage_cfgs = functools.lru_cache(maxsize=2048)(orig)
        except Exception:
            pass
    except Exception as e:
        import sys
        print(f"[MOE] CSV merge failed: {e}", flush=True, file=sys.stderr)
    fm.use_nt = lambda t, k, e: False

def custom_kernel(data: input_t) -> output_t:
    _patch()
    (hs, guw, dw, guws, dws, guw_sh, dw_sh, guws_sh, dws_sh, tw, ti, cfg) = data
    return fused_moe(
        hs, guw_sh, dw_sh, tw, ti,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=guws_sh, w2_scale=dws_sh,
        a1_scale=None, a2_scale=None,
        hidden_pad=cfg["d_hidden_pad"]-cfg["d_hidden"],
        intermediate_pad=cfg["d_expert_pad"]-cfg["d_expert"],
    )
