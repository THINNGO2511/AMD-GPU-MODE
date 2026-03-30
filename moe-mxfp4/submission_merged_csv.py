#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Merged CSV: Copy existing merged CSV + append E=33 entries.
This preserves E=257 tuned configs from DSv3 while adding our
proven E=33 kernel selections. Also enables OPUS + use_nt=False.
"""
import os
import shutil

S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

_common = "ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0"

# New E=33 entries to append
_e33_entries = [
    # E=33 d=512 bs=16: est_m=4 → block_m=32, S1_64
    f"256,16,7168,512,33,9,{_common},32,0,28,{S1_64},0.0,19,{S2_V1},0.0,47,0,0,0,",
    # E=33 d=512 bs=128: est_m=34 → block_m=32, S1_64
    f"256,128,7168,512,33,9,{_common},32,0,50,{S1_64},0.0,40,{S2_V1},0.0,90,0,0,0,",
    # E=33 d=512 bs=512: est_m=139 → block_m=64, S1_256
    f"256,512,7168,512,33,9,{_common},64,0,80,{S1_256},0.0,60,{S2_V1},0.0,140,0,0,0,",
    # E=33 d=2048 bs=512: est_m=139 → block_m=128, S1_256
    f"256,512,7168,2048,33,9,{_common},128,0,150,{S1_256},0.0,100,{S2_V1},0.0,250,0,0,0,",
]

# Build merged CSV: existing DSv3 configs + our E=33 entries
_dst = "/tmp/hackathon_merged_fmoe.csv"
_srcs = [
    "/home/runner/aiter/aiter/configs/tuned_fmoe.csv",
    "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
]

try:
    header = None
    all_lines = []
    for src in _srcs:
        if os.path.exists(src):
            with open(src) as f:
                lines = f.readlines()
            if header is None and lines:
                header = lines[0]
                all_lines.append(header)
            # Skip header, skip flydsl_fallback tagged entries
            for line in lines[1:]:
                if '_tag' in (header or '') and 'flydsl_fallback' in line:
                    continue
                all_lines.append(line if line.endswith('\n') else line + '\n')
    # Append our E=33 entries
    for entry in _e33_entries:
        all_lines.append(entry + '\n')
    with open(_dst, 'w') as f:
        f.writelines(all_lines)
    os.environ["AITER_CONFIG_FMOE"] = _dst
except Exception as e:
    import sys
    print(f"[CSV merge error: {e}]", file=sys.stderr)

import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    fm.use_nt = lambda token, topk, expert: False

    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True

    # block_m override for consistency
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm


def custom_kernel(data: input_t) -> output_t:
    _patch()
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
