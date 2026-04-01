#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Generate custom tuned_fmoe.csv targeting d=2048.
Previous attempts crashed due to wrong column format.
This uses the probe results to build a correct CSV.

Key insight: we need cu_num=256 entries for E=33 d=2048.
The existing CSV only has cu_num=80 entries for E=33.

Strategy: Write a CSV that ONLY has our benchmark shapes with
aggressive configs, set AITER_CONFIG_FMOE to point to it.
If the CSV doesn't match, fused_moe falls back to defaults.
So we only add entries we're confident about.
"""
import os
import tempfile
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# CSV column header (from probe / PICKUP_PROMPT)
CSV_HEADER = "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw"

# Known good kernel names from our CK injection
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

def _build_csv():
    """Build a custom config CSV with entries for our benchmark shapes."""
    rows = [CSV_HEADER]

    # Format: cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,
    #         q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,
    #         us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw

    # Common fields for FP4: act_type=silu, dtype=bf16, q_dtype=fp4x2, q_type=per_1x32
    common = "silu,bf16,fp4x2,fp4x2,per_1x32,0,0"

    # E=33 d=512 (inter_dim=512): bs=16 (token=16*9=144), bs=128 (1152), bs=512 (4608)
    # For E=33, est_m = token*topk/expert
    # bs=16: est_m = 16*9/33 = 4.4 -> block_m=32
    # bs=128: est_m = 128*9/33 = 34.9 -> block_m=32
    # bs=512: est_m = 512*9/33 = 139.6 -> block_m=64
    for token, bm in [(16, 32), (128, 32), (512, 64)]:
        rows.append(f"256,{token},7168,512,33,9,{common},{bm},0,0.0,{S1_64},0,0.0,{S2_V1},0,0.0,0,0.0,0.0")

    # E=33 d=2048: bs=512 (token=512*9=4608)
    # est_m = 512*9/33 = 139.6 -> try block_m=64 AND block_m=128
    # Try the bigger stage1 kernel for d=2048
    rows.append(f"256,512,7168,2048,33,9,{common},64,0,0.0,{S1_256},0,0.0,{S2_V1},0,0.0,0,0.0,0.0")

    # E=257 d=256: bs=16 (token=16*9=144), bs=128 (1152), bs=512 (4608)
    # est_m = token*9/257
    # bs=16: est_m=5 -> block_m=32
    # bs=128: est_m=40 -> block_m=32
    # bs=512: est_m=161 -> block_m=64
    for token, bm in [(16, 32), (128, 32), (512, 64)]:
        rows.append(f"256,{token},7168,256,257,9,{common},{bm},0,0.0,,0,0.0,,0,0.0,0,0.0,0.0")

    return "\n".join(rows) + "\n"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Write custom CSV
    csv_content = _build_csv()
    csv_path = "/tmp/custom_fmoe.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    os.environ["AITER_CONFIG_FMOE"] = csv_path

    # use_nt=False
    fm.use_nt = lambda token, topk, expert: False

    # Clear cached configs so our CSV gets picked up
    fm.cfg_2stages = None
    # Re-wrap get_2stage_cfgs to clear LRU cache
    try:
        orig = fm.get_2stage_cfgs.__wrapped__
        fm.get_2stage_cfgs = functools.lru_cache(maxsize=2048)(orig)
    except Exception:
        pass


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
