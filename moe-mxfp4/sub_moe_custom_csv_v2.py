#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE: Custom AITER_CONFIG_FMOE CSV with tuned kernel names per shape.
Injects optimal stage1/stage2 kernel combos for E=257 shapes via CSV override.
Uses AITER_CONFIG_FMOE env var to point to a custom CSV that gets merged.
"""
import os
import tempfile

# Build custom CSV with entries for our exact benchmark shapes
# CSV columns: token,model_dim,inter_dim,expert,topk,kernelName1,kernelName2,block_m,ksplit,us,tflops,bw,cu_num,dtype,q_dtype_a,q_dtype_w,q_type
# Key: specify pre-compiled kernel names (no JIT needed), ksplit=0

S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

CSV_HEADER = "token,model_dim,inter_dim,expert,topk,kernelName1,kernelName2,block_m,ksplit,us,tflops,bw,cu_num,dtype,q_dtype_a,q_dtype_w,q_type\n"

# E=257 d=256: try different block_m values and kernel combos
# bs=16: est_m = 16*9/257 ≈ 0.56 → very sparse
# bs=128: est_m = 128*9/257 ≈ 4.5
# bs=512: est_m = 512*9/257 ≈ 17.9
CSV_ROWS = [
    # E=257 bs=16: use smaller tile stage1 (64x32), block_m=32
    f"16,7168,256,257,9,{S1_64},{S2_V1},32,0,100,0,0,256,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32\n",
    # E=257 bs=128: use larger tile stage1 (256x32), block_m=32
    f"128,7168,256,257,9,{S1_256},{S2_V1},32,0,100,0,0,256,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32\n",
    # E=257 bs=512: use larger tile, block_m=64
    f"512,7168,256,257,9,{S1_256},{S2_V1},64,0,100,0,0,256,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32\n",
    # E=33 d=512 bs=16: stage1 64, block_m=32
    f"16,7168,512,33,9,{S1_64},{S2_V1},32,0,100,0,0,256,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32\n",
    # E=33 d=512 bs=128: stage1 256, block_m=64
    f"128,7168,512,33,9,{S1_256},{S2_V1},64,0,100,0,0,256,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32\n",
    # E=33 d=512 bs=512: stage1 256, block_m=64
    f"512,7168,512,33,9,{S1_256},{S2_V1},64,0,100,0,0,256,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32\n",
]

# Write CSV to /tmp (writable on runner)
csv_path = "/tmp/custom_fmoe.csv"
with open(csv_path, "w") as f:
    f.write(CSV_HEADER)
    for row in CSV_ROWS:
        f.write(row)

# Set env var BEFORE importing aiter
os.environ["AITER_CONFIG_FMOE"] = csv_path

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

    # Print what configs were loaded
    try:
        print(f"[CSV] AITER_CONFIG_FMOE={os.environ.get('AITER_CONFIG_FMOE', 'NOT SET')}")
        if hasattr(fm, 'cfg_2stages') and fm.cfg_2stages is not None:
            print(f"[CSV] cfg_2stages loaded: {len(fm.cfg_2stages)} entries")
    except:
        pass

    # use_nt=False for all
    fm.use_nt = lambda token, topk, expert: False


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
