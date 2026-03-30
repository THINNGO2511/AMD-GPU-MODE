"""MoE: Custom tuned CSV with proven kernel names for ALL shapes"""
import os
import tempfile

# Step 1: Create custom CSV with tuned configs for E=33 MXFP4
# The proven kernel names come from dsv3_fp4_tuned_fmoe.csv (E=257 entries)
# Adapted for E=33 shapes

CSV_HEADER = "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,preshuffle,hidden_pad,intermediate_pad,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw"

# CK kernel names (proven working for FP4X2)
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x64 = "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x128 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"

S2_64 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_none_FP4X2_FP4X2_B16"
S2_256 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_none_FP4X2_FP4X2_B16"

# E=33 entries — adapted from working E=257 configs
# est_m = token * topk / expert = token * 9 / 33
# bs=16: est_m=4.4 → block_m=32
# bs=128: est_m=35 → block_m=64
# bs=512: est_m=140 → block_m=128

ROWS = []
# Format: cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_a,q_w,q_type,g1u1,doweight,preshuffle,hpad,ipad,block_m,ksplit,us1,s1name,err1,us2,s2name,err2,us,run1s,tflops,bw
ACT = "ActivationType.Silu"
DT = "torch.bfloat16"
QA = "torch.float4_e2m1fn_x2"
QW = "torch.float4_e2m1fn_x2"
QT = "QuantType.per_1x32"

for token in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    est_m = token * 9 // 33
    if est_m < 10:
        bm, s1, s2 = 32, S1_64, S2_64
    elif est_m < 50:
        bm, s1, s2 = 64, S1_256, S2_256
    elif est_m < 200:
        bm, s1, s2 = 128, S1_256x64, S2_256
    else:
        bm, s1, s2 = 128, S1_256x128, S2_256
    
    for inter_dim in [512, 2048]:
        ipad = ((inter_dim + 255) // 256) * 256
        hpad = ((7168 + 255) // 256) * 256
        ROWS.append(f"256,{token},7168,{inter_dim},33,9,{ACT},{DT},{QA},{QW},{QT},False,False,True,{hpad},{ipad},{bm},0,0,{s1},0,0,{s2},0,0,False,0,0")

# Also add E=257 entries (copy proven configs)
for token in [16, 32, 64, 128, 256, 512, 1024]:
    est_m = token * 9 // 257
    if est_m < 10:
        bm, s1, s2 = 32, S1_64, S2_64
    elif est_m < 50:
        bm, s1, s2 = 64, S1_256, S2_256
    else:
        bm, s1, s2 = 128, S1_256x64, S2_256
    
    ipad = ((256 + 255) // 256) * 256
    hpad = ((7168 + 255) // 256) * 256
    ROWS.append(f"256,{token},7168,256,257,9,{ACT},{DT},{QA},{QW},{QT},False,False,True,{hpad},{ipad},{bm},0,0,{s1},0,0,{s2},0,0,False,0,0")

csv_content = CSV_HEADER + "\n" + "\n".join(ROWS) + "\n"

# Write CSV to /tmp
csv_path = "/tmp/custom_tuned_fmoe_v2.csv"
with open(csv_path, "w") as f:
    f.write(csv_content)

# Step 2: Set env vars BEFORE importing aiter
os.environ["AITER_CONFIG_FMOE"] = csv_path
os.environ["CU_NUM"] = "256"
os.environ["AITER_USE_NT"] = "0"

import torch
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType

fm.use_nt = lambda t, k, e: False

from typing import Dict, Tuple
input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

def custom_kernel(data: input_t) -> output_t:
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
