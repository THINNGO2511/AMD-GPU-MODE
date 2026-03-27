"""MoE: Custom CSV v3 — CORRECT format from dsv3_fp4_tuned_fmoe.csv"""
import os
import tempfile

# CORRECT CSV header (24 columns from tuned_fmoe.csv — NO _tag column)
# use_g1u1=1 (not False), doweight_stage1=0 (not False)
CSV_HEADER = "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw"

S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x64 = "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_64 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_none_FP4X2_FP4X2_B16"
S2_256 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_none_FP4X2_FP4X2_B16"

ACT = "ActivationType.Silu"
DT = "torch.bfloat16"
QA = "torch.float4_e2m1fn_x2"
QW = "torch.float4_e2m1fn_x2"
QT = "QuantType.per_1x32"

ROWS = []
# E=33 shapes (the ones with ZERO tuned configs)
for token in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    est_m = max(1, token * 9 // 33)
    if est_m < 10:
        bm, s1, s2 = 32, S1_64, S2_64
    elif est_m < 50:
        bm, s1, s2 = 64, S1_256, S2_256
    elif est_m < 200:
        bm, s1, s2 = 128, S1_256x64, S2_256
    else:
        bm, s1, s2 = 128, S1_256x64, S2_256
    for inter_dim in [512, 2048]:
        # cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_a,q_w,q_type,use_g1u1,doweight,block_m,ksplit,us1,s1,err1,us2,s2,err2,us,run_1stage,tflops,bw
        ROWS.append(f"256,{token},7168,{inter_dim},33,9,{ACT},{DT},{QA},{QW},{QT},1,0,{bm},0,0,{s1},0,0,{s2},0,0,0,0,0")

csv_content = CSV_HEADER + "\n" + "\n".join(ROWS) + "\n"
csv_path = "/tmp/custom_tuned_fmoe_v3.csv"
with open(csv_path, "w") as f:
    f.write(csv_content)

os.environ["AITER_CONFIG_FMOE"] = csv_path
os.environ["CU_NUM"] = "256"
os.environ["AITER_USE_NT"] = "0"

import torch
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType
from typing import Dict, Tuple

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

fm.use_nt = lambda t, k, e: False

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
