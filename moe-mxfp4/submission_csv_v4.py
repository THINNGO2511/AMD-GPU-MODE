"""MoE: Custom CSV v4 — EXACT key format from source code analysis"""
import os

# KEY FINDING: use_g1u1 must be True (not 1), doweight_stage1 must be False (not 0)
# The CSV uses pandas, and the index is created from string values
# The lookup key uses: str(activation), str(dtype), str(q_dtype_a), etc.
# And use_g1u1/doweight_stage1 are Python booleans: True/False

# Stage kernels from DSv3 CSV (proven working for E=257)
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_64 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_256 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# CSV header must match tuned_fmoe.csv EXACTLY (24 columns, no _tag)
HEADER = "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw"

# E=33 entries — the 4 shapes with ZERO existing configs
# use_g1u1 and doweight_stage1: need to match what pandas reads
# The source does: df.set_index(_INDEX_COLS).to_dict("index")
# pandas reads "True"/"False" from CSV as strings, but the lookup key uses Python bool
# ACTUALLY: the source code uses int(use_g1u1) and int(doweight_stage1) for the untune file
# But the tune file has raw values. Let me match the DSv3 CSV format exactly.
# DSv3 CSV has: 1,0 for use_g1u1,doweight_stage1

ROWS = []
for token in [16, 128, 512]:
    for inter_dim, d_name in [(512, "d512"), (2048, "d2048")]:
        if token == 16:
            bm, s1, s2 = 32, S1_64, S2_64
        elif token == 128:
            bm, s1, s2 = 32, S1_64, S2_64
        else:  # 512
            bm, s1, s2 = 32, S1_256, S2_256
        ROWS.append(f"256,{token},7168,{inter_dim},33,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,{bm},0,0,{s1},0,0,{s2},0,0,0,0,0")

# Also add the E=33 d=512 bs=512 + d=2048 bs=512 (our worst shapes)
# Try block_m=32 for ALL (DSv3 uses block_m=32 for all E=257 shapes)

csv_content = HEADER + "\n" + "\n".join(ROWS) + "\n"
csv_path = "/tmp/custom_fmoe_v4.csv"
with open(csv_path, "w") as f:
    f.write(csv_content)

# Set env BEFORE importing aiter
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
