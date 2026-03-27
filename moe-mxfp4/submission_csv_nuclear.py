"""MoE: Nuclear CSV override — write to model_configs dir + clear lru_cache"""
import os

# Write CSV to the ACTUAL model_configs directory that gets auto-merged
CSV_PATH = "/home/runner/aiter/aiter/configs/model_configs/e33_custom_tuned_fmoe.csv"

S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_64 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

HEADER = "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw"
ROWS = []
for token in [16, 128, 512]:
    for inter_dim in [512, 2048]:
        bm = 32 if token <= 128 else (128 if inter_dim >= 2048 else 64)
        s1 = S1_64 if token <= 128 else S1_256
        ROWS.append(f"256,{token},7168,{inter_dim},33,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,True,False,{bm},0,0,{s1},0,0,{S2_64},0,0,False,0,0")

try:
    with open(CSV_PATH, "w") as f:
        f.write(HEADER + "\n" + "\n".join(ROWS) + "\n")
    print(f"Wrote {len(ROWS)} rows to {CSV_PATH}")
except PermissionError:
    print(f"Cannot write to {CSV_PATH} — trying env var approach")
    import tempfile
    csv_path = "/tmp/e33_tuned_fmoe.csv"
    with open(csv_path, "w") as f:
        f.write(HEADER + "\n" + "\n".join(ROWS) + "\n")
    os.environ["AITER_CONFIG_FMOE"] = csv_path

os.environ["CU_NUM"] = "256"
os.environ["AITER_USE_NT"] = "0"

import torch
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType

# Clear lru_cache on get_2stage_cfgs and AITER_CONFIGS
try:
    fm.get_2stage_cfgs.cache_clear()
except:
    pass
try:
    fm.cfg_2stages = None  # Force re-read of CSV
except:
    pass

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
