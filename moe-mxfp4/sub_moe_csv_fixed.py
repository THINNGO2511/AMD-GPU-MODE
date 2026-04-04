import os
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

_patched = False

CSV_HEADER = "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw"

S1_64  = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x128 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1  = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_V3_128 = "moe_ck2stages_gemm2_64x128x128x128_1x1_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

ACT = "ActivationType.Silu"
DTYPE = "torch.bfloat16"
QDA = "torch.float4_e2m1fn_x2"
QDW = "torch.float4_e2m1fn_x2"
QT = "QuantType.per_1x32"

def _make_row(cu, tok, mdim, idim, exp, topk, bm, ks, s1, s2):
    return f"{cu},{tok},{mdim},{idim},{exp},{topk},{ACT},{DTYPE},{QDA},{QDW},{QT},1,0,{bm},{ks},1.0,{s1},0.0%,1.0,{s2},0.0%,1.0,0,999.0,999.0"

def _build_csv():
    rows = [CSV_HEADER]
    rows.append(_make_row(256, 16,  7168, 512, 33, 9, 32, 0, S1_64,  S2_V1))
    rows.append(_make_row(256, 128, 7168, 512, 33, 9, 32, 0, S1_64,  S2_V1))
    rows.append(_make_row(256, 512, 7168, 512, 33, 9, 64, 0, S1_256, S2_V1))
    rows.append(_make_row(256, 512, 7168, 2048, 33, 9, 128, 0, S1_256x128, S2_V3_128))
    return "\n".join(rows) + "\n"

def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    csv_content = _build_csv()
    csv_path = "/tmp/custom_fmoe_s19.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    os.environ["AITER_CONFIG_FMOE"] = csv_path
    fm.use_nt = lambda token, topk, expert: False
    fm.cfg_2stages = None
    try:
        orig = fm.get_2stage_cfgs.__wrapped__
        fm.get_2stage_cfgs = functools.lru_cache(maxsize=2048)(orig)
    except Exception:
        pass

def custom_kernel(data):
    _patch()
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data
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
