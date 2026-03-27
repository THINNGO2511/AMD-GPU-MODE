import os
import torch
from task import input_t, output_t

# Hybrid: v3 stage1 for E=257 (helps) + original kernels for E=33 (better)
# E=257: csv_v3 was 131/214/245 vs best_kernels 138/217/250 → v3 wins
# E=33 d=512: best_kernels was 94/116/181 vs csv_v3 88.6/120/219 → mixed
# E=33 d=2048: best_kernels was 336 vs csv_v3 407 → best_kernels wins

import tempfile
_csv_lines = [
    "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw,_tag",
    # E=257 d=256: v3 stage1 (proven better), v1 stage2
    "256,16,7168,256,257,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,41.0,moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16,0.0,26.0,moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16,1.3,67.0,0,0,0,",
    "256,128,7168,256,257,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,87.0,moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16,0.0,53.0,moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16,1.4,140.0,0,0,0,",
    "256,512,7168,256,257,9,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,97.0,moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16,0.0,70.0,moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16,1.3,167.0,0,0,0,",
]
# E=33: DON'T override — let default configs handle (our best_kernels injection works)

_csv_path = "/tmp/hackathon_tuned_fmoe.csv"
with open(_csv_path, "w") as f:
    f.write("\n".join(_csv_lines))
os.environ["AITER_CONFIG_FMOE"] = _csv_path
os.environ["AITER_KSPLIT"] = "0"

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _fm
import functools

# Inject E=33 kernel names via monkey-patch (same as best_kernels approach)
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

_orig_get_2stage = None
if hasattr(_fm, 'get_2stage_cfgs'):
    _orig_get_2stage = _fm.get_2stage_cfgs.__wrapped__ if hasattr(_fm.get_2stage_cfgs, '__wrapped__') else _fm.get_2stage_cfgs

def _custom_get_2stage(*args, **kwargs):
    result = _orig_get_2stage(*args, **kwargs)
    if result is not None:
        cfg = list(result) if not isinstance(result, list) else result
        # Check if this is E=33 d=512 (inter_dim=512 or 2048)
        # For E=33: use v3 stage1 + v1 stage2 (proven best from best_kernels)
        if len(args) >= 5:
            inter_dim = args[2] if len(args) > 2 else kwargs.get('inter_dim', 0)
            expert = args[4] if len(args) > 4 else kwargs.get('expert', 0)
            if expert <= 64 and inter_dim <= 512:
                # Small experts: inject our proven kernels
                pass  # Let CSV handle E=257, default handle E=33
        return result if not isinstance(result, list) else tuple(result)
    return result

# Don't monkey-patch for now — let CSV handle E=257, defaults handle E=33


def custom_kernel(data: input_t) -> output_t:
    (hidden_states, gate_up_weight, down_weight, gate_up_weight_scale,
     down_weight_scale, gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids, expert_mask=None,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
