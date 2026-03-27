#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""MoE: best_kernels + AITER_USE_NT=0 (CC says ~4% improvement)"""
import os
os.environ["AITER_USE_NT"] = "0"
os.environ["AITER_KSPLIT"] = "0"

import torch, functools
from task import input_t, output_t
from aiter import ActivationType, QuantType
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe

STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

_orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else None
if _orig:
    @functools.lru_cache(maxsize=None)
    def _custom(*args, **kwargs):
        result = _orig(*args, **kwargs)
        if result is not None:
            try:
                int_args = [a for a in args if isinstance(a, int)]
                if len(int_args) >= 5:
                    inter_dim, expert = int_args[3], int_args[4]
                    token, topk = int_args[1], int_args[5] if len(int_args) > 5 else 9
                    if expert <= 64:
                        est_m = token * topk // expert
                        kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                        # Use replace approach since _asdict doesn't work on MOEMetadata
                        result = result._replace(kernelName1=kn1, kernelName2=STAGE2_V1, block_m=32)
            except Exception:
                pass
        return result
    fm.get_2stage_cfgs = _custom
    fm.cfg_2stages = None

def custom_kernel(data: input_t) -> output_t:
    (hidden_states, gate_up_weight, down_weight, gate_up_weight_scale,
     down_weight_scale, gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data
    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids, expert_mask=None,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=config["d_hidden_pad"] - config["d_hidden"],
        intermediate_pad=config["d_expert_pad"] - config["d_expert"],
    )
