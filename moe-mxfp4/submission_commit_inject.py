"""MoE: Inject kernel names from aiter commit fc0c54bb for E=33 d=2048"""
import os
os.environ['CU_NUM'] = '256'
os.environ['AITER_USE_NT'] = '0'

import torch, functools
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
import aiter
from aiter import ActivationType, QuantType
from typing import Dict, Tuple

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

# Kernel names from commit fc0c54bb for E=32 d=2048 (adapted for E=33 topk=9)
K_S1 = {
    16:  "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    32:  "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    64:  "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    128: "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    256: "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    512: "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
}
K_S2 = {
    16:  "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    32:  "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    64:  "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    128: "moe_ck2stages_gemm2_256x64x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    256: "moe_ck2stages_gemm2_256x64x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    512: "moe_ck2stages_gemm2_256x128x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
}
BM = {16:32, 32:32, 64:32, 128:64, 256:64, 512:128}

_cur_token = 0
_cur_inter = 0
_orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else fm.get_2stage_cfgs
try: fm.get_2stage_cfgs.cache_clear()
except: pass

def _injected(*args, **kwargs):
    result = _orig(*args, **kwargs)
    if len(args) >= 4 and args[3] <= 64 and args[2] < 2048:  # E<=64, d>=2048
        token = args[0]
        closest = min(K_S1.keys(), key=lambda x: abs(x - token))
        try:
            return fm.MOEMetadata(
                functools.partial(aiter.ck_moe_stage1, kernelName=K_S1[closest],
                                 activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                                 splitk=0, use_non_temporal_load=False),
                functools.partial(aiter.ck_moe_stage2_fwd, kernelName=K_S2[closest],
                                 activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                                 use_non_temporal_load=False),
                BM.get(closest, 128), 0, False)
        except: pass
    return result

fm.get_2stage_cfgs = _injected
fm.use_nt = lambda t,k,e: True if _cur_inter >= 2048 else False
_orig_bm = fm.get_block_size_M
fm.get_block_size_M = lambda t,k,e,d: 32 if e>64 else _orig_bm(t,k,e,d)

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
