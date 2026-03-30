"""MoE: Best combined — d2048_tuned settings + inject_metadata CK injection"""
import os
os.environ['CU_NUM'] = '256'
os.environ['AITER_USE_NT'] = '0'

import torch
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType
from typing import Dict, Tuple

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

_current_inter_dim = 0

# block_m: 32 for E=257, 128 for d=2048, heuristic for rest
_orig_bm = fm.get_block_size_M
def _bm(t, k, e, d):
    if e > 64: return 32
    if d >= 2048: return 128
    return _orig_bm(t, k, e, d)
fm.get_block_size_M = _bm

# use_nt: True for d=2048 ONLY (counterintuitive but helps with L2 thrashing)
def _nt(t, k, e):
    return True if _current_inter_dim >= 2048 else False
fm.use_nt = _nt

# CK kernel injection for E=33 d<2048 (proven faster)
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_none_FP4X2_FP4X2_B16"
S2_V1_256 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_none_FP4X2_FP4X2_B16"

import functools
_orig_cfgs = fm.get_2stage_cfgs
try:
    _orig_cfgs_unwrap = fm.get_2stage_cfgs.__wrapped__
    fm.get_2stage_cfgs.cache_clear()
except:
    _orig_cfgs_unwrap = _orig_cfgs

def _custom_cfgs(*args, **kwargs):
    result = _orig_cfgs_unwrap(*args, **kwargs)
    # Only inject for E<=64 and d<2048
    if len(args) >= 4:
        inter_dim = args[2]
        expert = args[3]
        token = args[0]
        if expert <= 64 and inter_dim < 2048:
            est_m = token * 9 // expert if expert > 0 else 0
            s1 = S1_256 if est_m >= 100 else S1_64
            s2 = S2_V1
            try:
                return fm.MOEMetadata(
                    functools.partial(fm.ck_moe_stage1, kernelName=s1, 
                                    activation=ActivationType.Silu, 
                                    quant_type=QuantType.per_1x32,
                                    dtype=torch.bfloat16, splitk=0,
                                    use_non_temporal_load=False),
                    functools.partial(fm.aiter.ck_moe_stage2_fwd, kernelName=s2,
                                    activation=ActivationType.Silu,
                                    quant_type=QuantType.per_1x32,
                                    use_non_temporal_load=False),
                    result.block_m, result.ksplit, result.run_1stage)
            except:
                pass
    return result

fm.get_2stage_cfgs = _custom_cfgs

def custom_kernel(data: input_t) -> output_t:
    global _current_inter_dim
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    _current_inter_dim = config['d_expert']
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
