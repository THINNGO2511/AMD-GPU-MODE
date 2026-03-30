import os
os.environ["CU_NUM"] = "256"
os.environ["AITER_USE_NT"] = "0"

import torch
from typing import Dict, Tuple
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType
import aiter.fused_moe as fm

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

_patched = False
def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    
    # Force use_nt=False globally
    fm.use_nt = lambda t, k, e: False
    
    # CK injection for E<=64 d=512 ONLY (NOT d=2048 per competitor "drop_2048_injection")
    _orig_get_2stage = fm.get_2stage_cfgs
    
    S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
    S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
    S2_V1 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_none_FP4X2_FP4X2_B16"
    S2_V1_64 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_none_FP4X2_FP4X2_B16"
    
    def custom_get_2stage(*args, **kwargs):
        try:
            cfg = _orig_get_2stage(*args, **kwargs)
        except Exception:
            return None
        if cfg is None:
            return None
        
        # Extract expert count and inter_dim from args
        try:
            E = args[4] if len(args) > 4 else kwargs.get('E', 0)
            inter_dim = args[3] if len(args) > 3 else kwargs.get('inter_dim', 0)
        except Exception:
            return cfg
        
        # Only inject for E<=64 AND d<2048 (drop 2048 injection)
        if E <= 64 and inter_dim < 2048:
            try:
                est_m = args[1] * args[5] // E if len(args) > 5 else 0
                if est_m >= 100:
                    cfg = (cfg[0], S1_256, cfg[2], S2_V1, *cfg[4:]) if len(cfg) > 4 else cfg
                else:
                    cfg = (cfg[0], S1_64, cfg[2], S2_V1_64, *cfg[4:]) if len(cfg) > 4 else cfg
            except Exception:
                pass
        
        return cfg
    
    fm.get_2stage_cfgs = custom_get_2stage

def custom_kernel(data: input_t) -> output_t:
    _patch()
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    
    return fused_moe(
        hidden_states, w1_qw, w2_qw,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=w1_qs, w2_scale=w2_qs,
    )
