#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""MoE: Drop d=2048 injection. Defensive monkey-patch that inspects result, not args."""
import os, torch, functools
from task import input_t, output_t

os.environ["AITER_KSPLIT"] = "0"

from aiter import ActivationType, QuantType
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe

STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Inspect the original get_2stage_cfgs to understand its args
_orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else None

if _orig:
    @functools.lru_cache(maxsize=None)
    def _custom_get_2stage(*args, **kwargs):
        result = _orig(*args, **kwargs)
        if result is None:
            return result
        try:
            d = result._asdict()
            # Only inject for inter_dim <= 512 (from the result, not args)
            # Check: the config has inter_dim or we detect from kernel names
            # Safer: check the args by finding the inter_dim positionally
            # get_2stage_cfgs signature: (cu_num, token, model_dim, inter_dim, expert, topk, ...)
            # But some args may be dtype. Let's find inter_dim by checking which args are int
            int_args = [(i, a) for i, a in enumerate(args) if isinstance(a, int)]
            # inter_dim is usually the 4th int arg (idx 3), expert is 5th (idx 4)
            if len(int_args) >= 5:
                inter_dim = int_args[3][1]
                expert = int_args[4][1]
                token = int_args[1][1]
                topk = int_args[5][1] if len(int_args) > 5 else 9
                
                # Only inject for d<=512 AND E<=64
                if expert <= 64 and inter_dim <= 512:
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    d['kernelName1'] = kn1
                    d['kernelName2'] = STAGE2_V1
                    d['block_m'] = 32
                    return type(result)(**d)
        except Exception as e:
            import sys
            print(f"[patch err: {e}, args types: {[type(a).__name__ for a in args]}]", file=sys.stderr)
        return result
    
    fm.get_2stage_cfgs = _custom_get_2stage
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
