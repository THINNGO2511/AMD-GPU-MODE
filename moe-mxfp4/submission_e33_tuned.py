#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""MoE: Force use_nt=False for E=33 + optimized block_m + kernel injection.
Vanilla shows E=33 uses default configs with suboptimal use_nt=True for small bs."""
import os, torch, functools
from task import input_t, output_t

# Force no non-temporal loads globally
os.environ["AITER_USE_NT"] = "0"

from aiter import ActivationType, QuantType
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe

STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

_orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else None

if _orig:
    @functools.lru_cache(maxsize=None)
    def _custom_get_2stage(*args, **kwargs):
        result = _orig(*args, **kwargs)
        if result is None:
            return result
        try:
            d = result._asdict()
            int_args = [(i, a) for i, a in enumerate(args) if isinstance(a, int)]
            if len(int_args) >= 5:
                inter_dim = int_args[3][1]
                expert = int_args[4][1]
                token = int_args[1][1]
                topk = int_args[5][1] if len(int_args) > 5 else 9

                # Inject for E<=64 (E=33 cases)
                if expert <= 64:
                    est_m = token * topk // expert
                    # Optimized block_m: 32 for small est_m, 64 for medium
                    if est_m < 50:
                        kn1 = STAGE1_64
                        d['block_m'] = 32
                    elif est_m < 128:
                        kn1 = STAGE1_256
                        d['block_m'] = 64
                    else:
                        kn1 = STAGE1_256
                        d['block_m'] = 64
                    d['kernelName1'] = kn1
                    d['kernelName2'] = STAGE2_V1
                    return type(result)(**d)
        except Exception:
            pass
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
