#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Try NEW CK kernel tile sizes discovered from probe:
Stage1: 256x64x128x128_1x4 (block_m=64) for large est_m E=33 cases
Stage2: 64x128x128x128_1x1_v3 (larger tiles) for large cases
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Existing kernels
S1_SMALL = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_MEDIUM = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"

# NEW kernel variants from probe
S1_BM64 = "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_SMALL_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_LARGE_V3 = "moe_ck2stages_gemm2_64x128x128x128_1x1_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    fm.use_nt = lambda t, k, e: False

    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def new_get_2stage(token, model_dim, inter_dim, expert, topk,
                       dtype, q_dtype_a, q_dtype_w, q_type,
                       use_g1u1, activation, doweight_stage1,
                       hidden_pad, intermediate_pad, is_shuffled=True):
        result = orig_get_2stage(token, model_dim, inter_dim, expert, topk,
                                dtype, q_dtype_a, q_dtype_w, q_type,
                                use_g1u1, activation, doweight_stage1,
                                hidden_pad, intermediate_pad, is_shuffled)

        # E<=64 (E=33): inject with NEW tile sizes
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            est_m = token * topk // expert
            if est_m >= 64:
                # Large batch: use 256x64 stage1 (block_m=64) + v3 stage2
                return fm.MOEMetadata(
                    functools.partial(fm.ck_moe_stage1,
                        kernelName=S1_BM64, activation=activation,
                        quant_type=q_type, dtype=dtype,
                        splitk=0, use_non_temporal_load=False),
                    functools.partial(aiter.ck_moe_stage2_fwd,
                        kernelName=S2_LARGE_V3, activation=activation,
                        quant_type=q_type, use_non_temporal_load=False),
                    64, 0, False)
            elif est_m >= 4:
                # Medium: 256x32 stage1 + v1 stage2
                return fm.MOEMetadata(
                    functools.partial(fm.ck_moe_stage1,
                        kernelName=S1_MEDIUM, activation=activation,
                        quant_type=q_type, dtype=dtype,
                        splitk=0, use_non_temporal_load=False),
                    functools.partial(aiter.ck_moe_stage2_fwd,
                        kernelName=S2_SMALL_V1, activation=activation,
                        quant_type=q_type, use_non_temporal_load=False),
                    32, 0, False)
            else:
                # Small: 64x32 stage1 + v1 stage2
                return fm.MOEMetadata(
                    functools.partial(fm.ck_moe_stage1,
                        kernelName=S1_SMALL, activation=activation,
                        quant_type=q_type, dtype=dtype,
                        splitk=0, use_non_temporal_load=False),
                    functools.partial(aiter.ck_moe_stage2_fwd,
                        kernelName=S2_SMALL_V1, activation=activation,
                        quant_type=q_type, use_non_temporal_load=False),
                    32, 0, False)

        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    _patch()
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data
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
