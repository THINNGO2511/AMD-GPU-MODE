"""
MoE: Wider block_m for E=33 shapes + stage2 kernel variant.
Competitor "blockmwide_stage2_sepqsort" suggests:
1. block_m=128 for E=33 bs=512 (est_m ~140)
2. Different stage2 kernel (v3 instead of v1 for d=512)
3. Still use opus sorting + no NT
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Small tiles
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
# Stage2 variants - try v3 for larger shapes
S2_V1_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_V3_32 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # Wider block_m: 128 for large est_m, 64 for medium, 32 for small
    orig_bsm = fm.get_block_size_M
    def wide_bsm(t, k, e, d):
        if e <= 64:
            est_m = t * k // e
            if est_m >= 100:
                return 128
            elif est_m >= 30:
                return 64
            else:
                return 32
        return orig_bsm(t, k, e, d)
    fm.get_block_size_M = wide_bsm

    try:
        fm._USE_OPUS_MOE_SORTING = True
    except:
        pass

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

        if expert <= 64 and q_type == QuantType.per_1x32 and not result.run_1stage:
            est_m = token * topk // expert
            if inter_dim < 2048:
                try:
                    kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                    if not kw.get('kernelName', ''):
                        # Stage1: use larger tile for medium est_m
                        kn1 = S1_256 if est_m >= 50 else S1_64
                        # Stage2: try v3 for larger shapes, v1 for small
                        kn2 = S2_V3_32 if est_m >= 100 else S2_V1_32
                        bm = 128 if est_m >= 100 else (64 if est_m >= 30 else 32)
                        return fm.MOEMetadata(
                            functools.partial(fm.ck_moe_stage1,
                                kernelName=kn1, activation=activation,
                                quant_type=q_type, dtype=dtype,
                                splitk=0, use_non_temporal_load=False),
                            functools.partial(aiter.ck_moe_stage2_fwd,
                                kernelName=kn2, activation=activation,
                                quant_type=q_type, use_non_temporal_load=False),
                            bm, 0, False)
                except:
                    pass

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
