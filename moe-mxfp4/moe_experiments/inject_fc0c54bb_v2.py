#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Inject fc0c54bb kernels v2: UNCONDITIONAL injection for d>=2048.
v1 failed because heuristic default sets kernelName, so the
'if not kernelName' check skipped our injection. Fixed: bypass that check.

Also adds debug prints to confirm injection is firing.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Stage1 kernels from commit fc0c54bb
S1_64x32 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x32 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x64 = "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x128 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"

# Stage2 kernels from commit (v3 variants, larger tiles)
S2_256x32_V3 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_256x64_V3 = "moe_ck2stages_gemm2_256x64x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_256x128_V3 = "moe_ck2stages_gemm2_256x128x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Old stage2 for E<=64 d<2048
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    fm.use_nt = lambda token, topk, expert: False

    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

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

        # Only inject for E<=64 FP4 2-stage
        if not (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage):
            return result

        est_m = token * topk // expert

        if inter_dim >= 2048:
            # UNCONDITIONAL injection for d>=2048 (bypass kernelName check)
            if est_m >= 100:
                kn1, kn2, bm = S1_256x128, S2_256x128_V3, 128
            elif est_m >= 32:
                kn1, kn2, bm = S1_256x64, S2_256x64_V3, 64
            else:
                kn1, kn2, bm = S1_64x32, S2_256x32_V3, 32
            print("INJECT_D2048: token=%d expert=%d topk=%d est_m=%d bm=%d s1=%s s2=%s" % (
                token, expert, topk, est_m, bm, kn1.split('_')[3], kn2.split('_')[3]))
            return fm.MOEMetadata(
                functools.partial(fm.ck_moe_stage1,
                    kernelName=kn1, activation=activation,
                    quant_type=q_type, dtype=dtype,
                    splitk=0, use_non_temporal_load=False),
                functools.partial(aiter.ck_moe_stage2_fwd,
                    kernelName=kn2, activation=activation,
                    quant_type=q_type, use_non_temporal_load=False),
                bm, 0, False)

        # d<2048: inject only if no existing kernelName
        try:
            kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
            if not kw.get('kernelName', ''):
                if est_m >= 100:
                    kn1 = S1_256x32
                else:
                    kn1 = S1_64x32
                return fm.MOEMetadata(
                    functools.partial(fm.ck_moe_stage1,
                        kernelName=kn1, activation=activation,
                        quant_type=q_type, dtype=dtype,
                        splitk=0, use_non_temporal_load=False),
                    functools.partial(aiter.ck_moe_stage2_fwd,
                        kernelName=S2_V1, activation=activation,
                        quant_type=q_type, use_non_temporal_load=False),
                    32, 0, False)
        except Exception:
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
