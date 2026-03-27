#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Inject kernel names from aiter commit fc0c54bb for E=33 d=2048.

The commit added tuned configs for (expert=32, topk=8, d=2048) but our
benchmark uses (expert=33, topk=9). The CSV won't match, so we monkey-patch.

Key difference from previous injection attempts:
- Previous: used small tiles (64x32, 256x32) for d=2048 → 17% WORSE
- This: uses commit's LARGE tiles (256x128) with v3 kernels → should be much better

Commit configs by token count (E=32 topk=8, mapped to our E=33 topk=9):
  token<=64:  block_m=32, S1=64x32x32x128_v3,     S2=256x32x128x128_v3
  token=128:  block_m=64, S1=256x64x128x128_v3,    S2=256x64x128x128_v3
  token>=256: block_m=128, S1=256x128x128x128_v3,  S2=256x128x128x128_v3
  (token=8 exception: S1=256x32x128x128_v3)
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Kernel names from commit fc0c54bb for d=2048, FP4, Silu

# Stage1 kernels (by tile size)
S1_64x32 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x32 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x64 = "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x128 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"

# Stage2 kernels (from commit — all v3, larger tiles than our previous v1)
S2_256x32_V3 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_256x64_V3 = "moe_ck2stages_gemm2_256x64x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_256x128_V3 = "moe_ck2stages_gemm2_256x128x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Old stage2 (for E<=64 d<2048 — keep existing injection)
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

        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert

                    if inter_dim >= 2048:
                        # NEW: Inject commit fc0c54bb kernels for d=2048
                        # These use larger tiles tuned specifically for d=2048
                        if est_m >= 100:
                            # Large token (bs=512): 256x128 tiles
                            kn1 = S1_256x128
                            kn2 = S2_256x128_V3
                            bm = 128
                        elif est_m >= 32:
                            # Medium token (bs=128): 256x64 tiles
                            kn1 = S1_256x64
                            kn2 = S2_256x64_V3
                            bm = 64
                        else:
                            # Small token (bs=16): 64x32 tiles
                            kn1 = S1_64x32
                            kn2 = S2_256x32_V3
                            bm = 32

                        return fm.MOEMetadata(
                            functools.partial(fm.ck_moe_stage1,
                                kernelName=kn1, activation=activation,
                                quant_type=q_type, dtype=dtype,
                                splitk=0, use_non_temporal_load=False),
                            functools.partial(aiter.ck_moe_stage2_fwd,
                                kernelName=kn2, activation=activation,
                                quant_type=q_type, use_non_temporal_load=False),
                            bm, 0, False)
                    else:
                        # d<2048: existing injection (same as before)
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
