"""
MoE v4: Fix from v3 — preserve original behavior for E=257, only override E<=64.
The v3 broke E=257 shapes because custom_bsm fell through incorrectly.
Also: only inject d>=2048 kernels, don't touch other shapes' defaults.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Small expert kernels (proven good for d<2048)
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Large tile kernels for d=2048 (from fc0c54bb commit)
STAGE1_D2048 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_D2048 = "moe_ck2stages_gemm2_256x128x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Save originals BEFORE patching
    orig_use_nt = fm.use_nt
    orig_bsm = fm.get_block_size_M

    # Disable non-temporal loads only for E<=64
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # Custom block_size_M: only override E<=64, preserve E>64 exactly
    fm.get_block_size_M = lambda t, k, e, d: (
        (32 if t*k//e < 50 else 64) if e <= 64
        else orig_bsm(t, k, e, d)
    )

    # Enable opus sorting
    try:
        fm._USE_OPUS_MOE_SORTING = True
    except:
        pass

    # Override get_2stage_cfgs
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

        # Only override for E<=64 with MXFP4
        if expert <= 64 and q_type == QuantType.per_1x32 and not result.run_1stage:
            est_m = token * topk // expert

            if inter_dim >= 2048:
                # d=2048: Force large-tile kernels unconditionally
                bm = 128 if est_m >= 100 else 64
                return fm.MOEMetadata(
                    functools.partial(fm.ck_moe_stage1,
                        kernelName=STAGE1_D2048, activation=activation,
                        quant_type=q_type, dtype=dtype,
                        splitk=0, use_non_temporal_load=False),
                    functools.partial(aiter.ck_moe_stage2_fwd,
                        kernelName=STAGE2_D2048, activation=activation,
                        quant_type=q_type, use_non_temporal_load=False),
                    bm, 0, False)
            else:
                # d<2048: Inject small/medium CK kernels only if no kernel is set
                try:
                    kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                    if not kw.get('kernelName', ''):
                        kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                        return fm.MOEMetadata(
                            functools.partial(fm.ck_moe_stage1,
                                kernelName=kn1, activation=activation,
                                quant_type=q_type, dtype=dtype,
                                splitk=0, use_non_temporal_load=False),
                            functools.partial(aiter.ck_moe_stage2_fwd,
                                kernelName=STAGE2_32, activation=activation,
                                quant_type=q_type, use_non_temporal_load=False),
                            32, 0, False)
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
