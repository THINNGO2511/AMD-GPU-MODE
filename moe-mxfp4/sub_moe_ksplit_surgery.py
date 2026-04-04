#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — ksplit surgery: Modify MOEMetadata.ksplit AFTER get_2stage_cfgs returns.

Key insight: Previous ksplit=2 attempts overrode fm.get_ksplit, but that doesn't
propagate to the C++ layer. The fused_moe C++ code calls get_2stage_cfgs which
returns a MOEMetadata object with a splitk/ksplit field already baked in.
We need to intercept the MOEMetadata AFTER creation and replace ksplit.

For E>=257 (d=256): ksplit=2 enables a fast path in fused_moe_2stages that skips
the quant step when is_shuffled=True and q_dtype_a==fp4x2. This saves ~28% of
total time (the fused_dynamic_mxfp4_quant_moe_sort kernel).

For E<=64 (d<2048): Keep existing CK kernel injection with ksplit=0.
For E<=64 (d>=2048): Keep defaults, ksplit=2 triggers slow cktile path.

Also keeps ALL existing optimizations from optimized_v2:
- use_nt=False globally
- block_m tuning for E<=64
- CK kernel injection for E<=64 d<2048
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # 1. use_nt=False for ALL shapes (confirmed better)
    fm.use_nt = lambda token, topk, expert: False

    # 2. block_m tuning for E<=64
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128  # d=2048 large batch: default=128 is better
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # 3. Monkey-patch get_2stage_cfgs for BOTH:
    #    a) E<=64 d<2048: CK kernel injection (existing optimization)
    #    b) E>=257: ksplit surgery — replace ksplit=0 with ksplit=2
    #       This enables the fast path in fused_moe_2stages that skips quant
    #       when metadata.ksplit > 1 AND is_shuffled=True AND q_dtype_a==fp4x2
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

        # (a) E<=64 d<2048: inject proven CK kernel names
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=STAGE2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass

        # (b) E>=257 (d=256): ksplit surgery — set ksplit=2
        #     This does NOT trigger the slow cktile path (that's only for d=2048).
        #     For d=256, ksplit=2 enables the fast path that skips quant overhead.
        #     We also need to update the stage1 partial's splitk parameter to match.
        if (expert > 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and result.ksplit != 2):
            try:
                # Reconstruct stage1 partial with splitk=2 if it has splitk kwarg
                new_stage1 = result.stage1
                if hasattr(result.stage1, 'keywords'):
                    kw = dict(result.stage1.keywords)
                    if 'splitk' in kw:
                        kw['splitk'] = 2
                        new_stage1 = functools.partial(
                            result.stage1.func, *result.stage1.args, **kw)

                # Create new MOEMetadata with ksplit=2
                return fm.MOEMetadata(
                    new_stage1,
                    result.stage2,
                    result.block_m,
                    2,                 # ksplit=2 (was 0)
                    result.run_1stage)
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
