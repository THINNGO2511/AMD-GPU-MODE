#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE Ultimate v1: Combines ALL proven + competitor-confirmed techniques.

From our optimized_v2 (169μs):
  - use_nt=False for all shapes
  - CK kernel injection for E<=64 d<2048
  - block_m tuning for E<=64

From competitor intelligence (Ryan Mathieu ~151μs, Bortlesboat ~150.9μs):
  - ksplit=2 for E>=257 (confirmed by 2+ competitors)
  - block_m=16 for E>=257 small batches (Ryan's "16128" pattern)
  - sepqsort: token_num_quant_moe_sort_switch=0 (separate quant+sort)

Combined approach for each case:
  E=257 d=256: sepqsort + ksplit=2 + block_m=16/32 (competitor-proven)
  E=33 d=512: CK injection + block_m=32/64 + ksplit=0 (our proven)
  E=33 d=2048: default kernels + block_m=64/128 + ksplit=0 (avoid cktile)
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

    # 1. use_nt=False for ALL shapes (proven)
    fm.use_nt = lambda token, topk, expert: False

    # 2. Separate quant+sort for E>=257 (competitor-proven: Ryan Mathieu, Bortlesboat)
    # This bypasses the fused Triton quant kernel and uses per-op quant + separate sort
    try:
        fm.token_num_quant_moe_sort_switch = 0
    except Exception:
        pass

    # 3. ksplit=2 for E>=257 (competitor-confirmed), 0 for E<=64
    # E>=257 d=256: ksplit=2 improves parallelism for sparse experts
    # E<=64 d<=512: ksplit=0 (default is optimal)
    # E<=64 d=2048: ksplit=0 (ksplit>0 triggers slow cktile path)
    orig_get_ksplit = fm.get_ksplit
    def new_get_ksplit(token, topk, expert, inter_dim, model_dim):
        if expert >= 257:
            return 2
        return 0
    fm.get_ksplit = new_get_ksplit

    # 4. block_m tuning (combined: competitor + our proven)
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        est_m = token * topk // expert
        if expert >= 257:
            # Ryan Mathieu pattern: 16 for small, 32 for medium, 128 for large
            if est_m < 10:
                return 16
            elif est_m < 50:
                return 32
            else:
                return 128
        elif expert <= 64:
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128  # d=2048 large batch: default=128 is better
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # 5. CK kernel injection for E<=64 d<2048 (proven in optimized_v2)
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
