#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — "sepqsort" v2: Force separate quant + sort path.
The fused_dynamic_mxfp4_quant_moe_sort is used for token_num<=1024.
Force threshold to 0 so ALL cases use the separate path:
  1. quant_func(hidden_states) → a1_fp4, a1_scale
  2. fp4_utils.moe_mxfp4_sort(a1_scale, ...) → sorted scales
This might be faster if the fused kernel has overhead.
Combined with our best CK kernel injection.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # 1. use_nt=False for E<=64
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # 2. block_m for E<=64
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)

    # 3. CK injection for E<=64 d<2048
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
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None

    # 4. Force separate quant + sort path by monkey-patching fused_moe_2stages
    # The key: replace the fused quant+sort with separate operations
    orig_fused_2stages = fm.fused_moe_2stages

    def patched_fused_2stages(hidden_states, w1, w2, topk, sorted_ids,
                               sorted_weights, sorted_expert_ids, num_valid_ids,
                               moe_out, isG1U1, block_size_M,
                               activation=ActivationType.Silu,
                               quant_type=QuantType.No,
                               doweight_stage1=False,
                               q_dtype_a=None, q_dtype_w=None,
                               w1_scale=None, w2_scale=None,
                               a1_scale=None, a2_scale=None,
                               num_local_tokens=None,
                               hidden_pad=0, intermediate_pad=0,
                               bias1=None, bias2=None):
        # For MXFP4 per_1x32: force separate quant path by pretending token_num > 1024
        if quant_type == QuantType.per_1x32 and hidden_states.shape[0] <= 1024:
            # Trick: temporarily increase the visible token count
            # Actually, we can't easily trick the function
            # Instead, just call the original — it will use fused path
            pass
        return orig_fused_2stages(
            hidden_states, w1, w2, topk, sorted_ids,
            sorted_weights, sorted_expert_ids, num_valid_ids,
            moe_out, isG1U1, block_size_M,
            activation=activation, quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            q_dtype_a=q_dtype_a, q_dtype_w=q_dtype_w,
            w1_scale=w1_scale, w2_scale=w2_scale,
            a1_scale=a1_scale, a2_scale=a2_scale,
            num_local_tokens=num_local_tokens,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
            bias1=bias1, bias2=bias2)

    # Actually, the threshold is a LOCAL variable (token_num_quant_moe_sort_switch = 1024)
    # We can't patch it. Instead, try to replace fused_dynamic_mxfp4_quant_moe_sort
    # with separate quant + sort.
    orig_fused_quant_sort = fm.fused_dynamic_mxfp4_quant_moe_sort

    def sep_quant_sort(x, sorted_ids, num_valid_ids, token_num, topk, block_size, scaling_mode='even'):
        """Separate quant then sort — might be faster if fused kernel has overhead."""
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility import fp4_utils

        # Step 1: Quantize independently
        a1_quant, a1_scale_raw = dynamic_mxfp4_quant(x)

        # Step 2: Sort the scales
        a1_scale_sorted = fp4_utils.moe_mxfp4_sort(
            a1_scale_raw,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            block_size=block_size,
        )
        return a1_quant, a1_scale_sorted

    fm.fused_dynamic_mxfp4_quant_moe_sort = sep_quant_sort


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
