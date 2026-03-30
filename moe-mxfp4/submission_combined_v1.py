"""
MoE — Combined optimizations:
1. CK kernel injection for ALL E<=64 (including d=2048 with 64x32 kernel)
2. Pre-allocated sorting buffers (avoid CUDA malloc in hot loop)
3. use_nt=False globally (test for E=257 too)
4. Opus sorting enabled
5. Block_m optimization (32 for small, 64 for large est_m)
"""
import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_sort_cache = {}

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _prealloc_sort(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype,
                   block_size, expert_mask, num_local_tokens, dispatch_policy, use_opus):
    """Pre-allocate sorting buffers to avoid CUDA malloc in hot loop."""
    device = topk_ids.device
    M, topk = topk_ids.shape
    key = (M, topk, num_experts, block_size, model_dim, moebuf_dtype)

    if key not in _sort_cache:
        max_num_tokens_padded = int(M * topk + num_experts * block_size - topk)
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
        _sort_cache[key] = (
            torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device),
            torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device),
            torch.empty(max_num_m_blocks, dtype=torch.int32, device=device),
            torch.empty(2, dtype=torch.int32, device=device),
            torch.empty((M, model_dim), dtype=moebuf_dtype, device=device),
        )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sort_cache[key]

    fwd_fn = aiter.moe_sorting_opus_fwd if use_opus else aiter.moe_sorting_fwd
    fwd_fn(topk_ids, topk_weights, sorted_ids, sorted_weights, sorted_expert_ids,
           num_valid_ids, moe_buf, num_experts, int(block_size),
           expert_mask, num_local_tokens, dispatch_policy)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Pre-alloc sort buffers
    fm._moe_sorting_impl = _prealloc_sort

    # use_nt=False for ALL experts (test if E=257 also benefits)
    fm.use_nt = lambda t, k, e: False

    # Block_m optimization
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # CK kernel injection for ALL E<=64 (including d>=2048)
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
                    # d>=2048: always use 64x32 (256x32 crashes)
                    if inter_dim >= 2048:
                        kn1 = STAGE1_64
                    else:
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
