"""
MoE Optimized v2: Pre-allocated sorting buffers + global use_nt=False + CK injection.
Key changes vs submission_best_kernels.py:
1. Pre-allocate sorting buffers to avoid per-call allocation
2. use_nt=False for ALL experts (including E=257, previously only E<=64)
3. Opus sorting (AITER_USE_OPUS_MOE_SORTING=1)
4. CK kernel injection for E<=64 d<2048 (proven)
"""
import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import dtypes

_patched = False
_sort_cache = {}  # (M, topk, E, block_m, model_dim, device) -> pre-alloc'd buffers

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _prealloc_sorting(M, topk, num_experts, block_size, model_dim, moebuf_dtype, device):
    """Pre-allocate sorting buffers for a given shape."""
    max_num_tokens_padded = int(M * topk + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def _patched_moe_sorting_impl(
    topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype,
    block_size, expert_mask, num_local_tokens, dispatch_policy, use_opus,
):
    """Sorting with pre-allocated buffers to avoid per-call allocation."""
    device = topk_ids.device
    M, topk = topk_ids.shape
    key = (M, topk, num_experts, block_size, model_dim)

    if key not in _sort_cache:
        _sort_cache[key] = _prealloc_sorting(
            M, topk, num_experts, block_size, model_dim, moebuf_dtype, device
        )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sort_cache[key]
    # moe_sorting_fwd initializes moe_buf internally, no need to zero

    fwd_fn = aiter.moe_sorting_opus_fwd if use_opus else aiter.moe_sorting_fwd
    fwd_fn(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf,
        num_experts, int(block_size),
        expert_mask, num_local_tokens, dispatch_policy,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # 1. Replace sorting implementation with pre-allocated version
    fm._moe_sorting_impl = _patched_moe_sorting_impl

    # 2. Global use_nt=False (including E=257)
    fm.use_nt = lambda token, topk, expert: False

    # 3. block_m optimization for E<=64
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # 4. CK kernel injection for E<=64 d<2048
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
