#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Direct v3: Bypass C++ wrapper + cache ALL tensor allocations.
v2 proved the direct path works. v3 adds:
1. Cache sorting output tensors (sorted_ids, sorted_weights, etc.)
2. Cache moe_out tensor
3. Call opus sorting directly (skip Python wrapper overhead)
4. Pre-compute block_size_M per case
"""
import sys
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes as aiter_dtypes
import aiter.fused_moe as fm

_patched = False
_sort_buf_cache = {}
_out_cache = {}
_block_m_cache = {}

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Patch use_nt
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # Patch block_m
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # Patch get_2stage_cfgs
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


def _get_sort_bufs(M, topk, num_experts, model_dim, block_size):
    """Get or create cached sorting buffers."""
    key = (M, topk, num_experts, block_size)
    if key not in _sort_buf_cache:
        max_num_tokens_padded = M * topk + num_experts * block_size - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
        _sort_buf_cache[key] = (
            torch.empty(max_num_tokens_padded, dtype=torch.int32, device='cuda'),
            torch.empty(max_num_tokens_padded, dtype=torch.float32, device='cuda'),
            torch.empty(max_num_m_blocks, dtype=torch.int32, device='cuda'),
            torch.empty(2, dtype=torch.int32, device='cuda'),
            torch.empty((M, model_dim), dtype=torch.bfloat16, device='cuda'),
        )
    return _sort_buf_cache[key]


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

    token_num, model_dim = hidden_states.shape
    E = topk_ids.max().item() + 1
    topk = topk_ids.shape[1]
    inter_dim = gate_up_weight_shuffled.shape[1]

    # Get cached block_size_M
    bm_key = (token_num, topk, E, inter_dim)
    if bm_key not in _block_m_cache:
        _block_m_cache[bm_key] = fm.get_block_size_M(token_num, topk, E, inter_dim)
    block_m = _block_m_cache[bm_key]

    # Sorting with cached buffers
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = \
        _get_sort_bufs(token_num, topk, E, model_dim, block_m)

    # Call opus sorting directly with pre-allocated buffers
    sort_fn = aiter.moe_sorting_opus_fwd if hasattr(aiter, 'moe_sorting_opus_fwd') else aiter.moe_sorting_fwd
    sort_fn(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf,
        E, int(block_m),
        None,  # expert_mask
        None,  # num_local_tokens
        0,     # dispatch_policy
    )

    # Fresh output tensor each call (reusing crashes GPU on repeated iterations)
    moe_out = torch.zeros(
        (token_num, model_dim), dtype=torch.bfloat16, device='cuda'
    )

    # Call fused_moe_2stages directly
    fm.fused_moe_2stages(
        hidden_states,
        gate_up_weight_shuffled, down_weight_shuffled,
        topk,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_out,
        False,  # isG1U1
        block_m,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        q_dtype_a=aiter_dtypes.fp4x2,
        q_dtype_w=aiter_dtypes.fp4x2,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        num_local_tokens=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=None,
        bias2=None,
    )

    return moe_out
