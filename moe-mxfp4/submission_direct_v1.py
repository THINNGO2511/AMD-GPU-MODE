#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Direct CK stage1/stage2 calls, bypassing fused_moe wrapper.
Eliminates: get_2stage_cfgs lookup, enum conversions, get_padded_M,
MOEMetadata/functools.partial creation, _moe_sorting_impl allocations.
Pre-allocates ALL sorting buffers per shape.
"""
import torch
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.utility import dtypes
from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort

# Proven kernel names from DSv3 tuned CSV
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Pre-allocated buffer cache: keyed by (bs, E, inter_dim, block_m)
_bufs = {}


def _get_config(bs, E, inter_dim):
    """Determine block_m, kernel names, use_nt for each shape."""
    topk = 9
    est_m = bs * topk // E

    if E > 64:
        # E=257: use tuned kernel names
        block_m = 32
        if bs >= 128 and bs < 512:
            k1 = STAGE1_256
        else:
            k1 = STAGE1_64
        k2 = STAGE2_V1
        use_nt = False
    else:
        # E=33: use our proven configs
        block_m = 32 if est_m < 50 else 64
        use_nt = False
        if inter_dim < 2048:  # d_expert_pad < 2048
            k1 = STAGE1_256 if est_m >= 100 else STAGE1_64
            k2 = STAGE2_V1
        else:
            # d=2048: no proven kernel, use default
            k1 = ""
            k2 = ""

    return block_m, k1, k2, use_nt


def _get_bufs(bs, E, inter_dim, model_dim, block_m, device):
    """Pre-allocate or retrieve sorting + intermediate buffers."""
    key = (bs, E, inter_dim, block_m)
    if key in _bufs:
        return _bufs[key]

    topk = 9
    max_num_tokens_padded = bs * topk + E * block_m - topk
    max_num_m_blocks = (max_num_tokens_padded + block_m - 1) // block_m

    bufs = {
        'sorted_ids': torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device),
        'sorted_weights': torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device),
        'sorted_expert_ids': torch.empty(max_num_m_blocks, dtype=torch.int32, device=device),
        'num_valid_ids': torch.empty(2, dtype=torch.int32, device=device),
        'moe_out': torch.empty((bs, model_dim), dtype=torch.bfloat16, device=device),
        'a2': torch.empty((bs, topk, inter_dim), dtype=torch.bfloat16, device=device),
    }
    _bufs[key] = bufs
    return bufs


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    bs = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    E = gate_up_weight_shuffled.shape[0]
    # inter_dim = d_expert_pad (after SiLU gate*up fusion halves the 2*d_expert_pad)
    inter_dim = down_weight_shuffled.shape[2] * 2  # d_expert_pad from w2's K dim (fp4x2)
    model_dim = gate_up_weight_shuffled.shape[2] * 2  # d_hidden_pad (fp4x2 packed)

    # Get config for this shape
    block_m, k1_name, k2_name, use_nt = _get_config(bs, E, inter_dim)

    # Get pre-allocated buffers
    bufs = _get_bufs(bs, E, inter_dim, model_dim, block_m, hidden_states.device)
    sorted_ids = bufs['sorted_ids']
    sorted_weights = bufs['sorted_weights']
    sorted_expert_ids = bufs['sorted_expert_ids']
    num_valid_ids = bufs['num_valid_ids']
    moe_out = bufs['moe_out']
    a2 = bufs['a2']

    # Step 1: Sort tokens by expert
    aiter.moe_sorting_opus_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_out,  # moe_buf = output buffer
        E, block_m,
        None,  # expert_mask
        None,  # num_local_tokens
        0,     # dispatch_policy
    )

    # Step 2: Quantize A1 (hidden_states) to MXFP4 + sort scales
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=bs,
        topk=1,
        block_size=block_m,
    )

    # Step 3: Stage 1 GEMM (gate+up with SiLU)
    w1_scale_e8m0 = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    aiter.ck_moe_stage1_fwd(
        a1,                        # hidden_states (quantized)
        gate_up_weight_shuffled,   # w1
        down_weight_shuffled,      # w2 (unused in stage1 but required)
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        a2,                        # output
        topk,
        k1_name,                   # kernelName
        w1_scale_e8m0,             # w1_scale
        a1_scale,                  # a1_scale
        block_m,
        None,                      # sorted_weights (doweight_stage1=False)
        QuantType.per_1x32,
        ActivationType.Silu,
        0,                         # splitk (0 for per_1x32)
        use_nt,                    # use_non_temporal_load
        torch.bfloat16,            # dst_type
    )

    # Step 4: Quantize A2 (intermediate) to MXFP4 + sort scales
    a2_flat = a2.view(-1, inter_dim)
    a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=bs,
        topk=topk,
        block_size=block_m,
    )
    a2_q = a2_q.view(bs, topk, -1)

    # Step 5: Stage 2 GEMM (down projection with weighted accumulation)
    w2_scale_e8m0 = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    aiter.ck_moe_stage2_fwd(
        a2_q,                      # inter_states (quantized)
        gate_up_weight_shuffled,   # w1 (unused in stage2 but required)
        down_weight_shuffled,      # w2
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,                   # output
        topk,
        k2_name,                   # kernelName
        w2_scale_e8m0,             # w2_scale
        a2_scale,                  # a2_scale
        block_m,
        sorted_weights,            # sorted_weights (doweight_stage1=False → weight in stage2)
        QuantType.per_1x32,
        ActivationType.Silu,
        use_nt,                    # use_non_temporal_load
    )

    return moe_out
