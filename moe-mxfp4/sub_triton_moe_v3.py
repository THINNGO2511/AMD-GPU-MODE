#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Triton MXFP4 MoE v3. Fix: A_scale must not be None.
Pass torch.ones(1) as per-tensor scale (dynamic_mxfp4_quant absorbs scale into E8M0).
Also B_scale = torch.ones(1) (weight per-tensor scale is 1.0).
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

import torch
import triton.language as tl
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_ones = None

def _try_triton_moe(data):
    global _ones
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    try:
        from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
        from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        M = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        E = gate_up_weight.shape[0]
        d_hidden_pad = config["d_hidden_pad"]
        d_expert_pad = config["d_expert_pad"]
        d_hidden = config["d_hidden"]

        if _ones is None:
            _ones = torch.ones(1, dtype=torch.float32, device="cuda")

        # Sort
        block_size = 32
        max_num_tokens_padded = int(topk_ids.numel() + E * block_size - topk)
        max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)

        sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device="cuda")
        sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device="cuda")
        sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device="cuda")
        num_valid_ids = torch.empty(max_num_m_blocks * 2, dtype=torch.int32, device="cuda")
        moe_buf = torch.empty(M * topk, d_hidden_pad, dtype=torch.bfloat16, device="cuda")

        sort_fn = getattr(aiter, 'moe_sorting_opus_fwd', aiter.moe_sorting_fwd)
        sort_fn(
            topk_ids, topk_weights,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf,
            E, block_size,
        )

        # Quantize A
        a1_fp4, a1_mx_scale = dynamic_mxfp4_quant(hidden_states)

        # Stage 1
        intermediate = torch.empty(M * topk, 2 * d_expert_pad, dtype=torch.bfloat16, device="cuda")

        cfg = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 8}

        fused_moe_mxfp4_silu(
            a1_fp4.view(dtypes.fp4x2),
            gate_up_weight,
            intermediate,
            _ones,                   # A_scale (per-tensor = 1.0, scale in E8M0)
            _ones,                   # B_scale (per-tensor = 1.0, scale in E8M0)
            a1_mx_scale,             # A_mx_scale (E8M0 block scales)
            gate_up_weight_scale,    # B_mx_scale (RAW E8M0)
            topk_weights,
            topk_ids,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids[:1],       # num_tokens_post_padded (scalar tensor)
            False,                   # mul_routed_weight
            topk,
            False,                   # swizzle_mx_a
            False,                   # swizzle_mx_b
            cfg,
            tl.bfloat16,
        )

        # Requantize intermediate
        a2_fp4, a2_mx_scale = dynamic_mxfp4_quant(intermediate[:M * topk, :d_expert_pad])

        # Stage 2
        output = torch.zeros(M, d_hidden_pad, dtype=torch.bfloat16, device="cuda")

        fused_moe_mxfp4(
            a2_fp4.view(dtypes.fp4x2),
            down_weight,
            output,
            _ones,
            _ones,
            a2_mx_scale,
            down_weight_scale,
            topk_weights,
            topk_ids,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids[:1],
            True,                    # mul_routed_weight (stage2)
            topk,
            False,
            False,
            cfg,
            tl.bfloat16,
        )

        return output[:, :d_hidden]

    except Exception as e:
        print(f"Triton MoE v3 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def custom_kernel(data: input_t) -> output_t:
    result = _try_triton_moe(data)
    if result is not None:
        return result

    (hidden_states, *_, topk_weights, topk_ids, config) = data
    fm.use_nt = lambda t, k, e: False
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(
        hidden_states, data[5], data[6], topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=data[7], w2_scale=data[8],
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
