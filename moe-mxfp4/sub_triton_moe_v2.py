#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Triton MXFP4 MoE kernels with CORRECT API.
Bypasses CK 2-stage pipeline. Uses tl.dot_scaled native FP4 MFMA.
API from probe: fused_moe_mxfp4_silu(A, B, C, A_scale, B_scale,
  A_mx_scale, B_mx_scale, topk_weights, topk_ids, sorted_token_ids,
  expert_ids, num_tokens_post_padded, mul_routed_weight, top_k,
  swizzle_mx_a, swizzle_mx_b, config, compute_type)
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

_patched = False


def _try_triton_moe(data):
    """Attempt Triton MoE path. Returns output or None on failure."""
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
        d_hidden = config["d_hidden"]
        d_expert = config["d_expert"]
        d_hidden_pad = config["d_hidden_pad"]
        d_expert_pad = config["d_expert_pad"]

        # Sort tokens
        block_size = 32
        max_num_tokens_padded = int(topk_ids.numel() + E * block_size - topk)
        max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)

        sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device="cuda")
        sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device="cuda")
        sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device="cuda")
        num_valid_ids = torch.empty(max_num_m_blocks * 2, dtype=torch.int32, device="cuda")
        moe_buf = torch.empty(M * topk, d_hidden_pad, dtype=torch.bfloat16, device="cuda")

        if hasattr(aiter, 'moe_sorting_opus_fwd'):
            aiter.moe_sorting_opus_fwd(
                topk_ids, topk_weights,
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf,
                E, block_size,
            )
        else:
            aiter.moe_sorting_fwd(
                topk_ids, topk_weights,
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf,
                E, block_size,
            )

        # Quantize A to MXFP4
        a1_fp4, a1_mx_scale = dynamic_mxfp4_quant(hidden_states)

        # Stage 1: gate_up GEMM + SiLU
        intermediate = torch.empty(M * topk, 2 * d_expert_pad, dtype=torch.bfloat16, device="cuda")

        triton_config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 8,
        }

        # A_scale = per-tensor (None for dynamic quant, let kernel handle)
        # A_mx_scale = E8M0 block scales from dynamic_mxfp4_quant
        # B_scale = None (no per-tensor weight scale)
        # B_mx_scale = E8M0 weight scales (RAW, not shuffled)
        fused_moe_mxfp4_silu(
            a1_fp4.view(dtypes.fp4x2),     # A: quantized input
            gate_up_weight,                  # B: RAW weight (not shuffled)
            intermediate,                    # C: output
            None,                            # A_scale (per-tensor, None for dynamic)
            None,                            # B_scale (per-tensor, None)
            a1_mx_scale,                     # A_mx_scale (E8M0)
            gate_up_weight_scale,            # B_mx_scale (RAW E8M0)
            topk_weights,                    # routing weights
            topk_ids,                        # expert IDs
            sorted_ids,                      # sorted token IDs
            sorted_expert_ids,               # expert IDs per block
            num_valid_ids[:max_num_m_blocks], # num tokens post padding
            False,                           # mul_routed_weight (stage1: no)
            topk,                            # top_k
            False,                           # swizzle_mx_a
            False,                           # swizzle_mx_b
            triton_config,                   # config
            tl.bfloat16,                     # compute_type
        )

        # Requantize intermediate for stage 2
        # Take only the SiLU-activated portion (first d_expert_pad dims)
        inter_for_stage2 = intermediate[:M * topk, :d_expert_pad]
        a2_fp4, a2_mx_scale = dynamic_mxfp4_quant(inter_for_stage2)

        # Stage 2: down projection + weighted sum
        output = torch.zeros(M, d_hidden_pad, dtype=torch.bfloat16, device="cuda")

        fused_moe_mxfp4(
            a2_fp4.view(dtypes.fp4x2),
            down_weight,
            output,
            None,
            None,
            a2_mx_scale,
            down_weight_scale,
            topk_weights,
            topk_ids,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids[:max_num_m_blocks],
            True,                            # mul_routed_weight (stage2: yes)
            topk,
            False,
            False,
            triton_config,
            tl.bfloat16,
        )

        return output[:, :d_hidden]

    except Exception as e:
        print(f"Triton MoE v2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def custom_kernel(data: input_t) -> output_t:
    # Try Triton path first
    result = _try_triton_moe(data)
    if result is not None:
        return result

    # Fallback to CK path
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data
    fm.use_nt = lambda t, k, e: False
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
