#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Use aiter's TRITON MXFP4 MoE kernels directly.
Bypasses CK 2-stage pipeline entirely → no 130s JIT timeout.
Uses tl.dot_scaled with native MFMA FP4 on gfx950.
Takes RAW (un-shuffled) weights and scales from input.

Key paths on runner:
  aiter.ops.triton.moe.moe_op_mxfp4_silu_fused.fused_moe_mxfp4_silu  (Stage 1)
  aiter.ops.triton.moe.moe_op_mxfp4.fused_moe_mxfp4                   (Stage 2)
"""
import torch
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as fm

_patched = False
_sort_cache = {}


def _init():
    """Probe and import the Triton MoE kernels."""
    global _patched
    if _patched:
        return
    _patched = True

    # Probe what's available
    try:
        from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
        print(f"FOUND fused_moe_mxfp4_silu: {type(fused_moe_mxfp4_silu)}")
    except ImportError as e:
        print(f"fused_moe_mxfp4_silu import error: {e}")
        # Try alternative paths
        try:
            from aiter.ops.triton.moe import moe_op_mxfp4_silu_fused
            print(f"Found module: {dir(moe_op_mxfp4_silu_fused)}")
        except ImportError as e2:
            print(f"Module import error: {e2}")

    try:
        from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4
        print(f"FOUND fused_moe_mxfp4: {type(fused_moe_mxfp4)}")
    except ImportError as e:
        print(f"fused_moe_mxfp4 import error: {e}")

    # Check what's in the triton moe directory
    try:
        import aiter.ops.triton.moe as moe_mod
        print(f"aiter.ops.triton.moe contents: {[x for x in dir(moe_mod) if not x.startswith('_')]}")
    except Exception as e:
        print(f"moe module error: {e}")

    # Also probe sorting
    try:
        print(f"moe_sorting_opus_fwd: {type(aiter.moe_sorting_opus_fwd)}")
    except:
        pass


def custom_kernel(data: input_t) -> output_t:
    _init()

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Try the Triton MoE path first
    try:
        from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
        from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4

        M = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        E = gate_up_weight.shape[0]
        d_hidden = config["d_hidden"]
        d_expert = config["d_expert"]
        d_hidden_pad = config["d_hidden_pad"]
        d_expert_pad = config["d_expert_pad"]

        # Step 1: Sort tokens by expert
        block_size = 32
        num_experts = E
        max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
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
                num_experts, block_size,
            )
        else:
            aiter.moe_sorting_fwd(
                topk_ids, topk_weights,
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf,
                num_experts, block_size,
            )

        # Step 2: Quantize activations to MXFP4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        a1_fp4, a1_scale = dynamic_mxfp4_quant(hidden_states)

        # Step 3: Stage 1 — gate_up GEMM + SiLU (Triton, raw weights)
        intermediate = torch.empty(M * topk, 2 * d_expert_pad, dtype=torch.bfloat16, device="cuda")

        fused_moe_mxfp4_silu(
            a1_fp4.view(dtypes.fp4x2) if hasattr(a1_fp4, 'view') else a1_fp4,
            gate_up_weight,  # RAW un-shuffled weights
            intermediate,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            topk,
            w1_scale=gate_up_weight_scale,  # RAW un-shuffled scales
            a1_scale=a1_scale,
            block_m=block_size,
        )

        # Step 4: Requantize intermediate
        a2_fp4, a2_scale = dynamic_mxfp4_quant(intermediate[:M * topk, :d_expert_pad])

        # Step 5: Stage 2 — down GEMM + weighted sum (Triton, raw weights)
        output = torch.zeros(M, d_hidden_pad, dtype=torch.bfloat16, device="cuda")

        fused_moe_mxfp4(
            a2_fp4.view(dtypes.fp4x2) if hasattr(a2_fp4, 'view') else a2_fp4,
            down_weight,  # RAW un-shuffled weights
            output,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            topk,
            w2_scale=down_weight_scale,
            a2_scale=a2_scale,
            sorted_weights=sorted_weights,
            block_m=block_size,
        )

        return output[:, :d_hidden]

    except Exception as e:
        print(f"Triton MoE failed: {e}, falling back to fused_moe")
        # Fallback to standard fused_moe
        fm.use_nt = lambda token, topk, expert: False
        from aiter.fused_moe import fused_moe
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
