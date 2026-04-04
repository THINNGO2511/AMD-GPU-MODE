#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Replace fused_dynamic_mxfp4_quant_moe_sort with separate pytorch quant + sort.
All our benchmark shapes (bs<=512) go through the fused Triton path.
Replacing with pytorch quant + separate moe_mxfp4_sort avoids Triton JIT overhead
and might be faster if the fused kernel is suboptimal.

Also: use_nt=False + block_m tuning (proven). NO CK injection (avoids JIT timeout).
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # 1. Replace fused quant+sort with separate pytorch quant + sort
    # fused_dynamic_mxfp4_quant_moe_sort does quant+sort in one Triton kernel
    # We replace with: get_torch_quant (pytorch) + moe_mxfp4_sort (separate Triton)
    try:
        torch_quant_fn = aiter.get_torch_quant(QuantType.per_1x32)

        original_fused = fm.fused_dynamic_mxfp4_quant_moe_sort

        def pytorch_quant_then_sort(x, sorted_ids, num_valid_ids, token_num, topk,
                                     block_size=32, scaling_mode='even'):
            # Step 1: Quantize with pytorch (no Triton JIT)
            x_fp4, x_scale = torch_quant_fn(x, quant_dtype=dtypes.fp4x2)
            # Step 2: Sort using separate moe_mxfp4_sort
            from aiter.ops.triton.quant import moe_mxfp4_sort
            sorted_x, sorted_scale = moe_mxfp4_sort(
                x_fp4, x_scale, sorted_ids, num_valid_ids,
                token_num=token_num, topk=topk, block_size=block_size
            )
            return sorted_x, sorted_scale

        fm.fused_dynamic_mxfp4_quant_moe_sort = pytorch_quant_then_sort
    except Exception as e:
        print(f"Quant patch failed: {e}")
        # Fall back — try replacing with triton quant instead
        try:
            triton_quant_fn = aiter.get_triton_quant(QuantType.per_1x32)

            def triton_quant_then_sort(x, sorted_ids, num_valid_ids, token_num, topk,
                                        block_size=32, scaling_mode='even'):
                x_fp4, x_scale = triton_quant_fn(x, quant_dtype=dtypes.fp4x2)
                from aiter.ops.triton.quant import moe_mxfp4_sort
                sorted_x, sorted_scale = moe_mxfp4_sort(
                    x_fp4, x_scale, sorted_ids, num_valid_ids,
                    token_num=token_num, topk=topk, block_size=block_size
                )
                return sorted_x, sorted_scale

            fm.fused_dynamic_mxfp4_quant_moe_sort = triton_quant_then_sort
        except Exception as e2:
            print(f"Triton quant fallback also failed: {e2}")

    # 2. use_nt=False for ALL shapes
    fm.use_nt = lambda token, topk, expert: False

    # 3. block_m tuning
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm


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
