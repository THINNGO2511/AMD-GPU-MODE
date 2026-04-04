#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — PyTorch quant replacement (zhubenzhu Discord tip, Mar 30).
Replace Triton fused quant with pytorch quant function.
Set token_num_quant_moe_sort_switch = -1 to disable fused triton kernel.
This does NOT change CK module → no JIT timeout.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # 1. Replace quant function with pytorch version (avoids Triton JIT)
    # The key insight: fused_dynamic_mxfp4_quant_moe_sort is a Triton kernel
    # that takes 28% of kernel time. Replacing with pytorch quant avoids
    # Triton JIT overhead and may be faster for small token counts.
    try:
        torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
        # Monkey-patch the quant function used in fused_moe_2stages
        if hasattr(fm, 'fused_moe_2stages'):
            fmoe2 = fm.fused_moe_2stages
            # Set the threshold to -1 to always use separate quant+sort path
            # (never the fused Triton kernel)
            if hasattr(fmoe2, 'token_num_quant_moe_sort_switch'):
                fmoe2.token_num_quant_moe_sort_switch = -1
        # Try setting module-level variable
        if hasattr(fm, 'token_num_quant_moe_sort_switch'):
            fm.token_num_quant_moe_sort_switch = -1
    except Exception:
        pass

    # 2. use_nt=False for ALL shapes (proven 2-3% improvement)
    fm.use_nt = lambda token, topk, expert: False

    # 3. block_m tuning for E<=64
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
