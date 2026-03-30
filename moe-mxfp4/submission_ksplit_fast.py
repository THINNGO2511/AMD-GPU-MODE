#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — ksplit fast path: Skip quantization entirely!
fused_moe_2stages has a fast path (line 53-64) that skips quant when:
  - quant_type == per_1x32 AND dtype == bf16 AND w1.dtype == fp4x2
  - AND (q_dtype_a == fp4x2 AND metadata.ksplit > 1 AND is_shuffled)
By setting AITER_KSPLIT=2 and w1.is_shuffled=True, we activate this path.
The fast path sets a1=hidden_states.to(dtype), a1_scale=None — NO quant at all.
This could save ~28% of total time (the quant+sort overhead).
RISK: kernel might not handle bf16 input correctly. Test mode first.
"""
import os
os.environ['AITER_KSPLIT'] = '2'

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

    # use_nt=False globally
    fm.use_nt = lambda token, topk, expert: False

    # OPUS sorting
    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True


def custom_kernel(data: input_t) -> output_t:
    _patch()
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    # Mark weights as shuffled to activate the fast path
    gate_up_weight_shuffled.is_shuffled = True
    down_weight_shuffled.is_shuffled = True

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
