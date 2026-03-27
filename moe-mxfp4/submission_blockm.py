#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Monkey-patch get_block_size_M to force block_m=64 for all sizes.
Default auto-selection optimizes CU utilization but not throughput.
block_m=64 is a middle ground — already compiled for medium batch sizes.
"""
import torch
import functools
from task import input_t, output_t
from aiter import ActivationType, QuantType
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe

# Monkey-patch: force block_m=64 for all sizes
_original_get_block_size_M = fm.get_block_size_M

@functools.lru_cache(maxsize=2048)
def _patched_get_block_size_M(token, topk, expert, inter_dim):
    return 64  # Force 64 for all

fm.get_block_size_M = _patched_get_block_size_M


def custom_kernel(data: input_t) -> output_t:
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
