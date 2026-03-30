#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Try ksplit>0 for E=33 cases + inject tuned kernel names for E=33.
The DSv3 CSV has configs with 256x tile kernels that may work for E=33.
Also try block_m tuning per problem size.
"""
import torch
import os
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    original_get_block_size_M = fm.get_block_size_M
    original_use_nt = fm.use_nt
    original_get_ksplit = fm.get_ksplit

    def patched_get_block_size_M(token, topk, expert, inter_dim):
        est_m = token * topk // expert
        if expert <= 64:
            # For E=33:
            # bs=16 (est_m=4): block_m=32 default OK
            # bs=128 (est_m=34): try block_m=32 (was 64, proved better)
            # bs=512 (est_m=139): try block_m=64 (default, or try 128)
            if est_m < 10:
                return 32
            elif est_m < 50:
                return 32  # block_m=32 better for moderate batches
            elif inter_dim >= 2048:
                return 64  # d=2048: lower block_m for better CU utilization
            else:
                return 64
        return original_get_block_size_M(token, topk, expert, inter_dim)

    def patched_use_nt(token, topk, expert):
        # Disable NT load for E=33 cases (better cache behavior)
        if expert <= 64:
            return False
        return original_use_nt(token, topk, expert)

    def patched_get_ksplit(token, topk, expert, inter_dim, model_dim):
        # Try ksplit=2 for large E=33 cases with d=2048
        est_m = token * topk // expert
        if expert <= 64 and inter_dim >= 2048 and est_m >= 100:
            return 2
        return original_get_ksplit(token, topk, expert, inter_dim, model_dim)

    fm.get_block_size_M = patched_get_block_size_M
    fm.use_nt = patched_use_nt
    fm.get_ksplit = patched_get_ksplit
    fm.get_2stage_cfgs.cache_clear()
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
