#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Tune block_m and use_nt for E=33 cases.
E=257 cases already use tuned CK kernel configs from dsv3_fp4_tuned_fmoe.csv.
E=33 cases use defaults — try optimizing block_m selection.

Baseline E=33 timings:
  bs=16, d=512: block_m=32, use_nt=True  → 94.5μs
  bs=128, d=512: block_m=64, use_nt=True → 129μs
  bs=512, d=512: block_m=64, use_nt=False → 214μs
  bs=512, d=2048: block_m=128, use_nt=False → 351μs
"""
import torch
import os
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False


def _patch_block_m():
    global _patched
    if _patched:
        return
    _patched = True

    # Override block_m selection for better performance
    original_get_block_size_M = fm.get_block_size_M

    def patched_get_block_size_M(token, topk, expert, inter_dim):
        est_m = token * topk // expert
        # For E=33 (fewer experts, more tokens per expert):
        if expert <= 64:
            if est_m < 10:
                return 32  # same as default
            elif est_m < 50:
                return 32  # try smaller block for better occupancy
            elif est_m < 200:
                return 64  # same
            else:
                return 128
        return original_get_block_size_M(token, topk, expert, inter_dim)

    fm.get_block_size_M = patched_get_block_size_M

    # Also try forcing non-temporal load off for better cache utilization
    original_use_nt = fm.use_nt

    def patched_use_nt(token, topk, expert):
        est_m = token * topk // expert
        # For small batches with E=33, disable NT load
        if expert <= 64 and est_m < 50:
            return False  # try without non-temporal load
        return original_use_nt(token, topk, expert)

    fm.use_nt = patched_use_nt

    # Clear LRU cache
    fm.get_2stage_cfgs.cache_clear()
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    _patch_block_m()

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
