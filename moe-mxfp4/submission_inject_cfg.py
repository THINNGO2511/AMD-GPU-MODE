#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Inject tuned CK kernel configs by fixing cu_num mismatch.

Root cause: dsv3_fp4_tuned_fmoe.csv has cu_num=256 (MI300X) but MI355X has 304 CUs.
get_cu_num() returns 304, so key lookup in cfg_2stages always misses.

Fix: monkey-patch get_cu_num() to return 256 so configs match,
OR inject configs with cu_num=304 into the dict.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False


def _patch_configs():
    """Inject DSv3 FP4 tuned configs with correct cu_num for MI355X."""
    global _patched
    if _patched:
        return
    _patched = True

    # Monkey-patch get_cu_num to return 256 (what the CSV expects)
    original_get_cu_num = fm.get_cu_num

    def patched_get_cu_num():
        return 256

    fm.get_cu_num = patched_get_cu_num

    # Clear the LRU cache on get_2stage_cfgs so it re-reads with patched cu_num
    fm.get_2stage_cfgs.cache_clear()

    # Also clear the global cfg_2stages cache to force re-read
    fm.cfg_2stages = None

    print(f"[PATCH] Patched get_cu_num: {original_get_cu_num()} -> 256")
    print(f"[PATCH] Cleared get_2stage_cfgs LRU cache and cfg_2stages")


def custom_kernel(data: input_t) -> output_t:
    _patch_configs()

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
