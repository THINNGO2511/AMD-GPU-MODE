#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — NO CK kernel injection at all.
Hypothesis: CK injection helps d=512 but HURTS d=2048.
The d=2048 case (344μs) dominates the geomean.
By NOT injecting, we let the default Triton kernels handle everything,
which may be faster overall since the d=2048 bottleneck improves.
Only patches: use_nt=False for E<=64, block_m tuning, CU_NUM=256.
"""
import os
os.environ.setdefault("CU_NUM", "256")

import torch
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

    # Disable non-temporal loads for E<=64
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # Block size tuning for E<=64
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # NO kernel injection — let default handle everything
    # Clear any cached configs
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
