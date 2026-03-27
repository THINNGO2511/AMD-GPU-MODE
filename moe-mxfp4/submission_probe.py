#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe fused_moe signature + try doweight_stage1=True + cache output.
"""
import torch
import inspect
from typing import Dict
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_probed = False
_out_cache = {}


def custom_kernel(data: input_t) -> output_t:
    global _probed

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    # Probe on first call
    if not _probed:
        _probed = True
        try:
            sig = inspect.signature(fused_moe)
            print(f"[PROBE] fused_moe signature: {sig}")
        except Exception as e:
            print(f"[PROBE] signature failed: {e}")
        try:
            print(f"[PROBE] fused_moe doc: {fused_moe.__doc__[:500] if fused_moe.__doc__ else 'None'}")
        except:
            pass

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    output = fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=True,  # TRY: apply weights in stage 1
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )

    return output
