#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Split shared expert debug: check shapes and error magnitude.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_dbg = True

def custom_kernel(data: input_t) -> output_t:
    global _dbg
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    n_shared = config["n_shared_experts"]
    n_routed_topk = topk_ids.shape[1] - n_shared

    # Reference: single call
    ref = fused_moe(
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

    if _dbg and n_shared > 0:
        _dbg = False
        # Split approach
        routed = fused_moe(
            hidden_states,
            gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights[:, :n_routed_topk], topk_ids[:, :n_routed_topk],
            expert_mask=None, activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32, doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )
        shared = fused_moe(
            hidden_states,
            gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights[:, n_routed_topk:], topk_ids[:, n_routed_topk:],
            expert_mask=None, activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32, doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )
        combined = routed + shared
        diff = (ref - combined).abs()
        print(f"[DBG] ref shape: {ref.shape}, combined shape: {combined.shape}")
        print(f"[DBG] ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        print(f"[DBG] combined range: [{combined.min():.4f}, {combined.max():.4f}]")
        print(f"[DBG] max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
        print(f"[DBG] routed range: [{routed.min():.4f}, {routed.max():.4f}]")
        print(f"[DBG] shared range: [{shared.min():.4f}, {shared.max():.4f}]")
        rel_err = diff / (ref.abs() + 1e-6)
        print(f"[DBG] max rel err: {rel_err.max():.6f}")

    return ref
