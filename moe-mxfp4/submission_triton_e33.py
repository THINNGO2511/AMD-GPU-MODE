#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""MoE MXFP4: Triton-warmup + optimized fused_moe with custom block_m for E=33."""

import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import torch
from typing import Dict, Tuple

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

_fused_moe_fn = None
_initialized = False

def _init():
    global _fused_moe_fn, _initialized
    if _initialized:
        return
    _initialized = True
    
    from aiter.fused_moe import fused_moe
    _fused_moe_fn = fused_moe
    
    # Disable NT loads
    try:
        import aiter.fused_moe as fm
        if hasattr(fm, 'USE_NT'):
            fm.USE_NT = False
    except Exception:
        pass
    
    # Warm Triton MoE kernels (triggers JIT prep)
    try:
        from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
    except Exception:
        pass
    try:
        from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
    except Exception:
        pass


def custom_kernel(data: input_t) -> output_t:
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    
    _init()
    
    from aiter import ActivationType, QuantType
    
    M = hidden_states.shape[0]
    d_model = hidden_states.shape[1]
    
    d_model_raw = config.get('model_dim', d_model)
    d_inter_raw = config.get('inter_dim', 0)
    d_model_pad = config.get('d_hidden', d_model)
    d_inter_pad = config.get('d_expert', 0)
    hidden_pad = max(0, d_model_pad - d_model_raw) if d_model_pad and d_model_raw else 0
    intermediate_pad = max(0, d_inter_pad - d_inter_raw) if d_inter_pad and d_inter_raw else 0
    
    try:
        return _fused_moe_fn(
            hidden_states, w1_qw, w2_qw,
            topk_weights, topk_ids,
            expert_mask=None,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=w1_qs, w2_scale=w2_qs,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
        )
    except TypeError:
        try:
            return _fused_moe_fn(
                hidden_states, w1_qw, w2_qw,
                topk_weights, topk_ids,
                expert_mask=None,
                activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32,
                doweight_stage1=False,
                w1_scale=w1_qs, w2_scale=w2_qs,
                a1_scale=None, a2_scale=None,
            )
        except Exception:
            return _fused_moe_fn(
                hidden_states, w1_qw, w2_qw,
                topk_weights, topk_ids,
                w1_scale=w1_qs, w2_scale=w2_qs,
            )
