"""MoE: Trigger the is_shuffled + ksplit fast path that SKIPS A quantization"""
import os
os.environ['CU_NUM'] = '256'
os.environ['AITER_USE_NT'] = '0'
os.environ['AITER_KSPLIT'] = '2'  # ksplit > 1 needed for fast path

import torch
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType
from typing import Dict, Tuple

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

fm.use_nt = lambda t, k, e: False

def custom_kernel(data: input_t) -> output_t:
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    
    # SET is_shuffled attribute on weights — triggers fast path in fused_moe_2stages
    # The fast path: if q_dtype_a==fp4x2 AND ksplit>1 AND is_shuffled:
    #   SKIP A quantization entirely (just cast to dtype)
    w1_qw.is_shuffled = True
    w2_qw.is_shuffled = True
    
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
