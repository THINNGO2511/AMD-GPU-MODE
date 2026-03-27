"""MoE: d2048 tuning + inject_metadata approach (no get_2stage_cfgs patch)"""
import os
os.environ['CU_NUM'] = '256'
os.environ['AITER_USE_NT'] = '0'

import torch
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType
from typing import Dict, Tuple

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

_current_inter_dim = 0

# block_m: 32 for E=257, default for rest
_orig_bm = fm.get_block_size_M
fm.get_block_size_M = lambda t,k,e,d: 32 if e > 64 else _orig_bm(t,k,e,d)

# use_nt: True for d=2048, False for rest
fm.use_nt = lambda t,k,e: True if _current_inter_dim >= 2048 else False

def custom_kernel(data: input_t) -> output_t:
    global _current_inter_dim
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    _current_inter_dim = config['d_expert']
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
