"""MoE: Per-shape tuning — use_nt=True + block_m=256 for d=2048 specifically"""
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
_current_est_m = 0

# Per-shape block_m: 
# - d=2048: try block_m=256 (minimize weight re-reads)
# - d=512: keep heuristic defaults
# - E=257: block_m=32 (from DSv3 CSV)
_orig_bm = fm.get_block_size_M
def _custom_bm(t, k, e, d):
    est_m = t * k // e
    if d >= 2048 and est_m >= 64:
        return 128  # Large block for d=2048
    elif e > 64:
        return 32   # E=257 shapes
    return _orig_bm(t, k, e, d)

fm.get_block_size_M = _custom_bm

# Per-shape use_nt:
# - d=2048: use_nt=True (weights too large for L2, NT loads help)
# - everything else: use_nt=False
def _custom_nt(t, k, e):
    if _current_inter_dim >= 2048:
        return True  # NT loads for large d
    return False
fm.use_nt = _custom_nt

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
