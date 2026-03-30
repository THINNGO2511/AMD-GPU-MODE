"""MoE: selective ksplit=4 — only for E=33 d=2048 (olezhka's approach)"""
import os
os.environ['CU_NUM'] = '256'
os.environ['AITER_USE_NT'] = '0'
os.environ['AITER_KSPLIT'] = '0'

import torch
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType

_current_inter_dim = 0
_current_E = 0

_orig_get_ksplit = fm.get_ksplit

def _custom_ksplit(*args, **kwargs):
    # ksplit=4 ONLY for E=33 d=2048 (the geomean killer at 344μs)
    if _current_E <= 64 and _current_inter_dim >= 2048:
        return 4
    return 0  # auto for everything else

fm.get_ksplit = _custom_ksplit
fm.use_nt = lambda t, k, e: False

def custom_kernel(data):
    global _current_inter_dim, _current_E
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    _current_E = topk_ids.max().item() + 1
    _current_inter_dim = config['d_expert']
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
