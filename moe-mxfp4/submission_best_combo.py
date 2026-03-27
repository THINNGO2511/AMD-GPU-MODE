"""MoE best combo: CU_NUM=256 + use_nt=False + no d=2048 injection"""
import os
os.environ["CU_NUM"] = "256"
os.environ["AITER_USE_NT"] = "0"

import torch
from typing import Dict, Tuple
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType
import aiter.fused_moe as fm

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

_patched = False
def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    # Force use_nt=False for ALL shapes
    fm.use_nt = lambda t, k, e: False

def custom_kernel(data: input_t) -> output_t:
    _patch()
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs)
