import os
os.environ["CU_NUM"] = "256"
os.environ["AITER_USE_NT"] = "0"

import torch
from typing import Dict, Tuple
input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

_fm = None
def _get_fm():
    global _fm
    if _fm is None:
        from aiter.fused_moe import fused_moe as _fused_moe
        _fm = _fused_moe
    return _fm

def custom_kernel(data: input_t) -> output_t:
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    return _get_fm()(hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
                     topk_weights, topk_ids, **config)
