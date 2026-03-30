import os
os.environ['CU_NUM'] = '256'
os.environ['AITER_USE_NT'] = '0'
import torch, glob
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType
from typing import Dict, Tuple
input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor
_p = False
def custom_kernel(data):
    global _p
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    if not _p:
        _p = True
        # Show ALL E=33 rows from tuned_fmoe.csv
        with open("/home/runner/aiter/aiter/configs/tuned_fmoe.csv") as f:
            lines = f.readlines()
        print(f"Total rows: {len(lines)}")
        header = lines[0].strip()
        print(f"Header: {header[:100]}")
        e33_rows = [l for l in lines[1:] if ',33,' in l]
        print(f"E=33 rows: {len(e33_rows)}")
        for row in e33_rows[:5]:
            print(f"  ROW: {row.strip()[:200]}")
    fm.use_nt = lambda t,k,e: False
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
