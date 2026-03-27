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

_p = False
def custom_kernel(data):
    global _p
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    if not _p:
        _p = True
        # Print JUST the CSV header + first FP4 row
        import glob
        for f in sorted(glob.glob("/home/runner/aiter/aiter/configs/*fmoe*.csv")):
            with open(f) as fh:
                h = fh.readline().strip()
                print(f"H:{os.path.basename(f)}:{h[:200]}")
                for line in fh:
                    if 'per_1x32' in line or 'FP4' in line.lower():
                        print(f"R:{line.strip()[:200]}")
                        break
        for f in sorted(glob.glob("/home/runner/aiter/aiter/configs/model_configs/*fmoe*.csv")):
            with open(f) as fh:
                h = fh.readline().strip()
                print(f"H:{os.path.basename(f)}:{h[:200]}")
                for line in fh:
                    if 'per_1x32' in line:
                        print(f"R:{line.strip()[:200]}")
                        break
    fm.use_nt = lambda t,k,e: False
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
