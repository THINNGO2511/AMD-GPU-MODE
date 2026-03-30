"""Probe: list ALL FP4X2 kernel names available in fmoe_2stages CSVs"""
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
        # List FP4 kernel names from CSVs
        for f in sorted(glob.glob("/home/runner/aiter/aiter/configs/*fmoe*.csv") + glob.glob("/home/runner/aiter/aiter/configs/model_configs/*fmoe*.csv")):
            with open(f) as fh:
                for line in fh:
                    if 'FP4X2' in line:
                        parts = line.strip().split(',')
                        for p in parts:
                            if 'moe_ck' in p or 'flydsl' in p:
                                print(f"K:{os.path.basename(f)}:{p[:80]}")
                        break
        # Also list CK .co files with FP4
        cos = glob.glob("/home/runner/aiter/hsa/gfx950/fmoe_2stages/*FP4*")
        print(f"FP4 .co files: {len(cos)}")
        for c in sorted(cos)[:10]:
            print(f"CO:{os.path.basename(c)[:80]}")
    fm.use_nt = lambda t,k,e: False
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
