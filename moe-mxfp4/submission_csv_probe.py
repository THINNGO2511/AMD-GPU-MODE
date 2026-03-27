"""Probe: dump the actual CSV format from runner"""
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

_probed = False
def custom_kernel(data: input_t) -> output_t:
    global _probed
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    
    if not _probed:
        _probed = True
        # Dump CSV headers and first few rows
        for csvf in glob.glob("/home/runner/aiter/aiter/configs/*fmoe*.csv") + glob.glob("/home/runner/aiter/aiter/configs/model_configs/*fmoe*.csv"):
            with open(csvf) as f:
                lines = f.readlines()
            print(f"FILE: {csvf} ({len(lines)} lines)")
            print(f"  HEADER: {lines[0].strip()}")
            # Show first FP4 entry if any
            for line in lines[1:5]:
                print(f"  ROW: {line.strip()[:200]}")
        
        # Check AITER_CONFIG_FMOE env var
        print(f"AITER_CONFIG_FMOE: {os.environ.get('AITER_CONFIG_FMOE', 'NOT SET')}")
        
        # Check get_2stage_cfgs source for CSV parsing
        import inspect
        try:
            src = inspect.getsource(fm.get_2stage_cfgs)
            # Find the CSV column parsing
            for line in src.split('\n'):
                if 'csv' in line.lower() or 'config' in line.lower() or 'read' in line.lower() or 'column' in line.lower():
                    print(f"  GET_2STAGE: {line.strip()[:150]}")
        except:
            pass
    
    fm.use_nt = lambda t,k,e: False
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
