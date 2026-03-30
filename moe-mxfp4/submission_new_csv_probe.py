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
        # Check new CSV entries
        for csvf in glob.glob("/home/runner/aiter/aiter/configs/*fmoe*.csv") + glob.glob("/home/runner/aiter/aiter/configs/model_configs/*fmoe*.csv"):
            with open(csvf) as f:
                content = f.read()
            e33_count = content.count(',33,')
            e33_2048 = content.count(',2048,33,') + content.count(',33,9,') 
            if e33_count > 0:
                print(f"CSV:{csvf.split('/')[-1]}: E=33 rows={e33_count} E33+d2048 hits={e33_2048}")
                # Show first E=33 row
                for line in content.split('\n'):
                    if ',33,' in line and 'per_1x32' in line:
                        print(f"  SAMPLE: {line[:150]}")
                        break
        # Also check what the current call produces for E=33 d=2048
        E = topk_ids.max().item() + 1
        print(f"Current call: E={E}, d_expert={config['d_expert']}")
    fm.use_nt = lambda t,k,e: True if config.get('d_expert',0) >= 2048 else False
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
