import os, torch
os.environ["AITER_USE_NT"] = "0"
from typing import Dict, Tuple
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, Dict]
output_t = torch.Tensor

_probed = False
def custom_kernel(data):
    global _probed
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    
    if not _probed:
        _probed = True
        # List ONLY FP4X2 kernels (compact output)
        import glob
        base = "/home/runner/aiter/hsa/gfx950/fmoe_2stages/"
        all_co = sorted(glob.glob(base + "*.co"))
        fp4 = [os.path.basename(f) for f in all_co if "FP4X2" in f]
        print(f"TOTAL .co files: {len(all_co)}")
        print(f"FP4X2 kernels: {len(fp4)}")
        for k in fp4:
            # Parse tile size
            parts = k.split("_")
            for p in parts:
                if "x" in p and p[0].isdigit():
                    print(f"  {p} | {k[:80]}")
                    break
        
        # Check what default config is used for E=33 d=2048
        import aiter.fused_moe as fm
        try:
            src = fm.get_2stage_cfgs.__wrapped__  # unwrap lru_cache
        except:
            pass
        
        # Check CSV for E=33
        import csv
        for csvf in glob.glob("/home/runner/aiter/aiter/configs/*fmoe*.csv"):
            with open(csvf) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('expert','') == '33' and 'per_1x32' in row.get('q_type',''):
                        print(f"CSV E=33 MXFP4: {os.path.basename(csvf)} | bm={row.get('block_m','')} | s1={row.get('kernelName1','')[:40]} | s2={row.get('kernelName2','')[:40]}")
    
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs)
