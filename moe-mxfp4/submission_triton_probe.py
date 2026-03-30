"""Probe: What Triton MoE kernels does fused_moe actually use for MXFP4?"""
import os, torch, time
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
        import aiter.fused_moe as fm
        import inspect
        
        # Check what Triton MoE files exist
        import glob
        triton_moe = glob.glob("/home/runner/aiter/aiter/ops/triton/moe/*.py")
        print(f"Triton MoE files: {[os.path.basename(f) for f in triton_moe]}")
        
        # Check what functions fused_moe_ calls
        try:
            src = inspect.getsource(fm.fused_moe_)
            # Find kernel dispatch lines
            for line in src.split('\n'):
                if 'ck_moe' in line or 'triton' in line.lower() or 'stage1' in line or 'stage2' in line:
                    print(f"  fused_moe_: {line.strip()[:100]}")
        except:
            print("Could not inspect fused_moe_")
        
        # Check block_m for current shape
        E = topk_ids.shape[1]  
        bs = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        print(f"Shape: bs={bs}, E={E}, topk={topk}")
        print(f"block_m={fm.get_block_size_M(bs, topk, E, w1_qw.shape[1])}")
        
        # Time the current call
        torch.cuda.synchronize()
        t0 = time.time()
        result = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                          expert_mask=None, activation=ActivationType.Silu,
                          quant_type=QuantType.per_1x32, doweight_stage1=False,
                          w1_scale=w1_qs, w2_scale=w2_qs)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"Time: {(t1-t0)*1e6:.0f}us")
        return result
    
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs)
