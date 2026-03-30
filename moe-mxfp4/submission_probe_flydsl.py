"""Probe: Check if FlyDSL is available on the runner and what MoE configs exist."""
import torch
from task import input_t, output_t

def custom_kernel(data):
    import sys
    
    # Check FlyDSL
    try:
        import flydsl
        print(f"FlyDSL AVAILABLE: {flydsl.__version__ if hasattr(flydsl, '__version__') else 'yes'}")
    except ImportError:
        print("FlyDSL NOT available")
    
    # Check aiter FlyDSL integration
    try:
        from aiter.ops.flydsl.utils import is_flydsl_available
        print(f"is_flydsl_available: {is_flydsl_available()}")
    except Exception as e:
        print(f"FlyDSL utils import error: {e}")
    
    # Check sort optimization PR #2414
    try:
        from aiter.utility.fp4_utils import moe_mxfp4_sort
        import inspect
        src = inspect.getsource(moe_mxfp4_sort)
        if 'fused_n' in src or 'FUSED' in src:
            print("Sort optimization (PR #2414): FUSED variant detected")
        else:
            print(f"Sort kernel: standard (len={len(src)})")
    except Exception as e:
        print(f"Sort check error: {e}")
    
    # Check stage2 tune config fix PR #2438
    try:
        import os
        tune_files = []
        for root, dirs, files in os.walk('/home/runner/aiter/aiter/configs'):
            for f in files:
                if 'fmoe' in f.lower() or 'moe' in f.lower():
                    tune_files.append(os.path.join(root, f))
        print(f"MoE config files: {tune_files}")
        
        # Check if dsv3 config has E=256 entries
        for tf in tune_files:
            if 'dsv3' in tf or 'tuned' in tf:
                with open(tf) as fh:
                    content = fh.read()
                    e256_count = content.count('256,8')
                    e33_count = content.count(',33,')
                    print(f"  {os.path.basename(tf)}: E=256,k=8 entries={e256_count}, E=33 entries={e33_count}")
    except Exception as e:
        print(f"Config check error: {e}")
    
    # Run standard fused_moe as fallback
    import os
    os.environ["AITER_USE_NT"] = "0"
    from aiter.fused_moe import fused_moe
    from aiter import ActivationType, QuantType

    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
