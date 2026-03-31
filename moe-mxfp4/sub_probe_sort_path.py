#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe: find the separate moe sort function path.
Need to know where moe_mxfp4_sort or equivalent lives.
"""
import torch
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
import inspect


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    # Find sort-related functions
    print("=== aiter sort functions ===")
    for name in sorted(dir(aiter)):
        if 'sort' in name.lower() or 'mxfp4' in name.lower():
            obj = getattr(aiter, name, None)
            print(f"  aiter.{name} = {type(obj)}")

    print("\n=== fm sort functions ===")
    for name in sorted(dir(fm)):
        if 'sort' in name.lower() or 'mxfp4' in name.lower() or 'quant' in name.lower():
            obj = getattr(fm, name, None)
            print(f"  fm.{name} = {type(obj)}")

    # Check aiter.ops.triton.quant contents
    print("\n=== aiter.ops.triton.quant ===")
    try:
        import aiter.ops.triton.quant as tq
        for name in sorted(dir(tq)):
            if not name.startswith('_'):
                print(f"  {name}")
    except Exception as e:
        print(f"  error: {e}")

    # Check fused_dynamic_mxfp4_quant_moe_sort source
    print("\n=== fused_dynamic_mxfp4_quant_moe_sort source ===")
    try:
        src = inspect.getsource(fm.fused_dynamic_mxfp4_quant_moe_sort)
        print(src[:2000])
    except Exception as e:
        print(f"  error: {e}")

    # Check separate quant function return types
    print("\n=== get_torch_quant result ===")
    try:
        tqf = aiter.get_torch_quant(QuantType.per_1x32)
        print(f"  type: {type(tqf)}, name: {getattr(tqf, '__name__', '?')}")
        try:
            sig = inspect.signature(tqf)
            print(f"  sig: {sig}")
        except:
            pass
        # Try calling on small tensor
        x = torch.randn(4, 256, dtype=torch.bfloat16, device='cuda')
        out = tqf(x, quant_dtype=aiter.dtypes.fp4x2)
        print(f"  output: {type(out)}, len={len(out)}")
        for i, o in enumerate(out):
            print(f"    [{i}]: shape={o.shape}, dtype={o.dtype}")
    except Exception as e:
        print(f"  error: {e}")

    # Look for moe_sorting in various places
    print("\n=== moe_sorting search ===")
    for mod_name in ['aiter', 'aiter.fused_moe', 'aiter.ops']:
        try:
            mod = __import__(mod_name, fromlist=[''])
            for name in dir(mod):
                if 'moe' in name.lower() and ('sort' in name.lower() or 'mxfp4' in name.lower()):
                    print(f"  {mod_name}.{name}")
        except:
            pass

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
