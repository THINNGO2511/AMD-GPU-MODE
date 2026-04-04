#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Read the full source of fused_dynamic_mxfp4_quant_moe_sort to understand the injection point."""
import torch
import inspect
from task import input_t, output_t
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    # Read the full quant source
    print("=== fused_dynamic_mxfp4_quant_moe_sort source ===", flush=True)
    try:
        obj = getattr(fm, 'fused_dynamic_mxfp4_quant_moe_sort', None)
        if obj is None:
            from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
            obj = fused_dynamic_mxfp4_quant_moe_sort
        src = inspect.getsource(obj)
        print(src, flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

    # Also read the separate quant function
    print("\n=== dynamic_mxfp4_quant source ===", flush=True)
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        src = inspect.getsource(dynamic_mxfp4_quant)
        print(src[:3000], flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

    # Read fused_moe_2stages lines around quant call
    print("\n=== fused_moe_2stages full source ===", flush=True)
    try:
        from aiter.fused_moe import fused_moe_2stages
        src = inspect.getsource(fused_moe_2stages)
        lines = src.split('\n')
        for i, line in enumerate(lines):
            print(f"  {i}: {line}", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)


def custom_kernel(data: input_t) -> output_t:
    _probe()
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
