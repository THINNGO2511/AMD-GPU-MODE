#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Probe: Find the MoE quant injection point for custom HIP kernel."""
import torch
import inspect
import os
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. Find quant-related functions in fused_moe module
    print("=== Quant functions in aiter.fused_moe ===", flush=True)
    for name in sorted(dir(fm)):
        if any(kw in name.lower() for kw in ['quant', 'mxfp4', 'sort', 'fused_dynamic']):
            obj = getattr(fm, name)
            print(f"  {name}: {type(obj).__name__}", flush=True)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    print(f"    sig: {sig}", flush=True)
                except:
                    pass
                try:
                    src_file = inspect.getfile(obj)
                    print(f"    file: {src_file}", flush=True)
                except:
                    pass

    # 2. Read fused_moe_2stages source — focus on quant path
    print("\n=== fused_moe_2stages quant path ===", flush=True)
    try:
        from aiter.fused_moe import fused_moe_2stages
        src = inspect.getsource(fused_moe_2stages)
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ['quant', 'token_num', 'sort', 'mxfp4']):
                start = max(0, i-1)
                end = min(len(lines), i+3)
                for j in range(start, end):
                    print(f"  {j}: {lines[j]}", flush=True)
                print("  ...", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # 3. Find fused_dynamic_mxfp4_quant_moe_sort
    print("\n=== fused_dynamic_mxfp4_quant_moe_sort ===", flush=True)
    try:
        from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
        src = inspect.getsource(fused_dynamic_mxfp4_quant_moe_sort)
        print(src[:3000], flush=True)
    except ImportError:
        print("  Not found as direct import", flush=True)
        try:
            obj = getattr(fm, 'fused_dynamic_mxfp4_quant_moe_sort', None)
            if obj:
                src = inspect.getsource(obj)
                print(src[:3000], flush=True)
            else:
                # Search in module
                for name in dir(fm):
                    obj = getattr(fm, name)
                    if callable(obj) and 'quant' in str(obj).lower():
                        print(f"  Found: {name} = {obj}", flush=True)
        except Exception as e2:
            print(f"  ERROR: {e2}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # 4. Check if moe_mxfp4_sort exists separately
    print("\n=== moe_mxfp4_sort ===", flush=True)
    try:
        obj = getattr(fm, 'moe_mxfp4_sort', None)
        if obj:
            src = inspect.getsource(obj)
            print(src[:2000], flush=True)
        else:
            print("  Not found", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # 5. Check _moe_sorting_impl
    print("\n=== _moe_sorting_impl ===", flush=True)
    try:
        obj = getattr(fm, '_moe_sorting_impl', None)
        if obj:
            src = inspect.getsource(obj)
            print(src[:2000], flush=True)
        else:
            print("  Not found", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)


def custom_kernel(data: input_t) -> output_t:
    _probe()
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data
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
