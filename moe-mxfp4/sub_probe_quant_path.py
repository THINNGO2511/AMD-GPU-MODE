#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe the quantization path internals.
Find the exact variable names for token_num_quant_moe_sort_switch
and the quant function used in fused_moe_2stages.
"""
import torch
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    # Probe fused_moe module attributes
    print("=== fm module attrs ===")
    for attr in sorted(dir(fm)):
        if 'quant' in attr.lower() or 'switch' in attr.lower() or 'token' in attr.lower() or 'thresh' in attr.lower():
            val = getattr(fm, attr, '?')
            print(f"  fm.{attr} = {val}")

    # Probe fused_moe_2stages source
    if hasattr(fm, 'fused_moe_2stages'):
        fn = fm.fused_moe_2stages
        print(f"\n=== fused_moe_2stages type: {type(fn)} ===")
        try:
            src = inspect.getsource(fn)
            # Find quant-related lines
            for i, line in enumerate(src.split('\n')):
                if any(kw in line.lower() for kw in ['quant', 'switch', 'thresh', 'token_num']):
                    print(f"  L{i}: {line.rstrip()}")
        except Exception as e:
            print(f"  source error: {e}")

    # Probe get_torch_quant
    print("\n=== get_torch_quant ===")
    try:
        tq = aiter.get_torch_quant(QuantType.per_1x32)
        print(f"  type: {type(tq)}, callable: {callable(tq)}")
    except Exception as e:
        print(f"  error: {e}")

    # Probe get_triton_quant
    print("\n=== get_triton_quant ===")
    try:
        from aiter import get_triton_quant
        trq = get_triton_quant(QuantType.per_1x32)
        print(f"  type: {type(trq)}, callable: {callable(trq)}")
    except Exception as e:
        print(f"  error: {e}")

    # Look for fused_dynamic_mxfp4_quant_moe_sort
    print("\n=== fused_dynamic_mxfp4_quant_moe_sort ===")
    for attr in ['fused_dynamic_mxfp4_quant_moe_sort', '_fused_quant_moe_sort']:
        if hasattr(fm, attr):
            fn = getattr(fm, attr)
            print(f"  fm.{attr} = {type(fn)}")
            try:
                sig = inspect.signature(fn)
                print(f"  signature: {sig}")
            except:
                pass

    # Look for _moe_sorting_impl
    print("\n=== _moe_sorting_impl ===")
    if hasattr(fm, '_moe_sorting_impl'):
        try:
            src = inspect.getsource(fm._moe_sorting_impl)
            print(src[:500])
        except Exception as e:
            print(f"  error: {e}")

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
