#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE Probe: Count moe_sorting calls, dump CSV format, find sort function names.
"""
import torch
import sys
import inspect
from task import input_t, output_t
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

_call_counts = {}
_patched = False
_probed = False


def _probe_once():
    global _probed
    if _probed:
        return
    _probed = True

    # List all callable attributes
    attrs = []
    for n in sorted(dir(fm)):
        if n.startswith('__'):
            continue
        obj = getattr(fm, n, None)
        if callable(obj):
            attrs.append(n)
    print(f"PROBE: fm callables: {attrs}", file=sys.stderr)

    # Check specific sort-related functions
    for name in ['moe_sorting', '_moe_sorting_impl', 'moe_mxfp4_sort',
                  'fused_dynamic_mxfp4_quant_moe_sort', 'fused_moe_2stages',
                  '_fused_moe_2stages', 'moe_sorting_opus_fwd']:
        if hasattr(fm, name):
            fn = getattr(fm, name)
            print(f"PROBE: fm.{name} EXISTS type={type(fn).__name__}", file=sys.stderr)
            try:
                src = inspect.getsource(fn)
                # Print first 800 chars of source
                print(f"PROBE_SRC {name}:\n{src[:800]}", file=sys.stderr)
            except Exception as e:
                print(f"PROBE: {name} source error: {e}", file=sys.stderr)
        else:
            print(f"PROBE: fm.{name} MISSING", file=sys.stderr)

    # Check token_num_quant_moe_sort_switch
    for attr in ['token_num_quant_moe_sort_switch', 'TOKEN_NUM_QUANT_MOE_SORT_SWITCH']:
        if hasattr(fm, attr):
            print(f"PROBE: fm.{attr} = {getattr(fm, attr)}", file=sys.stderr)

    # Dump CSV
    try:
        import pandas as pd
        for p in ["/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
                  "/home/runner/aiter/aiter/configs/model_configs/tuned_fmoe.csv"]:
            try:
                df = pd.read_csv(p)
                print(f"PROBE_CSV {p}: {len(df)} rows", file=sys.stderr)
                print(f"PROBE_CSV cols: {list(df.columns)}", file=sys.stderr)
                print(f"PROBE_CSV dtypes:\n{df.dtypes}", file=sys.stderr)
                print(f"PROBE_CSV head(2):\n{df.head(2).to_csv(index=False)}", file=sys.stderr)
                e257 = df[df['expert'] == 257] if 'expert' in df.columns else pd.DataFrame()
                if len(e257) > 0:
                    print(f"PROBE_CSV E=257 ({len(e257)} rows) first 2:\n{e257.head(2).to_csv(index=False)}", file=sys.stderr)
                e33 = df[df['expert'] == 33] if 'expert' in df.columns else pd.DataFrame()
                if len(e33) > 0:
                    print(f"PROBE_CSV E=33 ({len(e33)} rows) first 2:\n{e33.head(2).to_csv(index=False)}", file=sys.stderr)
            except Exception as e:
                print(f"PROBE_CSV {p} error: {e}", file=sys.stderr)
    except ImportError:
        print("PROBE: pandas not available", file=sys.stderr)


def _patch_counters():
    global _patched
    if _patched:
        return
    _patched = True

    # Patch ALL sort-related functions to count calls
    for name in ['moe_sorting', '_moe_sorting_impl', 'moe_mxfp4_sort',
                  'fused_dynamic_mxfp4_quant_moe_sort']:
        if hasattr(fm, name) and callable(getattr(fm, name)):
            orig = getattr(fm, name)
            fn_name = name  # capture

            def make_counter(orig_fn, fn_name):
                def counted(*args, **kwargs):
                    _call_counts[fn_name] = _call_counts.get(fn_name, 0) + 1
                    if _call_counts[fn_name] <= 4:  # Only log first few
                        nargs = len(args)
                        arg_shapes = []
                        for i, a in enumerate(args[:5]):
                            if hasattr(a, 'shape'):
                                arg_shapes.append(f"arg{i}={a.shape}")
                            else:
                                arg_shapes.append(f"arg{i}={a}")
                        print(f"PROBE_CALL {fn_name} #{_call_counts[fn_name]}: "
                              f"nargs={nargs} {' '.join(arg_shapes)}", file=sys.stderr)
                    return orig_fn(*args, **kwargs)
                return counted

            setattr(fm, name, make_counter(orig, fn_name))

    fm.use_nt = lambda token, topk, expert: False


def custom_kernel(data: input_t) -> output_t:
    global _call_counts
    _call_counts = {}
    _probe_once()
    _patch_counters()

    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    result = fused_moe(
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

    print(f"PROBE_RESULT: bs={hidden_states.shape[0]} "
          f"E={topk_ids.shape[1] if topk_ids.dim()>1 else '?'} "
          f"calls={dict(_call_counts)}", file=sys.stderr)

    return result
