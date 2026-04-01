#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE Source Probe: Dump fused_moe_2stages source + all E=257 CSV entries + function signatures.
"""
import torch
import sys
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

    # 1. Dump fused_moe_2stages source (THE key function)
    for name in ['fused_moe_2stages', '_fused_moe_2stages', 'fused_moe_impl']:
        fn = getattr(fm, name, None)
        if fn is not None:
            try:
                src = inspect.getsource(fn)
                lines = src.split('\n')
                # Print in chunks to avoid truncation
                for i in range(0, min(len(lines), 120), 20):
                    chunk = '\n'.join(lines[i:i+20])
                    print(f"SRC_{name}_{i}:\n{chunk}", file=sys.stderr)
            except Exception as e:
                print(f"SRC_{name}_ERR: {e}", file=sys.stderr)
        else:
            print(f"SRC_{name}: NOT FOUND", file=sys.stderr)

    # 2. Dump fused_dynamic_mxfp4_quant_moe_sort source
    fn = getattr(fm, 'fused_dynamic_mxfp4_quant_moe_sort', None)
    if fn is not None:
        try:
            src = inspect.getsource(fn)
            lines = src.split('\n')
            for i in range(0, min(len(lines), 80), 20):
                chunk = '\n'.join(lines[i:i+20])
                print(f"SRC_fused_quant_sort_{i}:\n{chunk}", file=sys.stderr)
        except Exception as e:
            print(f"SRC_fused_quant_sort_ERR: {e}", file=sys.stderr)
    else:
        print(f"SRC_fused_quant_sort: NOT FOUND", file=sys.stderr)

    # 3. Dump moe_sorting source
    fn = getattr(fm, 'moe_sorting', None)
    if fn is not None:
        try:
            src = inspect.getsource(fn)
            print(f"SRC_moe_sorting:\n{src[:600]}", file=sys.stderr)
        except Exception as e:
            print(f"SRC_moe_sorting_ERR: {e}", file=sys.stderr)

    # 4. List ALL fm callables (one per line for grep)
    for n in sorted(dir(fm)):
        if n.startswith('__'):
            continue
        obj = getattr(fm, n, None)
        if callable(obj):
            print(f"FM_CALLABLE: {n} type={type(obj).__name__}", file=sys.stderr)

    # 5. Get ALL E=257 entries from dsv3 CSV
    try:
        import pandas as pd
        p = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
        df = pd.read_csv(p)
        e257 = df[df['expert'] == 257]
        print(f"E257_FULL ({len(e257)} rows):\n{e257.to_csv(index=False)}", file=sys.stderr)

        # E=33 entries
        e33 = df[df['expert'] == 33]
        if len(e33) > 0:
            print(f"E33_FULL ({len(e33)} rows):\n{e33.to_csv(index=False)}", file=sys.stderr)
    except Exception as e:
        print(f"CSV_ERR: {e}", file=sys.stderr)

    # 6. Get tuned_fmoe.csv (the merged one)
    try:
        import pandas as pd
        p = "/home/runner/aiter/aiter/configs/tuned_fmoe.csv"
        df = pd.read_csv(p)
        print(f"TUNED_CSV: {len(df)} rows, cols={list(df.columns)}", file=sys.stderr)
        # E=33 entries from tuned
        e33t = df[df['expert'] == 33] if 'expert' in df.columns else pd.DataFrame()
        if len(e33t) > 0:
            print(f"TUNED_E33 ({len(e33t)} rows):\n{e33t.head(5).to_csv(index=False)}", file=sys.stderr)
        # E=257 entries from tuned
        e257t = df[df['expert'] == 257] if 'expert' in df.columns else pd.DataFrame()
        if len(e257t) > 0:
            print(f"TUNED_E257 ({len(e257t)} rows):\n{e257t.head(5).to_csv(index=False)}", file=sys.stderr)
    except Exception as e:
        print(f"TUNED_CSV_ERR: {e}", file=sys.stderr)

    # 7. Check token_num_quant_moe_sort_switch
    for attr in ['token_num_quant_moe_sort_switch']:
        if hasattr(fm, attr):
            print(f"FM_ATTR: {attr} = {getattr(fm, attr)}", file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
    _probe()
    fm.use_nt = lambda token, topk, expert: False

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
