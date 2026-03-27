#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Read config files and probe all internal functions.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_probed = False


def custom_kernel(data: input_t) -> output_t:
    global _probed

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    if not _probed:
        _probed = True
        import os

        # Read tuned config
        for fname in ['tuned_fmoe.csv', 'untuned_fmoe.csv']:
            path = f"/home/runner/aiter/aiter/configs/{fname}"
            if os.path.exists(path):
                with open(path) as f:
                    lines = f.readlines()
                print(f"\n[CFG] {fname}: {len(lines)} lines")
                print(f"  Header: {lines[0].strip()}")
                for line in lines[1:8]:
                    print(f"  {line.strip()}")
                print(f"  ...")
                for line in lines[-3:]:
                    print(f"  {line.strip()}")

        # List ALL fused_moe module contents
        import aiter.fused_moe as fm
        print(f"\n[MOD] ALL fused_moe dir:")
        for name in sorted(dir(fm)):
            obj = getattr(fm, name, None)
            if callable(obj) and not name.startswith('__'):
                print(f"  {name}: {type(obj).__name__}")

        # Check fused_moe source file
        try:
            print(f"\n[SRC] fused_moe file: {fm.__file__}")
            with open(fm.__file__) as f:
                src = f.read()
            # Find function/class definitions
            import re
            defs = re.findall(r'^(?:def|class)\s+(\w+)', src, re.MULTILINE)
            print(f"[SRC] Definitions: {defs[:30]}")
            # Find moe_ck2stages references
            ck_refs = re.findall(r'(moe_ck\w+|ck2stages\w*)', src)
            print(f"[SRC] CK refs: {list(set(ck_refs))}")
            # Find sorting references
            sort_refs = re.findall(r'(moe_sort\w+|sorting\w*)', src)
            print(f"[SRC] Sort refs: {list(set(sort_refs))}")
        except Exception as e:
            print(f"[SRC] Error: {e}")

        # Print config info
        print(f"\n[DATA] config keys: {list(config.keys())}")
        print(f"[DATA] topk_ids shape: {topk_ids.shape}, topk_weights shape: {topk_weights.shape}")
        print(f"[DATA] w1_shuffled shape: {gate_up_weight_shuffled.shape}")
        print(f"[DATA] w1_scale_shuffled shape: {gate_up_weight_scale_shuffled.shape}")

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
