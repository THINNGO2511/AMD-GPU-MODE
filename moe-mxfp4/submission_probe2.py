#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Deep probe: find internal MoE functions and tuning options.
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
        import aiter
        import aiter.fused_moe as fm
        import inspect

        # List all public functions in fused_moe module
        print(f"[PROBE] fused_moe module members:")
        for name in sorted(dir(fm)):
            if not name.startswith('_'):
                obj = getattr(fm, name)
                if callable(obj):
                    try:
                        sig = inspect.signature(obj)
                        print(f"  {name}{sig}")
                    except:
                        print(f"  {name} (no sig)")

        # Check for MoE-specific config files
        import os, glob
        config_dir = "/home/runner/aiter/aiter/configs/"
        if os.path.exists(config_dir):
            moe_files = glob.glob(config_dir + "*moe*") + glob.glob(config_dir + "*fmoe*")
            print(f"\n[PROBE] MoE config files: {moe_files}")
            for f in moe_files:
                with open(f) as fh:
                    lines = fh.readlines()
                    print(f"  {f}: {len(lines)} lines, header: {lines[0].strip() if lines else 'empty'}")
                    # Show a few entries
                    for line in lines[1:4]:
                        print(f"    {line.strip()}")

        # Check for moe_sorting functions
        try:
            from aiter import moe_sorting_fwd
            sig = inspect.signature(moe_sorting_fwd)
            print(f"\n[PROBE] moe_sorting_fwd{sig}")
        except ImportError:
            print("\n[PROBE] moe_sorting_fwd not directly importable")

        # Check aiter top-level MoE functions
        print(f"\n[PROBE] aiter MoE-related:")
        for name in sorted(dir(aiter)):
            if 'moe' in name.lower() or 'sort' in name.lower():
                print(f"  aiter.{name}")

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    output = fused_moe(
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
    return output
