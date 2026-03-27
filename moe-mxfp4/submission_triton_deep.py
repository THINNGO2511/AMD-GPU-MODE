#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Deep probe aiter.ops.triton.moe module and its submodules.
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

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    if not _probed:
        _probed = True
        import os, inspect

        # 1. aiter.ops.triton.moe — all members including private
        try:
            import aiter.ops.triton.moe as triton_moe
            print(f"[MOE] {triton_moe.__file__}")
            for name in sorted(dir(triton_moe)):
                obj = getattr(triton_moe, name)
                if callable(obj) and not name.startswith('__'):
                    try:
                        sig = inspect.signature(obj)
                        print(f"  {name}{sig}")
                    except:
                        print(f"  {name}")
                elif not name.startswith('__'):
                    print(f"  {name} = {type(obj).__name__}")
        except Exception as e:
            print(f"[MOE] error: {e}")

        # 2. List the moe directory contents
        moe_dir = "/home/runner/aiter/aiter/ops/triton/moe/"
        if os.path.isdir(moe_dir):
            print(f"\n[DIR] {moe_dir}:")
            for f in sorted(os.listdir(moe_dir)):
                print(f"  {f}")
        else:
            # Check if it's a single file
            moe_file = "/home/runner/aiter/aiter/ops/triton/moe.py"
            if os.path.exists(moe_file):
                with open(moe_file) as fh:
                    lines = fh.readlines()
                print(f"\n[FILE] {moe_file}: {len(lines)} lines")
                for i, line in enumerate(lines[:80]):
                    print(f"  {i}: {line.rstrip()}")

        # 3. List all .py files in ops/triton/ (non-recursive)
        triton_dir = "/home/runner/aiter/aiter/ops/triton/"
        print(f"\n[LS] {triton_dir}:")
        for item in sorted(os.listdir(triton_dir)):
            full = os.path.join(triton_dir, item)
            if os.path.isdir(full):
                print(f"  {item}/ ({len(os.listdir(full))} files)")
            else:
                print(f"  {item}")

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
