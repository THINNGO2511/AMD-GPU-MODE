#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe how to invoke aiter's Triton MoE a4w4 kernel.
Read the launcher/wrapper files + moe_op_mxfp4_silu_fused.py.
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

        # 1. Read moe_op_mxfp4_silu_fused.py (smaller, SiLU-specific)
        try:
            path = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_mxfp4_silu_fused.py"
            with open(path) as f:
                lines = f.readlines()
            print(f"[SILU] {len(lines)} lines:")
            for i, line in enumerate(lines):
                print(f"  {i}: {line.rstrip()}")
        except Exception as e:
            print(f"[SILU] error: {e}")

        # 2. Read moe_op_mxfp4.py (base MXFP4 kernel)
        try:
            path2 = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_mxfp4.py"
            with open(path2) as f:
                lines2 = f.readlines()
            print(f"\n[MXFP4] {len(lines2)} lines:")
            for i, line in enumerate(lines2):
                print(f"  {i}: {line.rstrip()}")
        except Exception as e:
            print(f"[MXFP4] error: {e}")

        # 3. Look for wrapper/launcher functions
        import os
        triton_dir = "/home/runner/aiter/aiter/ops/triton/"
        for f in sorted(os.listdir(triton_dir)):
            if 'moe' in f.lower() and f.endswith('.py'):
                full = os.path.join(triton_dir, f)
                with open(full) as fh:
                    content = fh.read()
                print(f"\n[WRAP] {f}: {len(content)} chars")
                # Print first 80 lines
                for i, line in enumerate(content.split('\n')[:80]):
                    print(f"  {i}: {line}")

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
