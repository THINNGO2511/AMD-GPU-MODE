#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Read moe_op_gemm_a4w4.py Triton MoE kernel + fused_moe_ per_1x32 path.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

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

        # 1. Read moe_op_gemm_a4w4.py (the existing Triton MXFP4 MoE kernel!)
        path = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a4w4.py"
        try:
            with open(path) as f:
                lines = f.readlines()
            print(f"[A4W4] {path}: {len(lines)} lines")
            for i, line in enumerate(lines):
                print(f"  {i}: {line.rstrip()}")
        except Exception as e:
            print(f"[A4W4] error: {e}")

        # 2. Read fused_moe_ complete source (focus on per_1x32 / triton path)
        try:
            with open(fm.__file__) as f:
                src_lines = f.readlines()
            # Find and print the entire fused_moe_ function
            in_func = False
            func_start = -1
            for i, line in enumerate(src_lines):
                if 'def fused_moe_(' in line:
                    in_func = True
                    func_start = i
                elif in_func and line.strip() and not line[0].isspace() and i > func_start + 1:
                    # End of function
                    break
            if func_start >= 0:
                end = i
                print(f"\n[SRC] fused_moe_ lines {func_start}-{end}:")
                for j in range(func_start, end):
                    print(f"  {j}: {src_lines[j].rstrip()}")
        except Exception as e:
            print(f"[SRC] error: {e}")

        # 3. Check for Triton MoE wrapper functions
        triton_moe_dir = "/home/runner/aiter/aiter/ops/triton/"
        import os
        for root, dirs, files in os.walk(triton_moe_dir):
            for f in files:
                if 'moe' in f.lower() and f.endswith('.py') and not f.startswith('_'):
                    full = os.path.join(root, f)
                    print(f"\n[WRAPPER] {full}")
                    with open(full) as fh:
                        wlines = fh.readlines()
                    print(f"  {len(wlines)} lines")
                    for wl in wlines[:40]:
                        print(f"  {wl.rstrip()}")

        # 4. Look for triton moe launcher/wrapper
        for name in sorted(dir(fm)):
            if 'triton' in name.lower() or 'a4w4' in name.lower() or 'gemm_a4' in name.lower():
                print(f"\n[FM] {name}")

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
