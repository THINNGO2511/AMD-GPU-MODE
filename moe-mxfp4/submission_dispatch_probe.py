#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe dispatch logic: how to trigger Triton MoE path + read DSv3 FP4 tuned configs.
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

        # 1. Read dsv3_fp4_tuned_fmoe.csv
        try:
            path = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
            with open(path) as f:
                lines = f.readlines()
            print(f"[DSV3] {path}: {len(lines)} lines")
            for line in lines:
                print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"[DSV3] error: {e}")

        # 2. Read the FULL fused_moe_ function to understand dispatch
        try:
            with open(fm.__file__) as f:
                src_lines = f.readlines()

            # Print entire fused_moe_ function
            in_func = False
            for i, line in enumerate(src_lines):
                if 'def fused_moe_(' in line:
                    in_func = True
                if in_func:
                    print(f"[FM:{i}] {line.rstrip()}")
                    # Detect end of function (next def at same indent)
                    if i > 0 and in_func and line.startswith('def ') and 'fused_moe_' not in line:
                        break
                    if i > 500:  # safety limit
                        print("[FM] ... truncated at line 500")
                        break
        except Exception as e:
            print(f"[FM] error: {e}")

        # 3. Read get_2stage_cfgs function
        try:
            import inspect
            src = inspect.getsource(fm.get_2stage_cfgs)
            print(f"\n[CFG_FN] get_2stage_cfgs:")
            for i, line in enumerate(src.split('\n')):
                print(f"  {i}: {line}")
        except Exception as e:
            print(f"[CFG_FN] error: {e}")

        # 4. Read moe_op_mxfp4_silu_fused.py (has fused SiLU activation)
        try:
            path2 = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_mxfp4_silu_fused.py"
            with open(path2) as f:
                lines2 = f.readlines()
            print(f"\n[SILU_FUSED] {path2}: {len(lines2)} lines")
            for i, line in enumerate(lines2):
                print(f"  {i}: {line.rstrip()}")
        except Exception as e:
            print(f"[SILU_FUSED] error: {e}")

        # 5. Check for Triton MoE wrappers (non-kernel files)
        import os
        triton_dir = "/home/runner/aiter/aiter/ops/triton/"
        for f in sorted(os.listdir(triton_dir)):
            if 'moe' in f.lower() and f.endswith('.py'):
                full = os.path.join(triton_dir, f)
                print(f"\n[WRAPPER] {full}")
                with open(full) as fh:
                    wlines = fh.readlines()
                print(f"  {len(wlines)} lines, first 50:")
                for wl in wlines[:50]:
                    print(f"  {wl.rstrip()}")

        # 6. Look for cktile_moe functions (may have Triton dispatch)
        try:
            src = inspect.getsource(fm.cktile_moe_stage1)
            print(f"\n[CKTILE1] cktile_moe_stage1:")
            for i, line in enumerate(src.split('\n')[:30]):
                print(f"  {i}: {line}")
        except Exception as e:
            print(f"[CKTILE1] error: {e}")

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
