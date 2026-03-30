#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Probe: Dump fused_moe_2stages source, check threshold, and benchmark."""
import torch
import sys
import inspect
import textwrap
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. Dump fused_moe_2stages source
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        lines = src.split('\n')
        print(f"\n[SRC] fused_moe_2stages ({len(lines)} lines):", file=sys.stderr)
        for i, line in enumerate(lines):
            print(f"  {i:3d}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] fused_moe_2stages error: {e}", file=sys.stderr)

    # 2. Dump fused_dynamic_mxfp4_quant_moe_sort source
    try:
        src2 = inspect.getsource(fm.fused_dynamic_mxfp4_quant_moe_sort)
        lines2 = src2.split('\n')
        print(f"\n[SRC] fused_dynamic_mxfp4_quant_moe_sort ({len(lines2)} lines):", file=sys.stderr)
        for i, line in enumerate(lines2):
            print(f"  {i:3d}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] fused_dynamic_mxfp4_quant_moe_sort error: {e}", file=sys.stderr)

    # 3. Dump _moe_sorting_impl source
    try:
        src3 = inspect.getsource(fm._moe_sorting_impl)
        lines3 = src3.split('\n')
        print(f"\n[SRC] _moe_sorting_impl ({len(lines3)} lines):", file=sys.stderr)
        for i, line in enumerate(lines3):
            print(f"  {i:3d}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] _moe_sorting_impl error: {e}", file=sys.stderr)

    # 4. Check OPUS sorting
    try:
        import aiter
        has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
        use_opus = getattr(fm, '_USE_OPUS_MOE_SORTING', 'unset')
        print(f"\n[OPUS] has_opus={has_opus}, _USE_OPUS_MOE_SORTING={use_opus}", file=sys.stderr)
    except Exception as e:
        print(f"[OPUS] error: {e}", file=sys.stderr)

    # 5. Check moe_mxfp4_sort signature
    try:
        from aiter.utility.fp4_utils import moe_mxfp4_sort
        src4 = inspect.getsource(moe_mxfp4_sort)
        lines4 = src4.split('\n')
        print(f"\n[SRC] moe_mxfp4_sort ({len(lines4)} lines):", file=sys.stderr)
        for i, line in enumerate(lines4[:30]):
            print(f"  {i:3d}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] moe_mxfp4_sort error: {e}", file=sys.stderr)

    # 6. Check get_ksplit
    try:
        src5 = inspect.getsource(fm.get_ksplit)
        lines5 = src5.split('\n')
        print(f"\n[SRC] get_ksplit ({len(lines5)} lines):", file=sys.stderr)
        for i, line in enumerate(lines5):
            print(f"  {i:3d}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] get_ksplit error: {e}", file=sys.stderr)

    # 7. List all available stage2 FP4 kernels
    try:
        import os
        fmoe_dir = "/home/runner/aiter/hsa/gfx950/fmoe_2stages"
        if os.path.isdir(fmoe_dir):
            s2_files = [f for f in os.listdir(fmoe_dir) if 'gemm2' in f and 'FP4' in f]
            print(f"\n[KERNELS] Stage2 FP4 kernels ({len(s2_files)}):", file=sys.stderr)
            for f in sorted(s2_files):
                print(f"  {f}", file=sys.stderr)
    except Exception as e:
        print(f"[KERNELS] error: {e}", file=sys.stderr)

    sys.stderr.flush()


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
