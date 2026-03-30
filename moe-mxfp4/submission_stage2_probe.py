#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Probe: List all stage2 FP4 kernels + get_2stage_cfgs source."""
import torch
import sys
import os
import inspect
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

    # 1. get_2stage_cfgs FULL source
    try:
        orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else fm.get_2stage_cfgs
        src = inspect.getsource(orig)
        lines = src.split('\n')
        print(f"\n=== get_2stage_cfgs ({len(lines)} lines) ===", file=sys.stderr)
        for i, line in enumerate(lines):
            print(f"  {i:3d}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"get_2stage_cfgs error: {e}", file=sys.stderr)

    # 2. get_ksplit FULL source
    try:
        orig_ks = fm.get_ksplit.__wrapped__ if hasattr(fm.get_ksplit, '__wrapped__') else fm.get_ksplit
        src2 = inspect.getsource(orig_ks)
        lines2 = src2.split('\n')
        print(f"\n=== get_ksplit ({len(lines2)} lines) ===", file=sys.stderr)
        for i, line in enumerate(lines2):
            print(f"  {i:3d}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"get_ksplit error: {e}", file=sys.stderr)

    # 3. All stage2 FP4 kernel binaries
    try:
        dirs = [
            "/home/runner/aiter/hsa/gfx950/fmoe_2stages",
            "/home/runner/aiter/hsa/gfx950/fmoe",
        ]
        for d in dirs:
            if os.path.isdir(d):
                all_files = sorted(os.listdir(d))
                s2 = [f for f in all_files if 'gemm2' in f and 'FP4' in f]
                s1 = [f for f in all_files if 'gemm1' in f and 'FP4' in f]
                print(f"\n=== {d} (total={len(all_files)}, s1_fp4={len(s1)}, s2_fp4={len(s2)}) ===", file=sys.stderr)
                print(f"\nStage1 FP4 kernels:", file=sys.stderr)
                for f in s1:
                    print(f"  {f}", file=sys.stderr)
                print(f"\nStage2 FP4 kernels:", file=sys.stderr)
                for f in s2:
                    print(f"  {f}", file=sys.stderr)
    except Exception as e:
        print(f"kernel listing error: {e}", file=sys.stderr)

    # 4. DSv3 CSV E=33 entries
    try:
        csv_path = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
        with open(csv_path) as f:
            lines = f.readlines()
        e33 = [l for l in lines if ',33,' in l]
        print(f"\n=== DSv3 CSV E=33 entries ({len(e33)}) ===", file=sys.stderr)
        for l in e33:
            print(f"  {l.rstrip()}", file=sys.stderr)
    except Exception as e:
        print(f"CSV error: {e}", file=sys.stderr)

    # 5. MOEMetadata fields
    try:
        print(f"\n=== MOEMetadata fields ===", file=sys.stderr)
        print(f"  type: {type(fm.MOEMetadata)}", file=sys.stderr)
        if hasattr(fm.MOEMetadata, '_fields'):
            print(f"  fields: {fm.MOEMetadata._fields}", file=sys.stderr)
        elif hasattr(fm.MOEMetadata, '__init__'):
            sig = inspect.signature(fm.MOEMetadata.__init__)
            print(f"  __init__ params: {list(sig.parameters.keys())}", file=sys.stderr)
        # Also try getting source
        src_meta = inspect.getsource(fm.MOEMetadata)
        print(f"  source ({len(src_meta.split(chr(10)))} lines):", file=sys.stderr)
        for i, line in enumerate(src_meta.split('\n')[:20]):
            print(f"    {i}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"MOEMetadata error: {e}", file=sys.stderr)

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
