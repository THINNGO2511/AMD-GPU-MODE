#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe ALL available CK kernel .co files for stage1/stage2.
List every kernel name available in fmoe_2stages/ directory.
Then try injecting different tile sizes for E=33 d=512 cases.
"""
import os
import sys
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False
_patched = False

# All known kernel prefixes
S1_PREFIX = "moe_ck2stages_gemm1_"
S2_PREFIX = "moe_ck2stages_gemm2_"


def _probe_kernels():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. List ALL .co files in fmoe_2stages
    fmoe_dir = "/home/runner/aiter/hsa/gfx950/fmoe_2stages/"
    print(f"\n=== CK kernel .co files in {fmoe_dir} ===", file=sys.stderr)

    stage1_kernels = []
    stage2_kernels = []
    other_kernels = []

    if os.path.exists(fmoe_dir):
        for f in sorted(os.listdir(fmoe_dir)):
            if f.endswith('.co'):
                name = f[:-3]  # strip .co
                if 'gemm1' in name:
                    stage1_kernels.append(name)
                elif 'gemm2' in name:
                    stage2_kernels.append(name)
                else:
                    other_kernels.append(name)

    print(f"\nStage1 kernels ({len(stage1_kernels)}):", file=sys.stderr)
    for k in stage1_kernels:
        # Extract tile size from name
        parts = k.split('_')
        for i, p in enumerate(parts):
            if 'x' in p and p[0].isdigit():
                print(f"  {p}: {k}", file=sys.stderr)
                break
        else:
            print(f"  ???: {k}", file=sys.stderr)

    print(f"\nStage2 kernels ({len(stage2_kernels)}):", file=sys.stderr)
    for k in stage2_kernels:
        parts = k.split('_')
        for i, p in enumerate(parts):
            if 'x' in p and p[0].isdigit():
                print(f"  {p}: {k}", file=sys.stderr)
                break
        else:
            print(f"  ???: {k}", file=sys.stderr)

    if other_kernels:
        print(f"\nOther kernels ({len(other_kernels)}):", file=sys.stderr)
        for k in other_kernels[:10]:
            print(f"  {k}", file=sys.stderr)

    # 2. Check for FP4-specific kernels
    print(f"\n=== FP4/MXFP4 specific kernels ===", file=sys.stderr)
    fp4_s1 = [k for k in stage1_kernels if 'FP4' in k]
    fp4_s2 = [k for k in stage2_kernels if 'FP4' in k]
    print(f"Stage1 FP4: {len(fp4_s1)}", file=sys.stderr)
    for k in fp4_s1:
        print(f"  {k}", file=sys.stderr)
    print(f"Stage2 FP4: {len(fp4_s2)}", file=sys.stderr)
    for k in fp4_s2:
        print(f"  {k}", file=sys.stderr)

    # 3. Check dsv3 CSV for E=33 entries
    print(f"\n=== CSV entries for E=33 ===", file=sys.stderr)
    csv_path = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
    try:
        with open(csv_path) as f:
            header = f.readline().strip()
            print(f"Header: {header}", file=sys.stderr)
            for line in f:
                if ',33,' in line or 'expert' in line.lower():
                    print(f"  {line.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"  Error reading CSV: {e}", file=sys.stderr)

    # 4. Check tuned_fmoe.csv too
    print(f"\n=== tuned_fmoe.csv entries ===", file=sys.stderr)
    tuned_path = "/home/runner/aiter/aiter/configs/tuned_fmoe.csv"
    try:
        with open(tuned_path) as f:
            header = f.readline().strip()
            print(f"Header: {header}", file=sys.stderr)
            count = 0
            for line in f:
                count += 1
                if count <= 5 or ',33,' in line:
                    print(f"  {line.strip()}", file=sys.stderr)
            print(f"  Total rows: {count}", file=sys.stderr)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    # 5. Extract unique tile sizes from FP4 stage1 kernels
    print(f"\n=== Unique FP4 stage1 tile sizes ===", file=sys.stderr)
    tiles = set()
    for k in fp4_s1:
        parts = k.replace(S1_PREFIX, '').split('_')
        if parts and 'x' in parts[0]:
            tiles.add(parts[0])
    for t in sorted(tiles):
        matching = [k for k in fp4_s1 if t in k]
        print(f"  {t}: {len(matching)} variants", file=sys.stderr)
        for m in matching[:3]:
            print(f"    {m}", file=sys.stderr)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    fm.use_nt = lambda t, k, e: False

    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)


def custom_kernel(data: input_t) -> output_t:
    _probe_kernels()
    _patch()
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
