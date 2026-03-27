#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Test ASM 1-stage kernels for E=33 cases.
The gfx950 has assembly MoE kernels: fmoe_bf16_pertokenMXfp4_g1u1_*_silu_*
These might be faster for small expert counts by fusing all operations.
Try using them via run_1stage=True config injection.
"""
import torch
import functools
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_probed = False


def _probe():
    global _probed
    if _probed: return
    _probed = True

    # 1. Check what 1-stage configs are available for our combo
    print("[1STAGE] Checking fused_moe_1stage_dict:")
    try:
        from aiter.fused_moe import fused_moe_1stage_dict, get_gfx
        gfx = get_gfx()
        print(f"  gfx: {gfx}")
        if gfx in fused_moe_1stage_dict:
            configs = fused_moe_1stage_dict[gfx]
            print(f"  Available 1-stage combos ({len(configs)}):")
            for cfg in sorted(configs):
                print(f"    {cfg}")
        else:
            print(f"  No 1-stage configs for {gfx}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. Check asm_stage1 function
    print("\n[ASM] asm_stage1:")
    try:
        from aiter.fused_moe import asm_stage1
        sig = inspect.signature(asm_stage1)
        print(f"  {sig}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3. Check if there's a fused_moe_1stage function
    print("\n[FUSED1] Looking for 1-stage entry points:")
    for name in dir(fm):
        if '1stage' in name.lower() or 'fmoe_g1u1' in name.lower():
            obj = getattr(fm, name)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    print(f"  fm.{name}{sig}")
                except:
                    print(f"  fm.{name}")
            else:
                print(f"  fm.{name} = {obj}")

    # 4. Read the ASM kernel CSV to find MXfp4 entries
    print("\n[CSV] gfx950 MXfp4 1-stage kernels:")
    try:
        import glob
        csvs = glob.glob("/home/runner/aiter/hsa/gfx950/fmoe/silu/*.csv")
        for f in sorted(csvs):
            if 'MXfp4' in f or 'mxfp4' in f.lower():
                print(f"  File: {f}")
                with open(f) as fh:
                    for line in fh:
                        print(f"    {line.rstrip()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 5. Check fused_moe source for fmoe_g1u1 dispatching
    print("\n[SRC] fused_moe g1u1/1stage dispatch:")
    try:
        src = inspect.getsource(fm.fused_moe)
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if 'g1u1' in line.lower() or '1stage' in line.lower() or 'run_1stage' in line.lower():
                # Print context
                for j in range(max(0, i-1), min(len(lines), i+3)):
                    print(f"  {j}: {lines[j]}")
                print()
    except Exception as e:
        print(f"  ERROR: {e}")

    # 6. Check get_ksplit source
    print("\n[SRC] get_ksplit:")
    try:
        src = inspect.getsource(fm.get_ksplit)
        for i, line in enumerate(src.split('\n')):
            print(f"  {i}: {line}")
    except Exception as e:
        print(f"  ERROR: {e}")


def _patch():
    global _patched
    if _patched: return
    _patched = True

    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)


def custom_kernel(data: input_t) -> output_t:
    _patch()
    _probe()
    (hidden_states, gate_up_weight, down_weight, gate_up_weight_scale, down_weight_scale, gate_up_weight_shuffled, down_weight_shuffled, gate_up_weight_scale_shuffled, down_weight_scale_shuffled, topk_weights, topk_ids, config) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled, topk_weights, topk_ids, expert_mask=None, activation=ActivationType.Silu, quant_type=QuantType.per_1x32, doweight_stage1=False, w1_scale=gate_up_weight_scale_shuffled, w2_scale=down_weight_scale_shuffled, a1_scale=None, a2_scale=None, hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
