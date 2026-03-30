#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
Probe: Read full get_2stage_cfgs fallback logic + fused_moe_2stages rest + untuned CSV.
Focus on what happens for E=33 cases and the default/fallback kernel selection.
"""
import torch
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False

def _probe():
    global _probed
    if _probed: return
    _probed = True

    # 1. get_2stage_cfgs lines 75-200 (the critical fallback logic)
    print("=" * 60)
    try:
        orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else fm.get_2stage_cfgs
        src = inspect.getsource(orig)
        lines = src.split('\n')
        print(f"[get_2stage_cfgs] total lines: {len(lines)}")
        for i in range(75, min(200, len(lines))):
            print(f"  {i}: {lines[i]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. fused_moe_2stages lines 77-150
    print("\n[fused_moe_2stages] lines 77-150:")
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        lines = src.split('\n')
        print(f"  total lines: {len(lines)}")
        for i in range(77, min(170, len(lines))):
            print(f"  {i}: {lines[i]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3. untuned_fmoe.csv header + first 20 lines
    print("\n[untuned_fmoe.csv]:")
    try:
        with open("/home/runner/aiter/aiter/configs/untuned_fmoe.csv") as f:
            for i, line in enumerate(f):
                if i < 30:
                    print(f"  {i}: {line.rstrip()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 4. gfx950 MXFP4 1-stage CSV
    print("\n[gfx950 fmoe_bf16_pertokenMXfp4_g1u1_silu.csv]:")
    try:
        with open("/home/runner/aiter/hsa/gfx950/fmoe/silu/fmoe_bf16_pertokenMXfp4_g1u1_silu.csv") as f:
            for i, line in enumerate(f):
                print(f"  {i}: {line.rstrip()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 5. AITER_CONFIGS
    print("\n[AITER_CONFIGS]:")
    try:
        from aiter.fused_moe import AITER_CONFIGS
        print(f"  AITER_CONFIG_FMOE_FILE: {AITER_CONFIGS.AITER_CONFIG_FMOE_FILE}")
        for attr in dir(AITER_CONFIGS):
            if not attr.startswith('_'):
                print(f"  {attr}: {getattr(AITER_CONFIGS, attr, 'N/A')}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 6. original get_block_size_M and use_nt from the source file
    print("\n[original functions from fused_moe.py]:")
    try:
        fpath = inspect.getfile(fm)
        with open(fpath) as f:
            content = f.read()
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def use_nt(' in line or 'def get_block_size_M(' in line or 'def get_padded_M(' in line:
                for j in range(i, min(i + 25, len(lines))):
                    print(f"  {j}: {lines[j]}")
                    if j > i and lines[j].strip() and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                        break
                print()
    except Exception as e:
        print(f"  ERROR: {e}")

    # 7. _moe_sorting_impl (full)
    print("\n[_moe_sorting_impl]:")
    try:
        src = inspect.getsource(fm._moe_sorting_impl)
        for i, line in enumerate(src.split('\n')):
            if i < 60:
                print(f"  {i}: {line}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("=" * 60)


def custom_kernel(data: input_t) -> output_t:
    _probe()
    (hidden_states, gate_up_weight, down_weight, gate_up_weight_scale, down_weight_scale, gate_up_weight_shuffled, down_weight_shuffled, gate_up_weight_scale_shuffled, down_weight_scale_shuffled, topk_weights, topk_ids, config) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled, topk_weights, topk_ids, expert_mask=None, activation=ActivationType.Silu, quant_type=QuantType.per_1x32, doweight_stage1=False, w1_scale=gate_up_weight_scale_shuffled, w2_scale=down_weight_scale_shuffled, a1_scale=None, a2_scale=None, hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
