#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe the Triton MoE MXFP4 kernel.
Find moe_op_mxfp4_silu_fused.py and understand how to call it.
Also explore if there are Python wrappers already available.
"""
import torch
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False
_patched = False


def _probe():
    global _probed
    if _probed: return
    _probed = True

    # 1. Search for Triton MoE kernel wrapper
    print("[TRITON] Searching for Triton MoE wrappers:")
    try:
        # Check aiter namespace
        for name in sorted(dir(aiter)):
            if any(k in name.lower() for k in ['triton_moe', 'mxfp4_moe', 'fused_moe_triton',
                                                 'moe_mxfp4', 'moe_triton', 'fmoe_triton']):
                obj = getattr(aiter, name)
                try:
                    sig = inspect.signature(obj)
                    print(f"  aiter.{name}{sig}")
                except:
                    print(f"  aiter.{name}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. Check fused_moe module for triton-related functions
    print("\n[FM] Triton-related in fused_moe module:")
    try:
        for name in sorted(dir(fm)):
            if any(k in name.lower() for k in ['triton', 'asm_stage', 'fmoe_g1u1', '1stage']):
                obj = getattr(fm, name)
                try:
                    sig = inspect.signature(obj)
                    print(f"  fm.{name}{sig}")
                except:
                    print(f"  fm.{name} = {type(obj).__name__}: {obj}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3. Read the Triton MoE kernel file
    print("\n[TRITON SRC] moe_op_mxfp4_silu_fused.py:")
    try:
        import glob
        files = glob.glob("/home/runner/aiter/**/moe_op_mxfp4*", recursive=True)
        print(f"  Files found: {files}")
        for f in files[:1]:
            with open(f) as fh:
                lines = fh.readlines()
            print(f"  {f} ({len(lines)} lines)")
            # Print all function/class defs and the last 50 lines (launcher)
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') or line.strip().startswith('class ') or line.strip().startswith('@triton'):
                    print(f"    {i}: {line.rstrip()}")
            print(f"\n  Last 80 lines (launcher):")
            for i in range(max(0, len(lines)-80), len(lines)):
                print(f"    {i}: {lines[i].rstrip()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 4. Check fused_moe_1stage_dict
    print("\n[1STAGE] Available 1-stage configs:")
    try:
        d = fm.fused_moe_1stage_dict
        for gfx, configs in d.items():
            print(f"  {gfx}: {len(configs)} configs")
            for cfg in sorted(configs):
                print(f"    {cfg}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 5. Check dispatch_policy options
    print("\n[DISPATCH] moe_sorting dispatch_policy:")
    try:
        src = inspect.getsource(fm._moe_sorting_impl)
        for line in src.split('\n'):
            if 'dispatch' in line.lower():
                print(f"  {line.strip()}")
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
