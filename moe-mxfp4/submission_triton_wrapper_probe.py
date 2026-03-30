#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
Probe: Read the Triton MoE kernel WRAPPER files to understand calling interface.
These are at /home/runner/aiter/aiter/ops/triton/moe/moe_op_mxfp4*.py
"""
import torch
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

    # 1. Read the Triton MoE wrapper (moe_op_mxfp4.py)
    print("=" * 60)
    print("[WRAPPER] moe_op_mxfp4.py:")
    try:
        with open("/home/runner/aiter/aiter/ops/triton/moe/moe_op_mxfp4.py") as f:
            for i, line in enumerate(f):
                print(f"  {i}: {line.rstrip()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. Read the fused SiLU wrapper
    print("\n[WRAPPER] moe_op_mxfp4_silu_fused.py:")
    try:
        with open("/home/runner/aiter/aiter/ops/triton/moe/moe_op_mxfp4_silu_fused.py") as f:
            for i, line in enumerate(f):
                print(f"  {i}: {line.rstrip()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3. Check if these are importable
    print("\n[IMPORT] Trying to import:")
    try:
        from aiter.ops.triton.moe import moe_op_mxfp4
        print(f"  moe_op_mxfp4: {dir(moe_op_mxfp4)}")
    except Exception as e:
        print(f"  moe_op_mxfp4 import error: {e}")
    try:
        from aiter.ops.triton.moe import moe_op_mxfp4_silu_fused
        print(f"  moe_op_mxfp4_silu_fused: {dir(moe_op_mxfp4_silu_fused)}")
    except Exception as e:
        print(f"  moe_op_mxfp4_silu_fused import error: {e}")

    print("=" * 60)

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
