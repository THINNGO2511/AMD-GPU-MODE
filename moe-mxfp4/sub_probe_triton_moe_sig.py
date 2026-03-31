#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Probe the exact signatures of aiter Triton MoE kernels."""
import inspect
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm


def custom_kernel(data: input_t) -> output_t:
    # Probe Triton MoE kernel signatures
    try:
        from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
        print(f"=== fused_moe_mxfp4_silu ===")
        try:
            sig = inspect.signature(fused_moe_mxfp4_silu)
            print(f"  sig: {sig}")
        except:
            pass
        try:
            src = inspect.getsource(fused_moe_mxfp4_silu)
            # Print first 60 lines (function def + args)
            for i, line in enumerate(src.split('\n')[:60]):
                print(f"  L{i}: {line}")
        except Exception as e:
            print(f"  source error: {e}")
    except ImportError as e:
        print(f"  import error: {e}")

    try:
        from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4
        print(f"\n=== fused_moe_mxfp4 ===")
        try:
            sig = inspect.signature(fused_moe_mxfp4)
            print(f"  sig: {sig}")
        except:
            pass
        try:
            src = inspect.getsource(fused_moe_mxfp4)
            for i, line in enumerate(src.split('\n')[:60]):
                print(f"  L{i}: {line}")
        except Exception as e:
            print(f"  source error: {e}")
    except ImportError as e:
        print(f"  import error: {e}")

    # Also list all functions in the moe module
    try:
        import aiter.ops.triton.moe as moe_mod
        print(f"\n=== aiter.ops.triton.moe contents ===")
        for name in sorted(dir(moe_mod)):
            if not name.startswith('_'):
                obj = getattr(moe_mod, name)
                print(f"  {name}: {type(obj).__name__}")
    except Exception as e:
        print(f"  error: {e}")

    # Fallback
    (hidden_states, *_, topk_weights, topk_ids, config) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    fm.use_nt = lambda t, k, e: False
    return fused_moe(
        hidden_states, data[5], data[6], topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=data[7], w2_scale=data[8],
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
