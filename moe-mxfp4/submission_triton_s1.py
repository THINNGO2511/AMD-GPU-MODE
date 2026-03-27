#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Try calling the Triton MoE mxfp4_silu kernel for stage1.
Read the wrapper API from aiter.ops.triton and try to invoke it.
"""
import torch
from task import input_t, output_t
import aiter
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
        import os, inspect

        # 1. Find ALL wrapper files for Triton MoE
        triton_dir = "/home/runner/aiter/aiter/ops/triton/"
        for f in sorted(os.listdir(triton_dir)):
            if ('moe' in f.lower() or 'fused' in f.lower()) and f.endswith('.py'):
                full = os.path.join(triton_dir, f)
                with open(full) as fh:
                    content = fh.read()
                funcs = [l.strip() for l in content.split('\n') if l.strip().startswith('def ') and not l.strip().startswith('def _')]
                if funcs:
                    print(f"\n[FILE] {f}: {len(content)} chars")
                    for func in funcs:
                        print(f"  {func}")

        # 2. Check fused_moe module for triton-related functions
        for name in sorted(dir(fm)):
            if 'triton' in name.lower() or 'mxfp4' in name.lower():
                obj = getattr(fm, name)
                if callable(obj):
                    try:
                        sig = inspect.signature(obj)
                        print(f"\n[FM] {name}{sig}")
                    except:
                        print(f"\n[FM] {name}")

        # 3. Try to import the Triton MoE wrapper
        try:
            from aiter.ops.triton import fused_moe as triton_fmoe
            print(f"\n[TRITON_FMOE] Found! Members:")
            for name in sorted(dir(triton_fmoe)):
                if not name.startswith('_'):
                    print(f"  {name}")
        except ImportError:
            print("\n[TRITON_FMOE] Not found as aiter.ops.triton.fused_moe")

        # 4. Search for fused_moe_mxfp4 or similar
        for mod_name in ['fused_moe', 'fused_moe_mxfp4', 'moe_mxfp4', 'moe']:
            try:
                mod = __import__(f'aiter.ops.triton.{mod_name}', fromlist=[mod_name])
                print(f"\n[MOD] aiter.ops.triton.{mod_name}:")
                for name in sorted(dir(mod)):
                    if not name.startswith('_'):
                        obj = getattr(mod, name)
                        if callable(obj):
                            try:
                                sig = inspect.signature(obj)
                                print(f"  {name}{sig}")
                            except:
                                print(f"  {name}")
            except ImportError:
                pass

        # 5. Look for the fused_moe_mxfp4_silu wrapper by searching file tree
        for root, dirs, files in os.walk("/home/runner/aiter/aiter/ops/triton/"):
            for f in files:
                if f.endswith('.py') and not f.startswith('_'):
                    full = os.path.join(root, f)
                    with open(full) as fh:
                        first_lines = fh.readlines()[:5]
                    for line in first_lines:
                        if 'moe' in line.lower() and 'import' not in line.lower():
                            print(f"\n[TREE] {full}: {line.strip()}")
                            break

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
