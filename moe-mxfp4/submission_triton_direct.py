#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe for Triton MoE wrapper functions + try calling them.
Find the Python-level API that invokes _fused_moe_kernel_mxfp4_silu.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_probed = False


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    original_get_block_size_M = fm.get_block_size_M
    original_use_nt = fm.use_nt

    def patched_get_block_size_M(token, topk, expert, inter_dim):
        est_m = token * topk // expert
        if expert <= 64:
            if est_m < 50:
                return 32
            else:
                return 64
        return original_get_block_size_M(token, topk, expert, inter_dim)

    def patched_use_nt(token, topk, expert):
        if expert <= 64:
            return False
        return original_use_nt(token, topk, expert)

    fm.get_block_size_M = patched_get_block_size_M
    fm.use_nt = patched_use_nt
    fm.get_2stage_cfgs.cache_clear()
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    global _probed
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

    if not _probed:
        _probed = True
        import os, inspect

        # 1. Find Triton MoE wrapper files
        triton_dir = "/home/runner/aiter/aiter/ops/triton/"
        for f in sorted(os.listdir(triton_dir)):
            if 'moe' in f.lower() and f.endswith('.py'):
                full = os.path.join(triton_dir, f)
                with open(full) as fh:
                    content = fh.read()
                # Find function definitions
                funcs = [l.strip() for l in content.split('\n') if l.strip().startswith('def ')]
                print(f"[WRAP] {f}: {len(content)} chars, functions:")
                for func in funcs:
                    print(f"    {func}")

        # 2. Search for triton MoE functions in aiter namespace
        import aiter
        print(f"\n[AITER] Triton/MoE functions:")
        for name in sorted(dir(aiter)):
            if 'triton' in name.lower() or ('moe' in name.lower() and 'fused' not in name.lower()):
                obj = getattr(aiter, name, None)
                if callable(obj):
                    try:
                        sig = inspect.signature(obj)
                        print(f"  aiter.{name}{sig}")
                    except:
                        print(f"  aiter.{name}")

        # 3. Search for triton MoE in ops.triton namespace
        try:
            import aiter.ops.triton as aot
            print(f"\n[AOT] aiter.ops.triton members:")
            for name in sorted(dir(aot)):
                if 'moe' in name.lower():
                    print(f"  {name}")
        except:
            pass

        # 4. Try importing specific modules
        for mod_name in ['fused_moe_mxfp4', 'moe_mxfp4', 'triton_moe', 'moe_triton']:
            try:
                mod = __import__(f'aiter.ops.triton.{mod_name}', fromlist=[mod_name])
                print(f"\n[MOD] aiter.ops.triton.{mod_name}:")
                for name in sorted(dir(mod)):
                    if not name.startswith('_'):
                        print(f"  {name}")
            except ImportError:
                pass

        # 5. Read the fused_moe_2stages source to find where Triton path could be triggered
        try:
            src = inspect.getsource(fm.fused_moe_2stages)
            # Find lines with 'triton' or 'mxfp4' or 'fp4'
            for i, line in enumerate(src.split('\n')):
                if any(kw in line.lower() for kw in ['triton', 'mxfp4', 'fp4', 'quant', 'scale']):
                    print(f"[2S:{i}] {line.rstrip()}")
        except Exception as e:
            print(f"[2S] error: {e}")

        # 6. Read the full fused_moe_2stages source
        try:
            src = inspect.getsource(fm.fused_moe_2stages)
            print(f"\n[2STAGE] Full source ({len(src)} chars):")
            for i, line in enumerate(src.split('\n')):
                print(f"  {i}: {line}")
        except Exception as e:
            print(f"[2STAGE] error: {e}")

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
