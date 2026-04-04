#!/usr/bin/env python3
"""
MoE Quant Probe — Find exact injection point for fused_dynamic_mxfp4_quant_moe_sort
Submit as: popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark moe-mxfp4/moe_quant_probe.py --no-tui
"""
import torch
import functools
import inspect
import subprocess

print("=" * 60)
print("PROBE: MoE quant function injection point")
print("=" * 60)

import aiter.fused_moe as fm

# 1. Find all quant-related functions
print("\n=== Quant-related functions in fused_moe ===")
for name in sorted(dir(fm)):
    if any(q in name.lower() for q in ['quant', 'mxfp4', 'sort', 'fused_dynamic']):
        obj = getattr(fm, name)
        print(f"\n{name}: {type(obj)}")
        if callable(obj):
            try:
                sig = inspect.signature(obj)
                print(f"  sig: {sig}")
                src = inspect.getsource(obj)
                lines = src.split('\n')[:30]
                print(f"  source ({len(src)} chars, {len(src.split(chr(10)))} lines):")
                for line in lines:
                    print(f"    {line}")
            except Exception as e:
                print(f"  could not inspect: {e}")

# 2. Find where fused_dynamic_mxfp4_quant_moe_sort is called
print("\n\n=== GREP: where is fused_dynamic_mxfp4_quant_moe_sort called? ===")
result = subprocess.run(
    ['grep', '-n', 'fused_dynamic_mxfp4_quant_moe_sort', '/home/runner/aiter/aiter/fused_moe.py'],
    capture_output=True, text=True
)
print(result.stdout)

# 3. Find ALL references across the codebase
result = subprocess.run(
    ['grep', '-rn', 'fused_dynamic_mxfp4_quant_moe_sort', '/home/runner/aiter/aiter/'],
    capture_output=True, text=True
)
print("\n=== All references ===")
print(result.stdout)

# 4. Check if it's Triton JIT or C++ op
result = subprocess.run(
    ['grep', '-n', 'triton.jit\|torch.ops\|def fused_dynamic', '/home/runner/aiter/aiter/fused_moe.py'],
    capture_output=True, text=True
)
print("\n=== Kernel type indicators ===")
print(result.stdout)

# 5. Print _fused_moe_2stages (where quant is likely called)
print("\n=== _fused_moe_2stages source (first 80 lines) ===")
try:
    src = inspect.getsource(fm._fused_moe_2stages)
    for i, line in enumerate(src.split('\n')[:80]):
        print(f"  {i}: {line}")
except Exception as e:
    print(f"  Error: {e}")

# 6. Check if moe_mxfp4_sort exists separately
print("\n=== Sort functions ===")
for name in ['moe_mxfp4_sort', 'moe_sort', '_moe_sort', 'moe_sorting_fwd']:
    has = hasattr(fm, name)
    print(f"  fm.{name}: {'EXISTS' if has else 'not found'}")
    if has:
        obj = getattr(fm, name)
        try:
            print(f"    sig: {inspect.signature(obj)}")
        except:
            pass

# 7. Check the Triton quant kernel file
print("\n=== Triton quant kernel location ===")
result = subprocess.run(
    ['find', '/home/runner/aiter/aiter/ops/triton/', '-name', '*quant*'],
    capture_output=True, text=True
)
print(result.stdout)

# 8. Print the actual quant kernel source
result = subprocess.run(
    ['grep', '-n', 'def fused_dynamic_mxfp4_quant_moe_sort', '/home/runner/aiter/aiter/ops/triton/quant.py'],
    capture_output=True, text=True
)
print(f"\n=== Quant kernel definition ===")
print(result.stdout)

# If found in quant.py, print the function
if result.stdout.strip():
    line_num = int(result.stdout.split(':')[1]) if ':' in result.stdout else 0
    if line_num > 0:
        result2 = subprocess.run(
            ['sed', '-n', f'{line_num},{line_num+80}p', '/home/runner/aiter/aiter/ops/triton/quant.py'],
            capture_output=True, text=True
        )
        print(result2.stdout)

# 9. Check token_num_quant_moe_sort_switch
result = subprocess.run(
    ['grep', '-n', 'token_num_quant_moe_sort_switch\|quant_moe_sort_switch', '/home/runner/aiter/aiter/fused_moe.py'],
    capture_output=True, text=True
)
print(f"\n=== quant_moe_sort_switch references ===")
print(result.stdout)

# 10. Check what separate quant functions exist
print("\n=== Separate quant functions ===")
try:
    from aiter.ops.triton import quant as triton_quant
    for name in sorted(dir(triton_quant)):
        if 'quant' in name.lower() or 'mxfp4' in name.lower():
            print(f"  triton_quant.{name}: {type(getattr(triton_quant, name))}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("PROBE COMPLETE")
print("=" * 60)

# Custom kernel for eval harness
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

def custom_kernel(hidden_states, w1, w2, topk_weights, topk_ids,
                  w1_scale=None, w2_scale=None, num_experts=None):
    return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                     w1_scale=w1_scale, w2_scale=w2_scale,
                     activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32)
