#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Probe: dump the actual _dynamic_mxfp4_quant_kernel Triton source."""
from task import input_t, output_t
import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    import inspect, os

    # Try to get the kernel source via inspect
    try:
        from aiter.ops.triton import quant as quant_module
        for name in dir(quant_module):
            obj = getattr(quant_module, name)
            if hasattr(obj, 'fn') and 'quant' in name.lower():
                print(f"\n=== {name} (has .fn) ===")
                try:
                    src = inspect.getsource(obj.fn)
                    print(src[:3000])
                    if len(src) > 3000:
                        print(f"... ({len(src) - 3000} more chars)")
                except Exception as e:
                    print(f"getsource failed: {e}")
            elif callable(obj) and 'kernel' in name.lower():
                print(f"\n=== {name} (callable) ===")
                try:
                    src = inspect.getsource(obj)
                    print(src[:3000])
                except:
                    pass
    except Exception as e:
        print(f"Module inspection failed: {e}")

    # Try finding the source file
    try:
        mod_file = inspect.getfile(quant_module)
        print(f"\nQuant module file: {mod_file}")
        if os.path.exists(mod_file):
            with open(mod_file) as f:
                src = f.read()
            lines = src.splitlines()
            print(f"Total lines: {len(lines)}")
            # Print lines containing the kernel
            in_kernel = False
            for i, line in enumerate(lines):
                if '_dynamic_mxfp4_quant_kernel' in line and '@' not in line and 'def' not in line:
                    continue
                if 'def _dynamic_mxfp4_quant_kernel' in line or '@triton.jit' in line:
                    in_kernel = True
                if in_kernel:
                    print(f"{i+1:4d}: {line}")
                    if line.strip().startswith('def ') and 'dynamic_mxfp4_quant' in line and '_kernel' not in line:
                        break
                    if i > 0 and line.strip().startswith('def ') and in_kernel and 'kernel' not in line.lower():
                        break
    except Exception as e:
        print(f"File inspection failed: {e}")

    # Also try the _triton_kernels path
    for path in [
        "/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant.py",
        "/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant/quant.py",
        "/home/runner/aiter/aiter/ops/triton/quant.py",
    ]:
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            print(f"\n=== {path} ({len(content.splitlines())} lines) ===")
            # Print lines around the kernel
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if '_dynamic_mxfp4_quant_kernel' in line or 'e2m1' in line.lower() or 'fp4' in line.lower():
                    start = max(0, i - 2)
                    end = min(len(lines), i + 30)
                    for j in range(start, end):
                        print(f"{j+1:4d}: {lines[j]}")
                    print("---")


def custom_kernel(data: input_t) -> output_t:
    _probe()
    A, B, B_q, B_shuffle, B_scale_sh = data
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale).view(dtypes.fp8_e8m0)
    return aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)
