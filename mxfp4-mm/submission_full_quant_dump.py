#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Dump the FULL quant source file (all 272 lines)."""
from task import input_t, output_t
import torch, aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True
    import os
    path = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant/quant.py"
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
        print(f"=== FULL quant.py ({len(lines)} lines) ===")
        for i, line in enumerate(lines):
            print(f"{i+1:4d}: {line}", end='')
        print("\n=== END ===")

    # Also check imports available
    try:
        from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
        print(f"\n_mxfp4_quant_op imported OK")
        import inspect
        sig = inspect.signature(_mxfp4_quant_op.fn)
        print(f"Signature: {sig}")
    except Exception as e:
        print(f"Import failed: {e}")

    # Check the outer kernel
    path2 = "/home/runner/aiter/aiter/ops/triton/quant/__init__.py"
    if os.path.exists(path2):
        with open(path2) as f:
            lines = f.readlines()
        print(f"\n=== quant/__init__.py ({len(lines)} lines) ===")
        for i, line in enumerate(lines):
            print(f"{i+1:4d}: {line}", end='')
        print("\n=== END ===")

def custom_kernel(data: input_t) -> output_t:
    _probe()
    A, B, B_q, B_shuffle, B_scale_sh = data
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale).view(dtypes.fp8_e8m0)
    return aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)
