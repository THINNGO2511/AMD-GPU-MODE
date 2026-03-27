#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Probe v2: Dump full source of quant kernel, test tl.dot_scaled,
try Triton GEMM with different dtypes. Falls back to baseline for correctness.
"""
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

    import os, inspect, importlib

    # 1. Dump quant.py - the FULL file (we need the @triton.jit kernel)
    quant_path = "/home/runner/aiter/aiter/ops/triton/quant.py"
    if os.path.exists(quant_path):
        with open(quant_path) as f:
            lines = f.readlines()
        print(f"=== quant.py ({len(lines)} lines) ===")
        for i, line in enumerate(lines):
            print(f"{i+1:4d}: {line}", end='')
        print("=== END quant.py ===")
    else:
        # Try to find it via inspect
        print(f"quant.py not at expected path, trying inspect...")
        try:
            src = inspect.getsource(dynamic_mxfp4_quant)
            print(f"=== dynamic_mxfp4_quant source ===")
            print(src)
            print("=== END ===")
        except Exception as e:
            print(f"inspect failed: {e}")

    # 2. Dump the Triton GEMM kernel source
    triton_gemm_paths = [
        "/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py",
        "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py",
    ]
    for p in triton_gemm_paths:
        if os.path.exists(p):
            with open(p) as f:
                lines = f.readlines()
            print(f"\n=== {os.path.basename(p)} from {p} ({len(lines)} lines) ===")
            # Print first 200 lines (kernel definition)
            for i, line in enumerate(lines[:300]):
                print(f"{i+1:4d}: {line}", end='')
            if len(lines) > 300:
                print(f"... ({len(lines) - 300} more lines)")
            print(f"=== END ===")
            break
    else:
        print("\nTriton GEMM kernel not found at expected paths")
        # Search for it
        for root, dirs, files in os.walk("/home/runner/aiter/aiter/ops/triton"):
            for f in files:
                if 'fp4' in f.lower() or 'afp4' in f.lower():
                    print(f"  Found: {os.path.join(root, f)}")

    # 3. Test tl.dot_scaled availability
    print("\n=== Testing tl.dot_scaled ===")
    try:
        import triton
        import triton.language as tl
        print(f"Triton version: {triton.__version__}")
        print(f"Has dot_scaled: {hasattr(tl, 'dot_scaled')}")

        # Check Triton dtype support
        for attr in ['float4_e2m1fn', 'float4_e2m1fn_x2', 'float8_e8m0fn',
                     'float8_e8m0fnu', 'int4', 'uint8', 'float8e4nv']:
            print(f"  tl.{attr}: {hasattr(tl, attr)}")
    except Exception as e:
        print(f"Triton test failed: {e}")

    # 4. Test calling gemm_afp4wfp4 with different dtype approaches
    print("\n=== Testing Triton GEMM imports ===")
    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        print("gemm_afp4wfp4 imported OK")
        sig = inspect.signature(gemm_afp4wfp4)
        print(f"Signature: {sig}")

        # Also check preshuffle variants
        try:
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffled_scales
            print("gemm_afp4wfp4_preshuffled_scales imported OK")
            sig2 = inspect.signature(gemm_afp4wfp4_preshuffled_scales)
            print(f"Signature: {sig2}")
        except ImportError:
            print("gemm_afp4wfp4_preshuffled_scales not found")

        try:
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
            print("gemm_afp4wfp4_preshuffle imported OK")
            sig3 = inspect.signature(gemm_afp4wfp4_preshuffle)
            print(f"Signature: {sig3}")
        except ImportError:
            print("gemm_afp4wfp4_preshuffle not found")
    except Exception as e:
        print(f"Import failed: {e}")

    # 5. Check the wrapper (gemm_op_a4w4.py) for the actual dispatch
    wrapper_path = "/home/runner/aiter/aiter/ops/gemm_op_a4w4.py"
    if os.path.exists(wrapper_path):
        with open(wrapper_path) as f:
            src = f.read()
        print(f"\n=== gemm_op_a4w4.py ({len(src.splitlines())} lines) ===")
        for i, line in enumerate(src.splitlines()[:150]):
            print(f"{i+1:4d}: {line}")
        print("=== END ===")

    # 6. List all available FP4 GEMM kernels/functions
    print("\n=== Available aiter GEMM functions ===")
    for attr in dir(aiter):
        if 'gemm' in attr.lower() or 'a4w4' in attr.lower() or 'fp4' in attr.lower():
            print(f"  aiter.{attr}")

    # 7. Check if there's a fused quant+gemm function
    print("\n=== Searching for fused functions ===")
    for attr in dir(aiter):
        if 'fuse' in attr.lower() or 'quant_gemm' in attr.lower() or 'gemm_quant' in attr.lower():
            print(f"  aiter.{attr}")
            try:
                sig = inspect.signature(getattr(aiter, attr))
                print(f"    Signature: {sig}")
            except:
                pass


def custom_kernel(data: input_t) -> output_t:
    _probe()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape

    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)

    return aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
