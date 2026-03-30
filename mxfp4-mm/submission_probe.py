#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Probe available CK kernels and aiter internals."""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    import os
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape
    n = B.shape[0]

    # List available f4gemm kernel binaries
    f4gemm_dir = "/home/runner/aiter/hsa/gfx950/f4gemm"
    if os.path.exists(f4gemm_dir):
        kernels = sorted(os.listdir(f4gemm_dir))
        print(f"Available f4gemm kernels ({len(kernels)}):")
        for k_name in kernels:
            print(f"  {k_name}")
    else:
        print(f"f4gemm dir not found at {f4gemm_dir}")
        # Try other paths
        for base in ["/home/runner/aiter/hsa", "/home/runner/aiter"]:
            if os.path.exists(base):
                print(f"Contents of {base}:")
                for item in sorted(os.listdir(base)):
                    print(f"  {item}")

    # Also check what get_GEMM_config returns
    try:
        from aiter.ops.gemm_op_a4w4 import get_GEMM_config
        for test_m, test_n, test_k in [(4,2880,512), (16,2112,7168), (256,3072,1536)]:
            config = get_GEMM_config(test_m, test_n, test_k)
            print(f"Config for M={test_m}, N={test_n}, K={test_k}: {config}")
    except Exception as e:
        print(f"get_GEMM_config error: {e}")

    # Standard baseline for correctness
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)

    out = aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
    return out
