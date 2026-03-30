#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Fused quant+shuffle using aiter's internal combined path.

The separate dynamic_mxfp4_quant + e8m0_shuffle takes ~34us.
Check if aiter has a combined quant path that does both in one kernel,
or if we can call the internal quant function with shuffle=True.
"""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Try using aiter's combined quant that outputs shuffled scales directly
    # The reference.py uses _quant_mxfp4(A, shuffle=True) which calls
    # dynamic_mxfp4_quant then e8m0_shuffle. But maybe there's a faster path.

    # Check if dynamic_mxfp4_quant accepts a shuffle parameter
    try:
        # Some versions of dynamic_mxfp4_quant support shuffle parameter
        A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A, shuffle=True)
        A_q = A_fp4.view(dtypes.fp4x2)
        A_scale_sh = A_scale_e8m0.view(dtypes.fp8_e8m0)
    except TypeError:
        # Fallback to separate quant + shuffle
        A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
        A_q = A_fp4.view(dtypes.fp4x2)
        A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)

    out = aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
    return out
