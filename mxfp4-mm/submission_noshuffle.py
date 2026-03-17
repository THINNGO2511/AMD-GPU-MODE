#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Skip e8m0_shuffle on A scales.

Profiling shows quant+shuffle = 34us, GEMM = 14us.
The e8m0_shuffle is pure overhead if we can use un-shuffled scales.
gemm_a4w4 with bpreshuffle=True expects shuffled scales on B, but
maybe we can pass un-shuffled A scales with bpreshuffle=False or
use a different code path.

Also test: does dynamic_mxfp4_quant already produce shuffled output?
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

    # Quantize A - skip e8m0_shuffle to save time
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    # Try WITHOUT shuffle - the reference submission.py does shuffle,
    # but maybe the ASM kernel handles un-shuffled A scales internally
    A_scale = A_scale_e8m0.view(dtypes.fp8_e8m0)

    # The GEMM kernel might internally shuffle A scales
    # bpreshuffle=True refers to B weights being pre-shuffled
    out = aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
    return out
