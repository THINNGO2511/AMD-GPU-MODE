#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Optimize quantization overhead.

Profiling shows quant+shuffle=34us, GEMM=14us. The quantization is the bottleneck.

Approaches to reduce quant overhead:
1. Use aiter.get_triton_quant which may be optimized differently
2. Try torch quant functions
3. Check if gemm_a4w4 can accept un-shuffled scales (skip e8m0_shuffle)
4. Pre-allocate output tensors to avoid allocation overhead
"""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    import aiter
    from aiter import dtypes, QuantType
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Try the get_triton_quant path which might be faster
    try:
        triton_quant = aiter.get_triton_quant(QuantType.per_1x32)
        A_q, A_scale = triton_quant(A)
        A_q = A_q.view(dtypes.fp4x2)
        A_scale = A_scale.view(dtypes.fp8_e8m0)
    except Exception:
        # Fallback to direct dynamic_mxfp4_quant
        A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
        A_q = A_fp4.view(dtypes.fp4x2)
        A_scale = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)

    out = aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
    return out
