#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Triton GEMM with uint8 dtype workaround.

The aiter Triton gemm_afp4wfp4 failed with KeyError on 'float4_e2m1fn_x2'.
This is because the config lookup uses dtype names. We bypass this by
using the raw Triton kernel directly with proper dtype handling.
"""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape
    n = B.shape[0]

    # Quantize A to MXFP4
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)

    # Use the standard wrapper but try to pass through the internal
    # blockscale path by checking if it's available
    try:
        # Try the blockscale (Triton) path directly
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4_blockscale
        out = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        out = gemm_a4w4_blockscale(
            A_q,
            B_shuffle,
            A_scale_sh,
            B_scale_sh,
            out,
            bpreshuffle=True,
        )
        return out
    except Exception:
        pass

    # Fallback: standard path
    out = aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
    return out
