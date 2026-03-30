#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
CK ASM path with auto kernel selection via aiter.gemm_a4w4.
Quantize A→fp4, shuffle, call CK GEMM with pre-shuffled B.
"""
import torch
from task import input_t, output_t
import aiter
from aiter import dtypes as aiter_dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

_cache = {}

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]

    # Quantize A to fp4
    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)

    # Call CK ASM GEMM with shuffled inputs
    C = aiter.gemm_a4w4(
        A_q.view(B_shuffle.dtype),
        B_shuffle,
        A_scale_sh.view(B_scale_sh.dtype),
        B_scale_sh,
        dtype=torch.bfloat16,
        bpreshuffle=True
    )
    return C[:M]
