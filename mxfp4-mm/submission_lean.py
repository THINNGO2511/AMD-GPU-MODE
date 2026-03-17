#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Lean submission minimizing Python overhead.

Pre-import everything at module level. Cache output tensors.
Call gemm_a4w4 with minimal argument overhead.
"""
from task import input_t, output_t
import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

_out_cache = {}


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)

    # Use cached output tensor to avoid allocation
    key = (m, n)
    if key not in _out_cache:
        _out_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    return aiter.gemm_a4w4(
        A_fp4.view(dtypes.fp4x2),
        B_shuffle,
        e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0),
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
