#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""GEMM: Use aiter.gemm_a4w4 with bpreshuffle=True (CK path, not Triton)"""
import torch
from task import input_t, output_t

_cache = {}

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]
    
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle
    import aiter
    
    # Quantize A
    A_q, A_s = dynamic_mxfp4_quant(A)
    A_s_sh = e8m0_shuffle(A_s)
    
    # Use gemm_a4w4 (CK path) with pre-shuffled B
    out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    return aiter.gemm_a4w4(
        A_q.view(B_shuffle.dtype),
        B_shuffle,
        A_s_sh.view(B_scale_sh.dtype),
        B_scale_sh,
        dtype=torch.bfloat16,
        bpreshuffle=True,
    )
