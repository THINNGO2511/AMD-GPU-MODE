#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Use aiter's gemm_a16wfp4 with pre-shuffled B weights.
Avoids un-shuffling B_scale. Uses AMD tuned PREQUANT configs.
"""
from task import input_t, output_t
import torch

_c_cache = {}


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Try gemm_a16wfp4 with pre-shuffled weights
    # B_shuffle is pre-shuffled fp4x2, B_scale_sh is pre-shuffled e8m0
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
        B_sh_u8 = B_shuffle.view(torch.uint8)
        return gemm_a16wfp4_preshuffle(A, B_sh_u8, B_scale_sh, prequant=True, dtype=torch.bfloat16)
    except (ImportError, AttributeError, TypeError):
        # Fallback: try non-preshuffle path
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        B_q_u8 = B_q.view(torch.uint8)
        # Unshuffle scales for the raw path
        s = B_scale_sh.view(torch.uint8)
        sm, sn = s.shape
        s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
        s = s.permute(0, 5, 3, 1, 4, 2).contiguous().view(sm, sn)
        return gemm_a16wfp4(A, B_q_u8, s, dtype=torch.bfloat16)
