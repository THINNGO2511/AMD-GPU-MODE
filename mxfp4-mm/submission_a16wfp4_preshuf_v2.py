#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Use gemm_a16wfp4_preshuffle with pre-shuffled B weights.
Eliminates ALL A quantization overhead AND scale unshuffle.
B_shuffle and B_scale_sh are passed directly.
"""
from task import input_t, output_t
import torch

_y_cache = {}
_b_ref = None
_b_sh_u8 = None
_b_scale_u8 = None


def custom_kernel(data: input_t) -> output_t:
    global _b_ref, _b_sh_u8, _b_scale_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Cache uint8 views (Triton doesn't recognize float8_e8m0fnu dtype)
    if _b_ref is not B_shuffle:
        _b_ref = B_shuffle
        _b_sh_u8 = B_shuffle.view(torch.uint8)
        _b_scale_u8 = B_scale_sh.view(torch.uint8)

    # Pre-allocate output
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    y = _y_cache[key]

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    return gemm_a16wfp4_preshuffle(
        A, _b_sh_u8, _b_scale_u8, prequant=True, dtype=torch.bfloat16, y=y
    )
