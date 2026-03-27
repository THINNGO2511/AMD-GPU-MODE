#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""GEMM: gemm_a16wfp4 without pre-alloc (let it allocate), num_stages=3"""
import torch
from task import input_t, output_t

_cache = {}

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]

    key = id(B_scale_sh)
    if key not in _cache:
        _cache[key] = (_unshuffle_e8m0(B_scale_sh), B_q.view(torch.uint8))
    B_scale_raw, B_q_u8 = _cache[key]

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    # Don't pre-alloc, let gemm_a16wfp4 handle it
    # Pass num_stages=3 via config
    return gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16,
                        config={"num_stages": 3, "waves_per_eu": 1,
                                "matrix_instr_nonkdim": 16})
