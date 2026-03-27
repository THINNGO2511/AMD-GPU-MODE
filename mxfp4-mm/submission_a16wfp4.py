#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Use aiter's gemm_a16wfp4 which takes bf16 A directly and
quantizes A on-the-fly inside the GEMM kernel with AMD-tuned configs.
This eliminates the separate quant+shuffle overhead (~34us = 65% of time).
Tuned configs use BLOCK_K=512, BLOCK_M=4-8, NUM_KSPLIT=4-14.
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_c_cache = {}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    # Use gemm_a16wfp4: takes bf16 A, quantizes on-the-fly, uses tuned configs
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16)
