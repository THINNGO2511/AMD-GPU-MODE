#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Use gemm_a16wfp4_preshuffle for ALL sizes.
Takes pre-shuffled B_shuffle + B_scale_sh directly.
Config file gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json
has NUM_KSPLIT=14 for the critical K=7168 case.
This eliminates quant + shuffle overhead entirely.
"""
from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    return gemm_a16wfp4_preshuffle(
        A,
        B_shuffle.view(torch.uint8),
        B_scale_sh.view(torch.uint8),
        dtype=torch.bfloat16,
    )
