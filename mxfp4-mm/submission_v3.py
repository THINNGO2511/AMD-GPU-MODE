#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Triton GEMM path via aiter with raw (un-shuffled) weights.

Strategy: Use aiter's Triton-based gemm_afp4wfp4 instead of the CK gemm_a4w4.
The Triton path supports split-K and has autotuning for different tile configs,
which may be better for the small-M decode workloads.

We re-quantize B to get un-shuffled scales (required by the non-preshuffle Triton path).
"""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A (un-shuffled scales for Triton path)
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale = A_scale.view(dtypes.fp8_e8m0)

    # Re-quantize B to get un-shuffled scales
    # (B_scale_sh is CK-shuffled, Triton needs raw scales)
    B_fp4, B_scale = dynamic_mxfp4_quant(B)
    B_q_raw = B_fp4.view(dtypes.fp4x2)
    B_scale_raw = B_scale.view(dtypes.fp8_e8m0)

    # Triton GEMM with raw (un-shuffled) weights
    out = gemm_afp4wfp4(
        A_q,
        B_q_raw,
        A_scale,
        B_scale_raw,
        dtype=torch.bfloat16,
    )

    return out
