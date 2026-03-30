#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — CK path via gemm_a4w4 with bpreshuffle=True.
Uses aiter's dynamic_mxfp4_quant (correct rounding) + CK compiled kernels.
CK path advantage: B_shuffle + B_scale_sh used directly (no unshuffle needed).
Goal: beat Triton a16wfp4 ceiling (~10μs bench, ~16μs ranked).
"""
from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter import dtypes
import aiter

_warmed = set()


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape

    # Warm up Triton quant JIT for this (M,K) — CK kernels are pre-compiled
    mk = (m, k)
    if mk not in _warmed:
        _warmed.add(mk)
        # Also warm other M sizes we'll see (quant only varies by M,K)
        for wm in (4, 16, 32, 64, 256):
            wk = k
            try:
                wA = torch.randn((wm, wk), dtype=torch.bfloat16, device='cuda')
                dynamic_mxfp4_quant(wA)
            except Exception:
                pass
        torch.cuda.synchronize()

    # Quantize A -> MXFP4 + shuffle scales
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale).view(dtypes.fp8_e8m0)

    # CK GEMM — B_shuffle and B_scale_sh used directly (already shuffled)
    return aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
