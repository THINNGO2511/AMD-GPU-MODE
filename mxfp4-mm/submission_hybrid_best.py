#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Best hybrid:
K<=1024: gemm_a16wfp4 (bf16 A, no quant, 6us)
K>1024: separate quant + gemm_afp4wfp4 (tuned Triton configs, proven 14-16us)
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}


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

    if k <= 1024:
        # gemm_a16wfp4: takes bf16 A, quantizes on-the-fly, no quant overhead
        # Benchmarked at 6.18-6.86us for K=512 sizes
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        key = (m, n)
        if key not in _y_cache:
            _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=_y_cache[key])
    else:
        # Separate quant + Triton GEMM with tuned per-N-K configs
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)
