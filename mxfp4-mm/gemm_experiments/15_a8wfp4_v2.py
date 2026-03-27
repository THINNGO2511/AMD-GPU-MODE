#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 15: gemm_a8wfp4 v2 — fixed caching bug
Previous attempt had w_scales shape mismatch because _bscale_raw was cached
from a different test case. Fix: always recompute scales per call.

gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config)
  x: FP8 E4M3 (M, K)
  w: FP4 packed uint8 (N, K//2)
  y: output (M, N)
  x_scales: FP32 per-row (M, 1)
  w_scales: E8M0 unshuffled (N, K//32)
"""
from task import input_t, output_t
import torch

_y_cache = {}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Always recompute scales (no caching — avoids shape mismatch)
    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    bq_u8 = B_q.view(torch.uint8)

    # K=1536: use afp4wfp4 (known best)
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), bq_u8, A_scale, bscale_raw,
                             dtype=torch.bfloat16)

    # fp8 per-row quantization of A
    from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4

    amax = A.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0
    A_fp8 = (A / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    return gemm_a8wfp4(A_fp8, bq_u8, _y_cache[key], scale, bscale_raw,
                       dtype=torch.bfloat16)
