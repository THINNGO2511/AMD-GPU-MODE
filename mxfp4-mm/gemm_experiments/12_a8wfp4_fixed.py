#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 12: gemm_a8wfp4 with correct FP8 quantization
Uses manual per-row FP8 quantization for A, then calls gemm_a8wfp4.
fp8 A = 50% less A bandwidth than bf16. Could help for larger M shapes.

API: gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config)
  x: FP8 E4M3 (M, K)
  w: FP4 packed uint8 (N, K//2) — same as B_q
  y: pre-allocated output (M, N)
  x_scales: FP32 per-row scale (M, 1)
  w_scales: E8M0 per-group scale (N, K//32) — same as our unshuffled B_scale
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_a8_y_cache = {}
_a8_scale_cache = {}
_warmed = False

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try:
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
        except Exception:
            pass
    torch.cuda.synchronize()


def _quant_fp8_perrow(A):
    """Quantize bf16 A to fp8 E4M3 with per-row FP32 scale."""
    m, k = A.shape
    # Per-row amax
    amax = A.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    # FP8 E4M3 max value is 448.0
    scale = amax / 448.0
    A_scaled = A / scale
    A_fp8 = A_scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return A_fp8, scale


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        _prewarm()

    # K=1536: use afp4wfp4 (known best)
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    # Try a8wfp4 for all other shapes
    from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4

    # Pre-allocate output
    key = (m, n)
    if key not in _a8_y_cache:
        _a8_y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    # Quantize A to fp8
    A_fp8, A_scale = _quant_fp8_perrow(A)

    return gemm_a8wfp4(A_fp8, _bq_u8, _a8_y_cache[key], A_scale, _bscale_raw,
                       dtype=torch.bfloat16)
