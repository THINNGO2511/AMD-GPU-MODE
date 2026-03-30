#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 06: Use afp4wfp4 (separate quant) for ALL shapes
Currently only K=1536 uses afp4wfp4. For other shapes, a16wfp4 (fused quant)
was assumed better. But afp4wfp4 has per-shape tuned configs from aiter's
config files (69 FP4 config files). Maybe it's actually better for some shapes.

This tests the theory that aiter's pre-tuned afp4wfp4 configs (tuned by AMD
engineers) might outperform a16wfp4 with our manual/default configs.
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_warmed = False


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
    # Warm quant for all K values
    for wk in (512, 1536, 2048, 7168):
        for wm in (4, 16, 32, 64, 256):
            try:
                wA = torch.randn((wm, wk), dtype=torch.bfloat16, device='cuda')
                dynamic_mxfp4_quant(wA)
            except Exception:
                pass
    torch.cuda.synchronize()


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

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                         dtype=torch.bfloat16)
