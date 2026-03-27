#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 13: gemm_a16wfp4_preshuffle — use B_shuffle directly!
The problem provides B_shuffle (pre-shuffled B) and B_scale_sh (shuffled scales).
gemm_a16wfp4_preshuffle takes these DIRECTLY — no unshuffle needed!

API: gemm_a16wfp4_preshuffle(x, w, w_scales, prequant=True, dtype, y, config, skip_reduce)
  x: bf16 A (M, K)
  w: pre-shuffled FP4 weight (N, K//2) — this is B_shuffle!
  w_scales: shuffled E8M0 scales (N//32, K//32) — this is B_scale_sh!
  prequant=True: quantize A internally (like a16wfp4)

Has tuned config: gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json
Could be faster because:
1. No unshuffle step (saves time and memory)
2. Pre-shuffled data is already in optimal memory layout for the kernel
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bshuffle_u8 = None
_y_cache = {}
_warmed = False


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


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bshuffle_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bshuffle_u8 = B_shuffle.view(torch.uint8)
        _prewarm()

    # K=1536: use afp4wfp4 (known best)
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.utility.fp4_utils import e8m0_shuffle
        # afp4wfp4 needs unshuffled scales, but we only have shuffled here
        # So unshuffle for this path only
        s = B_scale_sh.view(torch.uint8)
        sm, sn = s.shape
        s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
        s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
        bscale_raw = s.view(sm, sn)
        bq_u8 = B_q.view(torch.uint8)
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), bq_u8, A_scale, bscale_raw,
                             dtype=torch.bfloat16)

    # Use a16wfp4_preshuffle for all other shapes
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    # B_scale_sh is already in shuffled E8M0 format — pass directly!
    return gemm_a16wfp4_preshuffle(
        A, _bshuffle_u8, B_scale_sh,
        prequant=True, dtype=torch.bfloat16,
        y=_y_cache[key]
    )
