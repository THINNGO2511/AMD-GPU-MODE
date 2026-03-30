"""
GEMM Experiment: Pure gemm_a16wfp4 for ALL shapes (including K=1536).
Hypothesis: gemm_a16wfp4 fuses quantization inside the kernel. For K=1536,
the separate quant + afp4wfp4 path adds overhead. Maybe a16wfp4 is faster
even for K=1536 if we let it use its default config.

Also: don't cache output tensors (potential reward hack concern).
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
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    # Warm all M values for K=512 (common)
    for wm in (4, 16, 32, 64, 256):
        try:
            wA = torch.randn((wm, 512), dtype=torch.bfloat16, device='cuda')
            wB = torch.randint(0, 255, (512, 256), dtype=torch.uint8, device='cuda')
            wS = torch.randint(100, 150, (512, 8), dtype=torch.uint8, device='cuda')
            gemm_a16wfp4(wA, wB, wS, dtype=torch.bfloat16)
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

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    # Use a16wfp4 for ALL K values (including K=1536)
    # Let it use default config everywhere
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16)
