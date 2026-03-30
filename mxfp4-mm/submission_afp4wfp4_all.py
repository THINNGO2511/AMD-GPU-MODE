#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM: Use gemm_afp4wfp4 for ALL shapes (not just K=1536).
afp4wfp4 has 69 pre-tuned config files on gfx950. Maybe they're better
than a16wfp4 defaults for K=7168/K=2048 too.
Trade: extra A quantization cost, but potentially faster GEMM kernel.
"""
from task import input_t, output_t
import torch

_ref = None
_raw = None
_bq = None
_y = {}
_warmed = False

def _unshuffle(s):
    s = s.view(torch.uint8); sm, sn = s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)

def _prewarm():
    global _warmed
    if _warmed: return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        for wk in (512, 1536, 2048):
            try: dynamic_mxfp4_quant(torch.randn((wm, wk), dtype=torch.bfloat16, device='cuda'))
            except: pass
    torch.cuda.synchronize()

def custom_kernel(data: input_t) -> output_t:
    global _ref, _raw, _bq
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    if _ref is not B_scale_sh:
        _ref = B_scale_sh; _raw = _unshuffle(B_scale_sh); _bq = B_q.view(torch.uint8); _prewarm()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    # Quantize A to FP4 for ALL shapes
    af, asc = dynamic_mxfp4_quant(A)
    return gemm_afp4wfp4(af.view(torch.uint8), _bq, asc, _raw, dtype=torch.bfloat16)
