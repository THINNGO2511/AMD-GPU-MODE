#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Minimal probe: ONLY print _mxfp4_quant_op signature + tl.dot_scaled test."""
from task import input_t, output_t
import torch, sys, triton, triton.language as tl

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_done = False

def P(m): print(f"S: {m}", file=sys.stderr)

def _unshuffle_e8m0(s):
    s = s.view(torch.uint8); sm, sn = s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)

def _run():
    global _done
    if _done: return
    _done = True
    # Print ONLY lines 83-100 of quant.py (function signature)
    try:
        with open("/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant/quant.py") as f:
            lines = f.readlines()
        for i in range(82, min(100, len(lines))):
            P(f"{i+1}|{lines[i].rstrip()}")
    except Exception as e:
        P(f"ERR:{e}")
    # Test tl.dot_scaled
    try:
        @triton.jit
        def _t(a, b, c, sa, sb, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
            om = tl.arange(0, BM); on = tl.arange(0, BN); ok = tl.arange(0, BK//2); oks = tl.arange(0, BK//32)
            av = tl.load(a + om[:,None]*(BK//2) + ok[None,:]); bv = tl.load(b + on[:,None]*(BK//2) + ok[None,:])
            asv = tl.load(sa + om[:,None]*(BK//32) + oks[None,:]); bsv = tl.load(sb + on[:,None]*(BK//32) + oks[None,:])
            acc = tl.zeros((BM,BN), dtype=tl.float32)
            acc = tl.dot_scaled(av, asv, "e2m1", bv, bsv, "e2m1", acc)
            tl.store(c + om[:,None]*BN + on[None,:], acc.to(tl.bfloat16))
        A = torch.zeros((32,64), dtype=torch.uint8, device='cuda')
        B = torch.zeros((32,64), dtype=torch.uint8, device='cuda')
        As = torch.full((32,4), 127, dtype=torch.uint8, device='cuda')
        Bs = torch.full((32,4), 127, dtype=torch.uint8, device='cuda')
        C = torch.zeros((32,32), dtype=torch.bfloat16, device='cuda')
        _t[(1,)](A, B, C, As, Bs, BM=32, BN=32, BK=128, num_warps=4, num_stages=1)
        torch.cuda.synchronize()
        P(f"DOT_SCALED:OK C[0,0]={C[0,0].item()}")
    except Exception as e:
        P(f"DOT_SCALED:FAIL {type(e).__name__}:{str(e)[:200]}")

def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh; _bscale_raw = _unshuffle_e8m0(B_scale_sh); _bq_u8 = B_q.view(torch.uint8)
    _run()
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16)
