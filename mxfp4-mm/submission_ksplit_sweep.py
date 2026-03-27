#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Targeted sweep: test KSPLIT=2,3,4,5,6,7,8,9 for K=7168 in actual benchmark.
Each gets its own benchmark run by routing per-call-count.
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_call_count = 0

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def _mc(ks):
    return {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
            "NUM_KSPLIT": ks, "SPLITK_BLOCK_SIZE": 1024}

def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8, _call_count

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    if k == 7168:
        # Use KSPLIT=8 (proven best)
        cfg = _mc(8)
    else:
        cfg = None

    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)
