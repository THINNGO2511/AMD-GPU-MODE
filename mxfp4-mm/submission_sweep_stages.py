#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Config sweep: num_stages=3 for K=7168, KSPLIT for K=2048.
Based on optimal_v4. Tests if num_stages=3 gives the +30% from PR #2160.
Also tries KSPLIT=4 for K=2048 (M=64, N=7168).
All configs have cache_modifier="" (required).
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}

# K=7168: try num_stages=3 (PR #2160 claims +30% on gfx950)
_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": "",
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

# K=2048: try KSPLIT=4 to use more CUs
_K2048_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": "",
    "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
}


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
        cfg = _K7168_CONFIG
    elif k == 2048:
        cfg = _K2048_CONFIG
    else:
        cfg = None

    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)
