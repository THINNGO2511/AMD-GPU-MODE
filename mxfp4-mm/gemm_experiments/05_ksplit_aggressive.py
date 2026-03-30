#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 05: Aggressive split-K for all shapes
Try higher NUM_KSPLIT values. For memory-bound shapes (small M),
split-K increases parallelism at the cost of a reduction step.
MI355X has 304 CUs — with small M, many CUs are idle without split-K.

For M=4,N=2880,K=512: only ceil(4/BM)*ceil(2880/BN) = 1*45 = 45 tiles.
  304 CUs but only 45 tiles → CU utilization = 15%. Split-K=4 → 180 tiles = 59%.
For M=16,N=2112,K=7168: already uses KS=8 (our tuned config).
For M=32,N=4096,K=512: ceil(32/16)*ceil(4096/64) = 2*64 = 128 tiles. KS=2 → 256.
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_warmed = False

# Aggressive split-K configs
_CONFIGS = {
    (4, 2880, 512): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
    },
    (16, 2112, 7168): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 1024,
    },
    (32, 4096, 512): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
    },
    (32, 2880, 512): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
    },
    (64, 7168, 2048): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 1024,
    },
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

    cfg = _CONFIGS.get((m, n, k))

    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)
