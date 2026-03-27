#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 08: Very small tiles for maximum parallelism
For M=4 to M=64, large tiles waste CUs. Use BM=8, BN=32 to maximize
the number of tiles and CU utilization on the 304-CU MI355X.

For M=4: BM=8 means 1 tile row, BN=32 → 2880/32=90 tiles.
  With KS=4: 360 tiles → 1.18 tiles/CU.
For M=32: BM=8 → 4 tile rows, BN=32 → 90 cols → 360 tiles.
  With KS=2: 720 tiles → 2.37 tiles/CU.
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_warmed = False

_CONFIGS = {
    (4, 2880, 512): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
    },
    (16, 2112, 7168): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 1024,
    },
    (32, 4096, 512): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 512,
    },
    (32, 2880, 512): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 512,
    },
    (64, 7168, 2048): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None,
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
