#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — gemm_a16wfp4 with tuned split-K configs for ALL sizes.
K<=1024: default configs work great (6.18-6.86us)
K>1024: inject split-K configs (default has NUM_KSPLIT=1, terrible for large K)
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}

# Per-size tuned configs with split-K for large K
# Based on AFP4WFP4 tuned configs adapted for A16WFP4
_CONFIGS = {
    # K=512: single pass, no split-K needed (already fast with defaults)
    (4, 2880, 512): {
        "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024,
    },
    (32, 4096, 512): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024,
    },
    (32, 2880, 512): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024,
    },
    # K=7168: CRITICAL - needs aggressive split-K
    # 14 splits × 512 K-elements each = 7168
    (16, 2112, 7168): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 1024,
    },
    # K=2048: split-K=4
    (64, 7168, 2048): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 1024,
    },
    # K=1536: split-K=3
    (256, 3072, 1536): {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 1024,
    },
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

    # Pre-allocate output
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    y = _y_cache[key]

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    cfg = _CONFIGS.get((m, n, k))
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
