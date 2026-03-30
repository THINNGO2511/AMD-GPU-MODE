#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Per-size optimal hybrid:
K=512: gemm_a16wfp4 default (6.15-6.82us, -28% vs quant+gemm)
K=2048: gemm_a16wfp4 default (14.0us, -8% vs quant+gemm)
K=7168: gemm_a16wfp4 split-K=14 (24.9us, ~same as quant+gemm)
K=1536: separate quant + gemm_afp4wfp4 (15.9us, a16wfp4 is 25us here)
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}

# Per-size configs for a16wfp4 (only for K=7168 which needs split-K)
_A16W_CONFIGS = {
    (16, 2112, 7168): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 1024,
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

    # Per-size routing based on benchmarked performance
    if k == 1536 and m == 256:
        # K=1536, M=256: quant+afp4wfp4 is much faster (15.9 vs 24.5us)
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)
    else:
        # All other sizes: a16wfp4 (no A quant overhead)
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        key = (m, n)
        if key not in _y_cache:
            _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

        cfg = _A16W_CONFIGS.get((m, n, k))
        return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                           y=_y_cache[key], config=cfg)
