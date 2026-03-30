#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Sweep BM=4 configs for K=7168 (M=16, N=2112).
BM=4 matches AMD's auto-config for M=16. 4 M-tiles instead of 2.
Test: stages, wpe, KSPLIT, BN variations with BM=4.
"""
from task import input_t, output_t
import torch
import time

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None

_K7168 = [
    # [0] BM=4,BN=64,KS=8,s=2,wpe=2 (baseline BM=4)
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
    # [1] BM=4,stages=3
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
    # [2] BM=4,wpe=1
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
    # [3] BM=4,stages=3,wpe=1
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
    # [4] BM=4,KS=14
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": None},
    # [5] BM=4,BN=128,KS=8
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
    # [6] BM=8 baseline for comparison
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
]

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

_swept = set()

def custom_kernel(data):
    global _bscale_ref, _bscale_raw, _bq_u8
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    if k == 7168 and k not in _swept:
        _swept.add(k)
        y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        for i, cfg in enumerate(_K7168):
            try:
                for _ in range(3): gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20): gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
                torch.cuda.synchronize()
                us = (time.perf_counter() - t0) / 20 * 1e6
                print(f"[{i}] {us:7.1f}us BM={cfg['BLOCK_SIZE_M']},BN={cfg['BLOCK_SIZE_N']},KS={cfg['NUM_KSPLIT']},s={cfg['num_stages']},wpe={cfg['waves_per_eu']}")
            except Exception as e:
                print(f"[{i}] FAIL {e}")

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)

    cfg = _K7168[0] if k == 7168 else None
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
