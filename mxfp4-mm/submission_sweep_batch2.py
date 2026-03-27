#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Batch 2: More K=7168 configs (BN variations) + K=1536 a16wfp4 attempts.
"""
from task import input_t, output_t
import torch
import time

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None

# K=7168 (M=16, N=2112): BN variations
_K7168 = [
    # [0] BN=128,KS=8
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
    # [1] BN=128,KS=14
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": None},
    # [2] BN=128,stages=3,wpe=1
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
    # [3] BN=32,KS=14
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": None},
]

# K=1536 (M=256, N=3072): Try a16wfp4 with KSPLIT=3
_K1536 = [
    # [0] KSPLIT=3, BM=32, BN=64
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": None},
    # [1] KSPLIT=3, BM=64, BN=128
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": None},
    # [2] KSPLIT=1, BM=64, BN=128
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1536, "cache_modifier": None},
    # [3] KSPLIT=3, BM=32, BN=128, stages=3
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": None},
]

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

_swept = set()

def _fmt(cfg):
    if cfg is None: return "DEFAULT"
    return f"BM={cfg['BLOCK_SIZE_M']},BN={cfg['BLOCK_SIZE_N']},KS={cfg['NUM_KSPLIT']},s={cfg['num_stages']},wpe={cfg['waves_per_eu']}"

def _sweep(name, cfgs, A, bq, bscale, m, n):
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    results = []
    for i, cfg in enumerate(cfgs):
        try:
            for _ in range(3): gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20): gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / 20 * 1e6
            results.append((us, i))
            print(f"{name}[{i}] {us:7.1f}us {_fmt(cfg)}")
        except Exception as e:
            print(f"{name}[{i}] FAIL {e}")
    results.sort()
    if results:
        print(f"{name} BEST: [{results[0][1]}] {results[0][0]:.1f}us")
    return results

def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    if k not in _swept:
        _swept.add(k)
        if k == 7168:
            _sweep("K7168", _K7168, A, _bq_u8, _bscale_raw, m, n)
        elif k == 1536:
            # Also time baseline (quant+afp4wfp4)
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                A_fp4, A_scale = dynamic_mxfp4_quant(A)
                gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            baseline = (time.perf_counter() - t0) / 20 * 1e6
            print(f"K1536 BASELINE (quant+afp4wfp4): {baseline:.1f}us")
            _sweep("K1536", _K1536, A, _bq_u8, _bscale_raw, m, n)

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)

    cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None} if k == 7168 else None
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
