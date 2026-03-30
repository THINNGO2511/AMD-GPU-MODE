#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Compare a16wfp4 (default vs custom) vs quant+afp4wfp4 for ALL sizes.
This will determine the optimal kernel path for each benchmark size.
"""
from task import input_t, output_t
import torch
import time

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_compared = set()

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _compare(A, bq, bscale, m, n, k):
    if (m, n, k) in _compared:
        return
    _compared.add((m, n, k))

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    print(f"\n=== M={m} N={n} K={k} ===", flush=True)

    # 1. a16wfp4 default
    for _ in range(5):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y)
    torch.cuda.synchronize()
    a16_default = (time.perf_counter() - t0) / 30 * 1e6
    print(f"  a16wfp4 default:     {a16_default:7.1f}us", flush=True)

    # 2. quant+afp4wfp4
    for _ in range(5):
        A_fp4, A_sc = dynamic_mxfp4_quant(A)
        gemm_afp4wfp4(A_fp4.view(torch.uint8), bq, A_sc, bscale, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        A_fp4, A_sc = dynamic_mxfp4_quant(A)
        gemm_afp4wfp4(A_fp4.view(torch.uint8), bq, A_sc, bscale, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    afp4_total = (time.perf_counter() - t0) / 30 * 1e6
    print(f"  quant+afp4wfp4:      {afp4_total:7.1f}us", flush=True)

    # 3. a16wfp4 with BM=4 (for small M)
    if m <= 16:
        cfg = {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
               "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
               "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
               "NUM_KSPLIT": 8 if k >= 4096 else 1, "SPLITK_BLOCK_SIZE": 1024}
        for _ in range(5):
            gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
        torch.cuda.synchronize()
        a16_bm4 = (time.perf_counter() - t0) / 30 * 1e6
        print(f"  a16wfp4 BM=4 KS={'8' if k>=4096 else '1'}:   {a16_bm4:7.1f}us", flush=True)

    # 4. a16wfp4 with stages=3
    cfg_s3 = {"BLOCK_SIZE_M": 4 if m <= 16 else 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
              "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
              "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
              "NUM_KSPLIT": 8 if k >= 4096 else 1, "SPLITK_BLOCK_SIZE": 1024}
    for _ in range(5):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg_s3)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg_s3)
    torch.cuda.synchronize()
    a16_s3 = (time.perf_counter() - t0) / 30 * 1e6
    print(f"  a16wfp4 stages=3:    {a16_s3:7.1f}us", flush=True)

    # 5. a16wfp4 with wpe=1
    cfg_wpe1 = {"BLOCK_SIZE_M": 4 if m <= 16 else 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None,
                "NUM_KSPLIT": 8 if k >= 4096 else 1, "SPLITK_BLOCK_SIZE": 1024}
    for _ in range(5):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg_wpe1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg_wpe1)
    torch.cuda.synchronize()
    a16_wpe1 = (time.perf_counter() - t0) / 30 * 1e6
    print(f"  a16wfp4 wpe=1:       {a16_wpe1:7.1f}us", flush=True)

    winner = min(a16_default, afp4_total, a16_s3, a16_wpe1)
    print(f"  BEST: {winner:.1f}us", flush=True)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    _compare(A, _bq_u8, _bscale_raw, m, n, k)

    # Use proven path
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    cfg = {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
           "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024} if k == 7168 else None

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
