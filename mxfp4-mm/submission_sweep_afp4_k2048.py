#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Test if quant+afp4wfp4 path is faster for K=2048 (M=64,N=7168).
Rationale: a16wfp4 re-quantizes A for EACH of 56 N-tiles (N/128).
quant+afp4wfp4 quantizes A once, then fp4 A is half the bandwidth.
Also sweep afp4wfp4 with split-K configs for K=2048 and K=1536.
"""
from task import input_t, output_t
import torch
import sys

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_swept = set()

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def _benchmark_fn(fn, warmup=3, iters=20):
    """Time a function using CUDA events, returns median in us."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for j in range(iters):
        starts[j].record()
        fn()
        ends[j].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
    return times[iters // 2]

def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    key = (m, n, k)
    if key not in _swept:
        _swept.add(key)

        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

        y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

        print(f"\n=== COMPARE PATHS for M={m},N={n},K={k} ===")

        # Path 1: a16wfp4 default
        try:
            t = _benchmark_fn(lambda: gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y))
            print(f"  a16wfp4 default:     {t:.1f}us")
        except Exception as e:
            print(f"  a16wfp4 default:     FAILED ({e})")

        # Path 2: a16wfp4 with KSPLIT=2
        if k >= 1024:
            ks_vals = [2, 4] if k >= 2048 else [3]
            for ks in ks_vals:
                sbs = max(256, k // ks)
                cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
                       "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                       "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
                       "NUM_KSPLIT": ks, "SPLITK_BLOCK_SIZE": sbs}
                try:
                    t = _benchmark_fn(lambda: gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg))
                    print(f"  a16wfp4 KS={ks}:       {t:.1f}us")
                except Exception as e:
                    print(f"  a16wfp4 KS={ks}:       FAILED ({e})")

        # Path 3: quant + afp4wfp4
        try:
            def _quant_gemm():
                A_fp4, A_scale = dynamic_mxfp4_quant(A)
                return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                                     dtype=torch.bfloat16)
            t = _benchmark_fn(_quant_gemm)
            print(f"  quant+afp4wfp4:      {t:.1f}us")
        except Exception as e:
            print(f"  quant+afp4wfp4:      FAILED ({e})")

        # Path 4: quant + afp4wfp4 with pre-allocated quant output
        try:
            A_fp4_pre = torch.empty((m, k // 2), dtype=torch.uint8, device='cuda')
            A_scale_pre = torch.empty((m, k // 32), dtype=torch.uint8, device='cuda')
            def _quant_gemm_prealloc():
                A_fp4, A_scale = dynamic_mxfp4_quant(A)
                return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                                     dtype=torch.bfloat16)
            t = _benchmark_fn(_quant_gemm_prealloc)
            print(f"  quant+afp4wfp4 prealloc: {t:.1f}us")
        except Exception as e:
            print(f"  quant+afp4wfp4 pre:  FAILED ({e})")

        # Path 5: a16wfp4 with stages=3
        for bm in [8, 16, 32, 64]:
            for bn in [64, 128]:
                for ns in [2, 3]:
                    cfg = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 512,
                           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": ns,
                           "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
                           "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": k}
                    try:
                        t = _benchmark_fn(lambda: gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg), warmup=2, iters=10)
                        print(f"  a16wfp4 BM={bm:3d} BN={bn:3d} S={ns} KS=1: {t:.1f}us")
                    except Exception as e:
                        print(f"  a16wfp4 BM={bm:3d} BN={bn:3d} S={ns} KS=1: FAIL")

    # Use best known path
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    cfg = None
    if k == 7168:
        cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
               "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
               "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
               "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024}
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
