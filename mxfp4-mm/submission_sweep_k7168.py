#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Benchmark-focused sweep for K=7168 (M=16,N=2112) - biggest bottleneck.
Tests ~15 configs and picks the best. Also sweeps K=2048 (M=64,N=7168).
Falls back to proven approach for K=1536.
"""
from task import input_t, output_t
import torch
import sys

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_swept = set()
_best_configs = {}

# Configs to sweep for K=7168, M=16, N=2112
_K7168_CANDIDATES = [
    # KSPLIT=8 variants (currently best at 22.7us)
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 2, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": None, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # Try KSPLIT=7 (K=7168/512=14 iters, 7 splits=2 iters each)
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 7, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 2, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 7, "SPLITK_BLOCK_SIZE": 1024},
    # Try KSPLIT=14 (1 iter each, maximum parallelism)
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 2, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": None, "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 1024},
    # Try KSPLIT=4 with BK=256 (moderate parallelism)
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 1024},
    # Larger BN for better memory coalescing
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": None, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # Try num_stages=1 (less prefetching, lower register pressure)
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 2, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # More warps options
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 8, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 7, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT=8 with BK=256
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
]

# Configs for K=2048, M=64 (currently 14.2us with default KSPLIT=1)
_K2048_CANDIDATES = [
    # Default is KSPLIT=1 with BM=8,BN=128,BK=512 → 14.2us
    # Try KSPLIT=2 with smaller tiles
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    # Try KSPLIT=2
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024},
]


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _sweep(A, w, w_scales, m, n, k, candidates):
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    print(f"\n[SWEEP] M={m},N={n},K={k}: {len(candidates)} configs", file=sys.stderr)
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    best_t = float('inf')
    best_cfg = None
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for i, cfg in enumerate(candidates):
        try:
            gemm_a16wfp4(A, w, w_scales, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()

            start_ev.record()
            for _ in range(20):
                gemm_a16wfp4(A, w, w_scales, dtype=torch.bfloat16, y=y, config=cfg)
            end_ev.record()
            torch.cuda.synchronize()
            t = start_ev.elapsed_time(end_ev) / 20 * 1000

            bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
            ks, nw, ns = cfg["NUM_KSPLIT"], cfg["num_warps"], cfg["num_stages"]
            wpe = cfg["waves_per_eu"]
            print(f"  [{i}] {t:.1f}us BM={bm} BN={bn} BK={bk} KS={ks} W={nw} S={ns} WPE={wpe}", file=sys.stderr)

            if t < best_t:
                best_t = t
                best_cfg = cfg.copy()
        except Exception as e:
            print(f"  [{i}] FAIL: {e}", file=sys.stderr)

    if best_cfg:
        print(f"[SWEEP] BEST M={m},N={n},K={k}: {best_t:.1f}us = {best_cfg}", file=sys.stderr)
        _best_configs[(m, n, k)] = best_cfg


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
        if k == 7168:
            _sweep(A, _bq_u8, _bscale_raw, m, n, k, _K7168_CANDIDATES)
        elif k == 2048:
            _sweep(A, _bq_u8, _bscale_raw, m, n, k, _K2048_CANDIDATES)

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    mk = (m, n)
    if mk not in _y_cache:
        _y_cache[mk] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    cfg = _best_configs.get(key)
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[mk], config=cfg)
