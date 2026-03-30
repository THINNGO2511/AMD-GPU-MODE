#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Config sweep for K=512 sizes (3 of 6 benchmark sizes).
Currently 6.13-6.98us, target: <5.5us.
Also tests K=1536 with a16wfp4 and different configs.
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

# K=512 config candidates - focused sweep
_K512_CANDIDATES = [
    # Default: BM=4, BN=128, BK=512, W=4, WPE=2 → 6.13us (M=4)
    # Default: BM=8, BN=128, BK=512, W=8, WPE=2 → 6.59-6.98us (M=32)
    # Try smaller BN for more parallelism
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    # Try BK=256 (2 iterations, but each iteration smaller)
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    # Try more warps
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    # Try waves_per_eu=0 (let hardware decide)
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 1, "waves_per_eu": 0, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    # BM=16 for M=32 case
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    # Try num_stages=2 (pipelining)
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    # High waves for small tiles
    {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 2, "num_stages": 1, "waves_per_eu": 4, "matrix_instr_nonkdim": 16,
     "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
]

# K=7168 best config (from previous sweep)
_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


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
            ks, nw, ns, wpe = cfg["NUM_KSPLIT"], cfg["num_warps"], cfg["num_stages"], cfg["waves_per_eu"]
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
        if k == 512:
            _sweep(A, _bq_u8, _bscale_raw, m, n, k, _K512_CANDIDATES)

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
    if cfg is None and k == 7168:
        cfg = _K7168_CONFIG
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[mk], config=cfg)
