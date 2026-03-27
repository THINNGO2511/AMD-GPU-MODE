#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Targeted config sweep for gemm_a16wfp4.
~25 configs per size, focused on most impactful parameters.
"""
from task import input_t, output_t
import torch
import sys

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_swept = set()
_best_configs = {}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _get_sweep_configs(m, n, k):
    """Get targeted configs based on problem dimensions."""
    configs = []
    base = {"GROUP_SIZE_M": 1, "num_stages": 1, "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg", "SPLITK_BLOCK_SIZE": 1024}

    if k <= 512:
        # K=512: sweep BLOCK_M, BLOCK_N, warps, waves
        for bm in [4, 8, 16, 32]:
            if bm > m * 4:
                continue
            for bn in [16, 32, 64, 128]:
                for nw in [4, 8]:
                    for wpe in [0, 2]:
                        configs.append({**base, "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn,
                                       "BLOCK_SIZE_K": 512, "num_warps": nw,
                                       "waves_per_eu": wpe, "NUM_KSPLIT": 1})
    elif k <= 2048:
        # K=1536/2048: sweep KSPLIT and block sizes
        for bm in [16, 32, 64]:
            if bm > m * 2:
                continue
            for bn in [32, 64, 128]:
                for bk in [256, 512]:
                    for ks in [1, 2, 4]:
                        if ks > 1 and (k // bk) < ks:
                            continue
                        configs.append({**base, "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn,
                                       "BLOCK_SIZE_K": bk, "num_warps": 4,
                                       "waves_per_eu": 2, "NUM_KSPLIT": ks})
    else:
        # K=7168: sweep KSPLIT aggressively
        for bm in [8, 16, 32]:
            for bn in [16, 32, 64]:
                for bk in [256, 512]:
                    for ks in [4, 7, 8, 14]:
                        if (k // bk) < ks:
                            continue
                        configs.append({**base, "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn,
                                       "BLOCK_SIZE_K": bk, "num_warps": 4,
                                       "waves_per_eu": 1, "NUM_KSPLIT": ks})

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:40]  # Cap at 40


def _sweep(A, w, w_scales, m, n, k):
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    configs = _get_sweep_configs(m, n, k)
    print(f"\n[SWEEP] M={m},N={n},K={k}: {len(configs)} configs", file=sys.stderr)

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    best_t = float('inf')
    best_cfg = None
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for i, cfg in enumerate(configs):
        try:
            # Warmup (also compiles)
            gemm_a16wfp4(A, w, w_scales, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()

            # Time 10 iterations
            start_ev.record()
            for _ in range(10):
                gemm_a16wfp4(A, w, w_scales, dtype=torch.bfloat16, y=y, config=cfg)
            end_ev.record()
            torch.cuda.synchronize()
            t = start_ev.elapsed_time(end_ev) / 10 * 1000  # us

            if t < best_t:
                best_t = t
                best_cfg = cfg.copy()
                bm, bn, bk, ks = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"], cfg["NUM_KSPLIT"]
                nw, wpe = cfg["num_warps"], cfg["waves_per_eu"]
                print(f"  [{i}] {t:.1f}us BM={bm} BN={bn} BK={bk} KS={ks} W={nw} WPE={wpe}", file=sys.stderr)
        except Exception:
            continue

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
        _sweep(A, _bq_u8, _bscale_raw, m, n, k)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = _best_configs.get(key)
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg)
