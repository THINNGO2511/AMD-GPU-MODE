#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Comprehensive split-K sweep for ALL slow cases:
- K=7168 (M=16, N=2112): Currently 14.4us with KSPLIT=8
- K=2048 (M=64, N=7168): Currently 14.1us with KSPLIT=1 (default)
- K=1536 (M=256, N=3072): Currently 16us with quant+afp4wfp4
Sweep KSPLIT values and block sizes with CUDA events for accurate timing.
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

def _make_cfg(bm, bn, bk, gsm, nw, ns, wpe, ks, sbs, cm=None, mi=16):
    return {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": gsm, "num_warps": nw, "num_stages": ns,
            "waves_per_eu": wpe, "matrix_instr_nonkdim": mi, "cache_modifier": cm,
            "NUM_KSPLIT": ks, "SPLITK_BLOCK_SIZE": sbs}

# K=7168 configs: BK=512 → 14 iters. Try various KSPLIT.
_K7168_CFGS = [
    # Current best: KSPLIT=8
    _make_cfg(8, 64, 512, 1, 4, 2, 2, 8, 1024),
    # KSPLIT=14 (max parallelism, 1 iter each)
    _make_cfg(8, 64, 512, 1, 4, 2, 2, 14, 512),
    _make_cfg(8, 64, 512, 1, 4, 3, 2, 14, 512),
    _make_cfg(8, 128, 512, 1, 4, 2, 2, 14, 512),
    _make_cfg(8, 32, 512, 1, 4, 2, 2, 14, 512),
    # KSPLIT=7 (2 iters each)
    _make_cfg(8, 64, 512, 1, 4, 2, 2, 7, 1024),
    _make_cfg(8, 64, 512, 1, 4, 3, 2, 7, 1024),
    _make_cfg(8, 128, 512, 1, 4, 2, 2, 7, 1024),
    # KSPLIT=4 (3-4 iters each)
    _make_cfg(8, 64, 512, 1, 4, 2, 2, 4, 2048),
    _make_cfg(8, 128, 512, 1, 4, 2, 2, 4, 2048),
    # stages=3 variants (claimed +30% on gfx950)
    _make_cfg(8, 64, 512, 1, 4, 3, 2, 8, 1024),
    _make_cfg(8, 64, 512, 1, 4, 3, 1, 8, 1024),
    # waves_per_eu=1
    _make_cfg(8, 64, 512, 1, 4, 2, 1, 8, 1024),
    _make_cfg(8, 64, 512, 1, 4, 2, 1, 14, 512),
    # BN=128 + stages=3
    _make_cfg(8, 128, 512, 1, 4, 3, 2, 8, 1024),
    # cache_modifier=".cg"
    _make_cfg(8, 64, 512, 1, 4, 2, 2, 8, 1024, ".cg"),
    _make_cfg(8, 64, 512, 1, 4, 2, 2, 14, 512, ".cg"),
    # 8 warps
    _make_cfg(8, 64, 512, 1, 8, 2, 2, 8, 1024),
    _make_cfg(8, 64, 512, 1, 8, 2, 2, 14, 512),
    # BM=16
    _make_cfg(16, 64, 512, 1, 4, 2, 2, 8, 1024),
    _make_cfg(16, 64, 512, 1, 4, 2, 2, 14, 512),
    # BK=256 + higher KSPLIT
    _make_cfg(8, 64, 256, 1, 4, 2, 2, 14, 512),
    _make_cfg(8, 64, 256, 1, 4, 2, 2, 28, 256),
    # KSPLIT=2
    _make_cfg(8, 64, 512, 1, 4, 2, 2, 2, 4096),
]

# K=2048 configs: BK=512 → 4 iters. Currently defaults to KSPLIT=1.
_K2048_CFGS = [
    # KSPLIT=1 (current, ~14.2us)
    _make_cfg(8, 128, 512, 1, 4, 2, 2, 1, 2048),
    # KSPLIT=2
    _make_cfg(8, 128, 512, 1, 4, 2, 2, 2, 1024),
    _make_cfg(16, 128, 512, 1, 4, 2, 2, 2, 1024),
    _make_cfg(32, 128, 512, 1, 4, 2, 2, 2, 1024),
    _make_cfg(64, 128, 512, 1, 4, 2, 2, 2, 1024),
    _make_cfg(8, 64, 512, 1, 4, 2, 2, 2, 1024),
    _make_cfg(16, 64, 512, 1, 4, 2, 2, 2, 1024),
    _make_cfg(32, 64, 512, 1, 4, 2, 2, 2, 1024),
    # KSPLIT=4 (1 iter each)
    _make_cfg(8, 128, 512, 1, 4, 2, 2, 4, 512),
    _make_cfg(16, 128, 512, 1, 4, 2, 2, 4, 512),
    _make_cfg(32, 128, 512, 1, 4, 2, 2, 4, 512),
    _make_cfg(64, 128, 512, 1, 4, 2, 2, 4, 512),
    _make_cfg(8, 64, 512, 1, 4, 2, 2, 4, 512),
    _make_cfg(16, 64, 512, 1, 4, 2, 2, 4, 512),
    # stages=3
    _make_cfg(8, 128, 512, 1, 4, 3, 2, 2, 1024),
    _make_cfg(32, 128, 512, 1, 4, 3, 2, 2, 1024),
    _make_cfg(8, 128, 512, 1, 4, 3, 2, 4, 512),
    # waves_per_eu=1
    _make_cfg(8, 128, 512, 1, 4, 2, 1, 2, 1024),
    _make_cfg(8, 128, 512, 1, 4, 2, 1, 4, 512),
    # BN=256
    _make_cfg(8, 256, 512, 1, 4, 2, 2, 2, 1024),
    _make_cfg(16, 256, 512, 1, 4, 2, 2, 2, 1024),
    # BK=256 + KSPLIT
    _make_cfg(8, 128, 256, 1, 4, 2, 2, 4, 512),
    _make_cfg(8, 128, 256, 1, 4, 2, 2, 8, 256),
    # cache_modifier
    _make_cfg(8, 128, 512, 1, 4, 2, 2, 2, 1024, ".cg"),
    _make_cfg(8, 128, 512, 1, 4, 2, 2, 4, 512, ".cg"),
]

# K=1536 a16wfp4 configs: BK=512 → 3 iters
_K1536_CFGS = [
    # KSPLIT=3 (1 iter each)
    _make_cfg(32, 128, 512, 1, 4, 2, 2, 3, 512),
    _make_cfg(64, 128, 512, 1, 4, 2, 2, 3, 512),
    _make_cfg(128, 128, 512, 1, 4, 2, 2, 3, 512),
    _make_cfg(32, 64, 512, 1, 4, 2, 2, 3, 512),
    _make_cfg(64, 64, 512, 1, 4, 2, 2, 3, 512),
    _make_cfg(128, 64, 512, 1, 4, 2, 2, 3, 512),
    # KSPLIT=1 (default)
    _make_cfg(32, 128, 512, 1, 4, 2, 2, 1, 1536),
    _make_cfg(64, 128, 512, 1, 4, 2, 2, 1, 1536),
    _make_cfg(128, 128, 512, 1, 4, 2, 2, 1, 1536),
    # stages=3
    _make_cfg(32, 128, 512, 1, 4, 3, 2, 3, 512),
    _make_cfg(64, 128, 512, 1, 4, 3, 2, 3, 512),
    # BK=256
    _make_cfg(64, 128, 256, 1, 4, 2, 2, 3, 512),
    _make_cfg(64, 128, 256, 1, 4, 2, 2, 6, 256),
    _make_cfg(128, 128, 256, 1, 4, 2, 2, 6, 256),
]

def _sweep_configs(A, w, w_scales, m, n, k, configs, label):
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    print(f"\n=== {label} SWEEP (M={m},N={n},K={k}) ===")
    results = []
    for i, cfg in enumerate(configs):
        try:
            # Warmup
            gemm_a16wfp4(A, w, w_scales, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()

            # Time with CUDA events (20 iterations)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
            for j in range(20):
                starts[j].record()
                gemm_a16wfp4(A, w, w_scales, dtype=torch.bfloat16, y=y, config=cfg)
                ends[j].record()
            torch.cuda.synchronize()
            times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
            median = times[10]  # median of 20

            desc = f"BM={cfg['BLOCK_SIZE_M']:3d} BN={cfg['BLOCK_SIZE_N']:3d} BK={cfg['BLOCK_SIZE_K']:3d} KS={cfg['NUM_KSPLIT']:2d} W={cfg['num_warps']} S={cfg['num_stages']} WPE={cfg['waves_per_eu']} CM={cfg.get('cache_modifier','')}"
            results.append((median, i, desc))
            print(f"  [{i:2d}] {median:7.1f}us | {desc}")
        except Exception as e:
            print(f"  [{i:2d}] FAILED | {str(e)[:80]}")

    results.sort()
    print(f"\n  TOP-5 for {label}:")
    for t, idx, desc in results[:5]:
        print(f"    {t:7.1f}us | cfg[{idx}] {desc}")
    return results

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
            _sweep_configs(A, _bq_u8, _bscale_raw, m, n, k, _K7168_CFGS, "K7168")
        elif k == 2048:
            _sweep_configs(A, _bq_u8, _bscale_raw, m, n, k, _K2048_CFGS, "K2048")
        elif k == 1536:
            _sweep_configs(A, _bq_u8, _bscale_raw, m, n, k, _K1536_CFGS, "K1536")

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    if k == 1536:
        # Fall back to quant+afp4wfp4 for benchmark (known good)
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    cfg = _make_cfg(8, 64, 512, 1, 4, 2, 2, 8, 1024) if k == 7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
