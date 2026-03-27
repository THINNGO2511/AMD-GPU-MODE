#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Sweep v4: Test BM=4 for K=7168, afp4wfp4+config for K=2048/K=1536.
"""
from task import input_t, output_t
import torch

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

def _mc(bm, bn, bk, gsm, nw, ns, wpe, ks, sbs, cm=None, mi=16):
    return {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": gsm, "num_warps": nw, "num_stages": ns,
            "waves_per_eu": wpe, "matrix_instr_nonkdim": mi, "cache_modifier": cm,
            "NUM_KSPLIT": ks, "SPLITK_BLOCK_SIZE": sbs}

# K=7168: try BM=4 (default auto uses BM=4), .cg, stages=3
_K7168_CFGS = [
    _mc(8, 64, 512, 1, 4, 2, 2, 8, 1024),           # current best
    _mc(4, 64, 512, 1, 4, 2, 2, 8, 1024),            # BM=4
    _mc(4, 128, 512, 1, 4, 2, 2, 8, 1024),           # BM=4,BN=128
    _mc(4, 64, 512, 1, 4, 3, 2, 8, 1024),            # BM=4,stages=3
    _mc(8, 64, 512, 1, 4, 3, 2, 8, 1024),            # stages=3
    _mc(4, 64, 512, 1, 4, 2, 2, 14, 512),            # BM=4,KSPLIT=14
    _mc(8, 64, 512, 1, 4, 2, 2, 8, 1024, ".cg"),     # .cg
    _mc(4, 64, 512, 1, 4, 2, 2, 8, 1024, ".cg"),     # BM=4,.cg
    _mc(4, 128, 512, 1, 4, 1, 2, 8, 1024, ".cg"),    # auto-like
    _mc(4, 64, 512, 1, 4, 3, 2, 7, 1024),            # BM=4,s3,KS=7
    _mc(8, 64, 512, 1, 4, 3, 2, 7, 1024),            # s3,KS=7
]

def _sweep(label, A, w, ws, m, n, k, configs):
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    print(f"\n=== {label} (M={m},N={n},K={k}) ===")
    for i, cfg in enumerate(configs):
        try:
            gemm_a16wfp4(A, w, ws, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
            for j in range(20):
                starts[j].record()
                gemm_a16wfp4(A, w, ws, dtype=torch.bfloat16, y=y, config=cfg)
                ends[j].record()
            torch.cuda.synchronize()
            times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
            median = times[10]
            desc = f"BM={cfg['BLOCK_SIZE_M']:3d} BN={cfg['BLOCK_SIZE_N']:3d} KS={cfg['NUM_KSPLIT']:2d} S={cfg['num_stages']} CM={cfg.get('cache_modifier','')}"
            print(f"  [{i:2d}] {median:7.1f}us | {desc}")
        except Exception as e:
            print(f"  [{i:2d}] FAILED | {str(e)[:80]}")

def _sweep_afp4(label, A, w, ws, m, n, k):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    print(f"\n=== {label} afp4wfp4 (M={m},N={n},K={k}) ===")
    configs = [
        _mc(8, 128, 512, 1, 4, 2, 2, 1, 2*k),
        _mc(8, 128, 512, 1, 4, 2, 2, 2, 1024),
        _mc(8, 128, 512, 1, 4, 2, 2, 4, 512),
        _mc(32, 128, 512, 1, 4, 2, 2, 1, 2*k),
        _mc(64, 128, 512, 1, 4, 2, 2, 1, 2*k),
        _mc(8, 128, 256, 1, 4, 2, 2, 1, 2*k),
    ]
    for i, cfg in enumerate(configs):
        try:
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            gemm_afp4wfp4(A_fp4.view(torch.uint8), w, A_scale, ws,
                          dtype=torch.bfloat16, config=cfg)
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            for j in range(10):
                starts[j].record()
                A_fp4, A_scale = dynamic_mxfp4_quant(A)
                gemm_afp4wfp4(A_fp4.view(torch.uint8), w, A_scale, ws,
                              dtype=torch.bfloat16, config=cfg)
                ends[j].record()
            torch.cuda.synchronize()
            times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
            median = times[5]
            desc = f"BM={cfg['BLOCK_SIZE_M']:3d} BN={cfg['BLOCK_SIZE_N']:3d} BK={cfg['BLOCK_SIZE_K']:3d} KS={cfg['NUM_KSPLIT']:2d}"
            print(f"  [{i:2d}] {median:7.1f}us (quant+gemm) | {desc}")
        except Exception as e:
            print(f"  [{i:2d}] FAILED | {str(e)[:80]}")

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
            _sweep("K7168", A, _bq_u8, _bscale_raw, m, n, k, _K7168_CFGS)
        elif k == 2048:
            _sweep_afp4("K2048", A, _bq_u8, _bscale_raw, m, n, k)
        elif k == 1536:
            _sweep_afp4("K1536", A, _bq_u8, _bscale_raw, m, n, k)

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    cfg = _mc(8, 64, 512, 1, 4, 2, 2, 8, 1024) if k == 7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
