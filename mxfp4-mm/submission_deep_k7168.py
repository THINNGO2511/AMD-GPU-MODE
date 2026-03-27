#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Deep sweep around KSPLIT=10 for K=7168 (M=16, N=2112).
KSPLIT=10 was 16% faster than KSPLIT=8 in previous sweep.
Also probe: what does the default config path actually use?
"""
from task import input_t, output_t
import torch
import sys
import time

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_sweeps_done = set()

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def _ensure_cm(cfg):
    if cfg is not None and "cache_modifier" not in cfg:
        cfg = {**cfg, "cache_modifier": None}
    return cfg

# Deep sweep around KSPLIT=10 winner
_K7168_DEEP = [
    # Baseline KSPLIT=8
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT=9,10,11,12
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 9, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 11, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 12, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT=10 + wpe=1
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT=10 + stages=3
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT=10 + stages=3 + wpe=1
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT=10 + BN=128
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT=10 + BM=16
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT=10 + .cg
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": ".cg"},
    # KSPLIT=14 (also good in first sweep)
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512},
    # KSPLIT=14 + wpe=1
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512},
    # KSPLIT=14 + stages=3
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512},
    # KSPLIT=10 + SPLITK_BLOCK_SIZE=512
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 512},
    # KSPLIT=10 + SPLITK_BLOCK_SIZE=2048
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 2048},
]

# Also probe: what default config does the kernel actually auto-select?
_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True
    import inspect
    print("\n=== PROBING gemm_a16wfp4 internals ===", flush=True)
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        src = inspect.getsource(gemm_a16wfp4)
        print(f"gemm_a16wfp4 wrapper ({len(src)} chars):", flush=True)
        for line in src.split('\n')[:60]:
            print(line, flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

    # Also check what gemm_a16wfp4_ looks like
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_
        src = inspect.getsource(gemm_a16wfp4_)
        print(f"\ngemm_a16wfp4_ ({len(src)} chars):", flush=True)
        for line in src.split('\n')[:80]:
            print(line, flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    # Probe once
    _probe()

    # Sweep K=7168 once
    if k == 7168 and k not in _sweeps_done:
        _sweeps_done.add(k)
        y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        results = []
        print(f"\n=== DEEP K7168 SWEEP (M={m}, N={n}) ===", flush=True)
        for i, cfg in enumerate(_K7168_DEEP):
            cfg = _ensure_cm(cfg)
            try:
                for _ in range(3):
                    gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
                torch.cuda.synchronize()
                iters = 20
                t0 = time.perf_counter()
                for _ in range(iters):
                    gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
                torch.cuda.synchronize()
                us = (time.perf_counter() - t0) / iters * 1e6
                desc = f"KS={cfg['NUM_KSPLIT']:2d},BN={cfg['BLOCK_SIZE_N']:3d},BM={cfg['BLOCK_SIZE_M']:2d},s={cfg['num_stages']},wpe={cfg['waves_per_eu']},SBS={cfg['SPLITK_BLOCK_SIZE']},cm={cfg.get('cache_modifier','_')}"
                results.append((us, i, desc))
                print(f"  [{i:2d}] {us:7.1f}us | {desc}", flush=True)
            except Exception as e:
                print(f"  [{i:2d}] FAILED: {e}", flush=True)
        results.sort()
        print(f"\n  TOP 5:", flush=True)
        for us, i, desc in results[:5]:
            print(f"    {us:7.1f}us | [{i}] {desc}", flush=True)
        print(f"=== END DEEP K7168 ===\n", flush=True)

    # Correctness: use KSPLIT=10 for K=7168
    cfg = _ensure_cm({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
           "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024}) if k == 7168 else None

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
