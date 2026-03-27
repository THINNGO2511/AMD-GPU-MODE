#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Config sweep for ALL slow sizes: K=7168, K=2048, K=1536.
Prints timing for each config. Uses default path for correctness.

Benchmark sizes:
  (4,2880,512), (16,2112,7168), (32,4096,512), (32,2880,512), (64,7168,2048), (256,3072,1536)
"""
from task import input_t, output_t
import torch
import sys
import time

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

_sweeps_done = set()

# K=7168 configs: M=16, N=2112. Current best 14.7us
_K7168_CFGS = [
    # Baseline (current best)
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # waves_per_eu=1 (more registers per WF)
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # num_stages=3 (PR #2160 showed +30% on gfx950)
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT variations
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 2048},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 7, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024},
    # BN=128
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # BN=32
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512},
    # BN=256
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # BM=16
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # 8 warps
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    # cache_modifier=".cg"
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": ".cg"},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": ".cg"},
    # BK=256 with higher KSPLIT
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512},
    # Best combos
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 14, "SPLITK_BLOCK_SIZE": 512},
]

# K=2048 configs: M=64, N=7168. Current best 14.2us (default)
_K2048_CFGS = [
    # Baseline (default = no config)
    None,
    # KSPLIT=2
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024},
    # KSPLIT=4
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512},
    # BN=128 no split
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 2048},
    # BN=128 KSPLIT=2
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024},
    # BN=128 KSPLIT=4
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512},
    # stages=3
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024},
    # wpe=1
    {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512},
    # BM=32/64 (M=64, so BM=32/64 tiles work)
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024},
    # BN=256 (N=7168 is large, big N tiles may help)
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024},
    # .cg
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": ".cg"},
]

# K=1536 configs: M=256, N=3072. Current best 16us (quant+afp4wfp4)
_K1536_CFGS = [
    # a16wfp4 with KSPLIT=3 (K=1536, 1536/3=512 per split)
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    # KSPLIT=1 (no split)
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1536},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1536},
    # stages=3
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    # wpe=1
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    # BN=256
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    # 8 warps
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512},
    # .cg
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 3, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": ".cg"},
    # BK=256 + higher KSPLIT
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "NUM_KSPLIT": 6, "SPLITK_BLOCK_SIZE": 256},
]


def _ensure_cm(cfg):
    """Ensure cache_modifier key exists (required by Triton kernel)."""
    if cfg is not None and "cache_modifier" not in cfg:
        cfg = {**cfg, "cache_modifier": None}
    return cfg


def _run_sweep(name, cfgs, A, bq, bscale, m, n, k):
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    results = []
    print(f"\n=== {name} SWEEP (M={m}, N={n}, K={k}) ===")

    for i, cfg in enumerate(cfgs):
        cfg = _ensure_cm(cfg)
        try:
            # Warmup (2 iters)
            for _ in range(2):
                gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()

            # Time (10 iters)
            iters = 10
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / iters * 1e6

            if cfg is None:
                desc = "DEFAULT"
            else:
                desc = f"BM={cfg['BLOCK_SIZE_M']},BN={cfg['BLOCK_SIZE_N']},BK={cfg['BLOCK_SIZE_K']},KS={cfg['NUM_KSPLIT']},w={cfg['num_warps']},s={cfg['num_stages']},wpe={cfg['waves_per_eu']},cm={cfg.get('cache_modifier','_')}"
            results.append((us, i, desc))
            print(f"  [{i:2d}] {us:7.1f}us | {desc}", flush=True)
        except Exception as e:
            print(f"  [{i:2d}] FAILED: {e}", flush=True)

    results.sort()
    print(f"\n  TOP 5:", flush=True)
    for us, i, desc in results[:5]:
        print(f"    {us:7.1f}us | [{i}] {desc}", flush=True)
    print(f"=== END {name} ===\n", flush=True)
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

    # Run sweep for each K on first encounter
    if k not in _sweeps_done:
        _sweeps_done.add(k)
        if k == 7168:
            _run_sweep("K7168", _K7168_CFGS, A, _bq_u8, _bscale_raw, m, n, k)
        elif k == 2048:
            _run_sweep("K2048", _K2048_CFGS, A, _bq_u8, _bscale_raw, m, n, k)
        elif k == 1536:
            _run_sweep("K1536", _K1536_CFGS, A, _bq_u8, _bscale_raw, m, n, k)

    # Correctness path: use known good configs
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
           "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024} if k == 7168 else None

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
