#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Targeted config sweep for the two slowest shapes.
K=7168 (M=16,N=2112): tries ~8 configs around current best
K=2048 (M=64,N=7168): tries ~6 configs around current best
Prints timing for each, picks best. JIT compiles per config.
"""
from task import input_t, output_t
import torch
import time
import sys

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_best_configs = {}
_swept = set()


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# Current best configs
_K7168_BASE = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

_K2048_BASE = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
    "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 4096,
}


def _make_configs_k7168():
    """Generate configs to try for K=7168 shape."""
    base = dict(_K7168_BASE)
    configs = [("base", dict(base))]

    # Vary KSPLIT (most impactful parameter)
    for ks in [4, 6, 12, 16]:
        c = dict(base)
        c["NUM_KSPLIT"] = ks
        configs.append((f"ksplit={ks}", c))

    # Vary BN
    for bn in [32, 128]:
        c = dict(base)
        c["BLOCK_SIZE_N"] = bn
        configs.append((f"bn={bn}", c))

    # Vary BM
    for bm in [16, 32]:
        c = dict(base)
        c["BLOCK_SIZE_M"] = bm
        configs.append((f"bm={bm}", c))

    # stages=3
    c = dict(base)
    c["num_stages"] = 3
    configs.append(("stages=3", c))

    # waves=1
    c = dict(base)
    c["waves_per_eu"] = 1
    configs.append(("waves=1", c))

    # waves=4
    c = dict(base)
    c["waves_per_eu"] = 4
    configs.append(("waves=4", c))

    # warps=8
    c = dict(base)
    c["num_warps"] = 8
    configs.append(("warps=8", c))

    # BK=256
    c = dict(base)
    c["BLOCK_SIZE_K"] = 256
    configs.append(("bk=256", c))

    # cache_modifier=.cg
    c = dict(base)
    c["cache_modifier"] = ".cg"
    configs.append(("cg", c))

    # Combo: stages=3 + ksplit=12
    c = dict(base)
    c["num_stages"] = 3
    c["NUM_KSPLIT"] = 12
    configs.append(("s3_ks12", c))

    # Combo: bm=16 + ksplit=12
    c = dict(base)
    c["BLOCK_SIZE_M"] = 16
    c["NUM_KSPLIT"] = 12
    configs.append(("bm16_ks12", c))

    return configs


def _make_configs_k2048():
    """Generate configs to try for K=2048 shape."""
    base = dict(_K2048_BASE)
    configs = [("base", dict(base))]

    # Vary KSPLIT
    for ks in [2, 4]:
        c = dict(base)
        c["NUM_KSPLIT"] = ks
        configs.append((f"ksplit={ks}", c))

    # stages=3
    c = dict(base)
    c["num_stages"] = 3
    configs.append(("stages=3", c))

    # Vary BM
    for bm in [32, 64]:
        c = dict(base)
        c["BLOCK_SIZE_M"] = bm
        configs.append((f"bm={bm}", c))

    # waves=2
    c = dict(base)
    c["waves_per_eu"] = 2
    configs.append(("waves=2", c))

    # BN=64
    c = dict(base)
    c["BLOCK_SIZE_N"] = 64
    configs.append(("bn=64", c))

    # warps=4
    c = dict(base)
    c["num_warps"] = 4
    configs.append(("warps=4", c))

    # Combo: stages=3 + waves=2
    c = dict(base)
    c["num_stages"] = 3
    c["waves_per_eu"] = 2
    configs.append(("s3_w2", c))

    # Combo: bm=32 + bn=64
    c = dict(base)
    c["BLOCK_SIZE_M"] = 32
    c["BLOCK_SIZE_N"] = 64
    configs.append(("bm32_bn64", c))

    return configs


def _sweep_shape(A, bq_u8, bscale_raw, k, m, n, configs):
    """Run mini-sweep for a shape, return best config."""
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    results = []

    for name, cfg in configs:
        try:
            # Warmup (triggers JIT compile)
            gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()

            # Time 10 runs
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
            end.record()
            torch.cuda.synchronize()
            elapsed_us = start.elapsed_time(end) * 100  # ms*1000/10 = us per call
            results.append((elapsed_us, name, cfg))
            print(f"SWEEP K={k}: {name:20s} = {elapsed_us:.1f} us", flush=True)
        except Exception as e:
            print(f"SWEEP K={k}: {name:20s} FAILED: {e}", flush=True)

    results.sort(key=lambda x: x[0])

    print(f"\nSWEEP K={k} TOP 5:", flush=True)
    for t, name, cfg in results[:5]:
        print(f"  {t:.1f} us  {name}  {cfg}", flush=True)
    print(flush=True)

    if results:
        return results[0][2]  # best config
    return configs[0][1]  # fallback to base


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    # K=1536: always use quant+afp4wfp4 path (proven best)
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    # Run sweep on first call for K=7168 and K=2048
    if k == 7168 and k not in _swept:
        _swept.add(k)
        print(f"\n=== SWEEPING K=7168 (M={m}, N={n}) ===", flush=True)
        _best_configs[k] = _sweep_shape(A, _bq_u8, _bscale_raw, k, m, n, _make_configs_k7168())

    if k == 2048 and k not in _swept:
        _swept.add(k)
        print(f"\n=== SWEEPING K=2048 (M={m}, N={n}) ===", flush=True)
        _best_configs[k] = _sweep_shape(A, _bq_u8, _bscale_raw, k, m, n, _make_configs_k2048())

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    cfg = _best_configs.get(k, _K7168_BASE if k == 7168 else (_K2048_BASE if k == 2048 else None))

    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)
