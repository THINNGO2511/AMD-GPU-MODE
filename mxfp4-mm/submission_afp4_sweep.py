#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Test quant+afp4wfp4 for ALL sizes with tuned per-N-K configs.
The afp4wfp4 path has 69 gfx950 FP4 config files with per-shape tuning.
Compare: a16wfp4 (no quant, on-the-fly) vs quant+afp4wfp4 (pre-quant, tuned GEMM)
"""
from task import input_t, output_t
import torch
import sys
import time

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_probed = False

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _probe_and_time(A, bq, bscale, m, n, k):
    global _probed
    if _probed:
        return
    _probed = True

    import os, json, inspect

    # 1. List matching config files
    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    if os.path.isdir(config_dir):
        print(f"\n=== gfx950 FP4 config files matching our N values ===", flush=True)
        our_ns = {'2880', '2112', '4096', '7168', '3072'}
        for f in sorted(os.listdir(config_dir)):
            if 'gfx950' in f and 'FP4' in f:
                # Check if any of our N values match
                matching = [n for n in our_ns if n in f]
                if matching:
                    fpath = os.path.join(config_dir, f)
                    with open(fpath) as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        print(f"\n  {f} ({len(data)} entries):", flush=True)
                        for entry in data[:5]:
                            print(f"    {entry}", flush=True)
                    elif isinstance(data, dict):
                        print(f"\n  {f} (dict):", flush=True)
                        for key in list(data.keys())[:5]:
                            print(f"    {key}: {data[key]}", flush=True)

    # 2. Time both paths for current size
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    sizes = [(4, 2880, 512), (16, 2112, 7168), (32, 4096, 512),
             (32, 2880, 512), (64, 7168, 2048), (256, 3072, 1536)]

    print(f"\n=== Timing comparison for M={m},N={n},K={k} ===", flush=True)

    # Time a16wfp4 default
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    for _ in range(3):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y)
    torch.cuda.synchronize()
    a16_us = (time.perf_counter() - t0) / 20 * 1e6
    print(f"  a16wfp4 default: {a16_us:.1f}us", flush=True)

    # Time quant+afp4wfp4
    for _ in range(3):
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        gemm_afp4wfp4(A_fp4.view(torch.uint8), bq, A_scale, bscale, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        gemm_afp4wfp4(A_fp4.view(torch.uint8), bq, A_scale, bscale, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    qfp4_us = (time.perf_counter() - t0) / 20 * 1e6
    print(f"  quant+afp4wfp4: {qfp4_us:.1f}us", flush=True)

    # Time just quant
    for _ in range(3):
        dynamic_mxfp4_quant(A)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        dynamic_mxfp4_quant(A)
    torch.cuda.synchronize()
    quant_us = (time.perf_counter() - t0) / 20 * 1e6
    print(f"  quant only: {quant_us:.1f}us", flush=True)
    print(f"  afp4wfp4 GEMM only: {qfp4_us - quant_us:.1f}us", flush=True)

    # Time a16wfp4 with KSPLIT=10
    if k == 7168:
        cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
               "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
               "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
               "NUM_KSPLIT": 10, "SPLITK_BLOCK_SIZE": 1024}
        for _ in range(3):
            gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
        torch.cuda.synchronize()
        ks10_us = (time.perf_counter() - t0) / 20 * 1e6
        print(f"  a16wfp4 KSPLIT=10: {ks10_us:.1f}us", flush=True)

    # 3. Check what afp4wfp4 _get_config returns for this size
    print(f"\n=== afp4wfp4 internal config for M={m},N={n},K={k} ===", flush=True)
    try:
        from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config
        cfg_result = _get_config(m, n, k)
        print(f"  config: {cfg_result}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)

    # Also try a16wfp4 _get_config
    try:
        from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import _get_config
        cfg_result = _get_config(m, n, k)
        print(f"  a16wfp4 config: {cfg_result}", flush=True)
    except Exception as e:
        print(f"  a16wfp4 Error: {e}", flush=True)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    _probe_and_time(A, _bq_u8, _bscale_raw, m, n, k)

    # Use proven path for correctness
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
           "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024} if k == 7168 else None

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
