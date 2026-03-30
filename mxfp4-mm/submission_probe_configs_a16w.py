#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Probe: dump _get_config defaults for A16WFP4 at our benchmark sizes + time each."""
from task import input_t, output_t
import torch
import sys
import time

_probed = False


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    import inspect
    print("=== _get_config defaults for A16WFP4 ===", file=sys.stderr)
    try:
        from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import _get_config
        # Source
        src = inspect.getsource(_get_config)
        print(f"_get_config source ({len(src)} chars):", file=sys.stderr)
        print(src[:3000], file=sys.stderr)

        # Test our benchmark sizes
        sizes = [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]
        for m, n, k in sizes:
            try:
                cfg, name = _get_config(m, n, k)
                print(f"\n  M={m},N={n},K={k}: config={cfg}, name={name}", file=sys.stderr)
            except Exception as e:
                print(f"\n  M={m},N={n},K={k}: ERROR={e}", file=sys.stderr)

        # Also test preshuffle configs
        print("\n=== _get_config preshuffle ===", file=sys.stderr)
        for m, n, k in sizes:
            try:
                cfg, name = _get_config(m, n, k, True)
                print(f"  M={m},N={n},K={k} (preshuffle): config={cfg}, name={name}", file=sys.stderr)
            except Exception as e:
                print(f"  M={m},N={n},K={k} (preshuffle): ERROR={e}", file=sys.stderr)
    except Exception as e:
        print(f"_get_config import error: {e}", file=sys.stderr)

    # List all A16WFP4 / AFP4WFP4 config files and their contents
    import os, json
    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    print("\n=== gfx950 FP4 config file contents ===", file=sys.stderr)
    for f in sorted(os.listdir(config_dir)):
        if 'gfx950' in f and ('FP4' in f or 'AFP4' in f):
            fpath = os.path.join(config_dir, f)
            try:
                with open(fpath) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    print(f"\n  {f} ({len(data)} entries):", file=sys.stderr)
                    for entry in data[:5]:
                        print(f"    {entry}", file=sys.stderr)
                    if len(data) > 5:
                        print(f"    ... ({len(data)-5} more)", file=sys.stderr)
                elif isinstance(data, dict):
                    print(f"\n  {f} (dict, {len(data)} keys):", file=sys.stderr)
                    for key in list(data.keys())[:5]:
                        print(f"    {key}: {data[key]}", file=sys.stderr)
            except Exception as e:
                print(f"\n  {f}: ERROR={e}", file=sys.stderr)

    # Dump kernel source for _gemm_a16wfp4_kernel (first 100 lines)
    print("\n=== _gemm_a16wfp4_kernel source ===", file=sys.stderr)
    try:
        from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import _gemm_a16wfp4_kernel
        src = inspect.getsource(_gemm_a16wfp4_kernel.fn)
        lines = src.split('\n')
        for line in lines[:100]:
            print(line, file=sys.stderr)
        if len(lines) > 100:
            print(f"... ({len(lines)-100} more lines)", file=sys.stderr)
    except Exception as e:
        print(f"kernel source error: {e}", file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
    _probe()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    bq_u8 = B_q.view(torch.uint8)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16)
