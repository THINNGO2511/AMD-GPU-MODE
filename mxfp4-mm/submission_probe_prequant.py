#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Probe: Test gemm_afp4wfp4_pre_quant_atomic and gemm_a8wfp4.
Also dump available config files and kernel source.
"""
from task import input_t, output_t
import torch
import sys
import inspect

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

    import os

    # 1. Check gemm_afp4wfp4_pre_quant_atomic
    print("=== gemm_afp4wfp4_pre_quant_atomic ===", file=sys.stderr)
    try:
        import aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic as pqa
        names = [x for x in dir(pqa) if not x.startswith('_')]
        print(f"  Exports: {names}", file=sys.stderr)
        for name in names:
            obj = getattr(pqa, name)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    print(f"  {name}: {sig}", file=sys.stderr)
                except:
                    print(f"  {name}: (no signature)", file=sys.stderr)

        # Dump source
        fpath = "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_afp4wfp4_pre_quant_atomic.py"
        if os.path.exists(fpath):
            with open(fpath) as f:
                lines = f.readlines()
            print(f"\n  Source ({len(lines)} lines):", file=sys.stderr)
            for i, line in enumerate(lines[:200]):
                print(f"  {i+1:4d}: {line}", end='', file=sys.stderr)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    # 2. Check gemm_a8wfp4
    print("\n=== gemm_a8wfp4 ===", file=sys.stderr)
    try:
        import aiter.ops.triton.gemm.basic.gemm_a8wfp4 as a8w
        names = [x for x in dir(a8w) if not x.startswith('_')]
        print(f"  Exports: {names}", file=sys.stderr)
        for name in names:
            obj = getattr(a8w, name)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    print(f"  {name}: {sig}", file=sys.stderr)
                except:
                    pass
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    # 3. Dump the _gemm_a16wfp4_kernel source (inner Triton kernel)
    print("\n=== Inner kernel source ===", file=sys.stderr)
    kpath = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py"
    if os.path.exists(kpath):
        with open(kpath) as f:
            lines = f.readlines()
        print(f"  {kpath} ({len(lines)} lines):", file=sys.stderr)
        for i, line in enumerate(lines[:300]):
            print(f"  {i+1:4d}: {line}", end='', file=sys.stderr)
        if len(lines) > 300:
            print(f"  ... ({len(lines)-300} more lines)", file=sys.stderr)

    # 4. List per-N-K config files for AFP4WFP4 at our benchmark sizes
    print("\n=== Per-N-K configs matching our sizes ===", file=sys.stderr)
    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    for f in sorted(os.listdir(config_dir)):
        if 'gfx950' in f and any(s in f for s in ['2880', '2112', '4096', '7168', '3072']):
            print(f"  {f}", file=sys.stderr)
            import json
            try:
                with open(os.path.join(config_dir, f)) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    for entry in data[:3]:
                        print(f"    {entry}", file=sys.stderr)
                elif isinstance(data, dict):
                    for key in list(data.keys())[:3]:
                        print(f"    {key}: {data[key]}", file=sys.stderr)
            except:
                pass


def custom_kernel(data: input_t) -> output_t:
    _probe()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    bq_u8 = B_q.view(torch.uint8)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16)
