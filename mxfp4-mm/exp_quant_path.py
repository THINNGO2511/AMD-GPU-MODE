#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
TARGETED PROBE: Find the exact import path for _mxfp4_quant_op.
Only prints what we need — no full source dumps to avoid truncation.
"""
from task import input_t, output_t
import torch
import sys
import os

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_probed = False

def P(msg):
    print(f"Q: {msg}", file=sys.stderr)


def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. Search for _mxfp4_quant_op in aiter source
    P("=== SEARCHING FOR _mxfp4_quant_op ===")
    aiter_root = "/home/runner/aiter/aiter"
    for root, dirs, files in os.walk(aiter_root):
        for f in files:
            if f.endswith('.py'):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath) as fh:
                        content = fh.read()
                    if '_mxfp4_quant_op' in content:
                        rel = os.path.relpath(fpath, "/home/runner/aiter")
                        P(f"FOUND in: {rel}")
                        # Print context around the function
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if '_mxfp4_quant_op' in line:
                                start = max(0, i - 2)
                                end = min(len(lines), i + 10)
                                for j in range(start, end):
                                    P(f"  {j+1:4d}| {lines[j]}")
                                P("  ---")
                except Exception:
                    pass

    # 2. Also search for dot_scaled usage
    P("\n=== SEARCHING FOR tl.dot_scaled USAGE ===")
    for root, dirs, files in os.walk(aiter_root):
        for f in files:
            if f.endswith('.py'):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath) as fh:
                        content = fh.read()
                    if 'dot_scaled' in content:
                        rel = os.path.relpath(fpath, "/home/runner/aiter")
                        P(f"dot_scaled in: {rel}")
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'dot_scaled' in line:
                                P(f"  {i+1:4d}| {line.strip()}")
                except Exception:
                    pass

    # 3. List all Triton kernel quant files
    P("\n=== TRITON QUANT FILES ===")
    quant_dirs = [
        "/home/runner/aiter/aiter/ops/triton/quant.py",
        "/home/runner/aiter/aiter/ops/triton/_triton_kernels/",
        "/home/runner/aiter/aiter/ops/triton/gemm/basic/",
    ]
    for p in quant_dirs:
        if os.path.isfile(p):
            P(f"FILE: {p} ({os.path.getsize(p)} bytes)")
        elif os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                fp = os.path.join(p, f)
                if os.path.isfile(fp):
                    P(f"  {f} ({os.path.getsize(fp)} bytes)")

    # 4. Read the first 50 lines of quant.py for _mxfp4_quant_op function signature
    P("\n=== quant.py TOP 50 LINES ===")
    try:
        with open("/home/runner/aiter/aiter/ops/triton/quant.py") as fh:
            lines = fh.readlines()[:50]
        for i, line in enumerate(lines):
            P(f"  {i+1:4d}| {line.rstrip()}")
    except Exception as e:
        P(f"Error: {e}")

    # 5. Print gemm_a16wfp4.py imports and first 30 lines
    P("\n=== gemm_a16wfp4.py IMPORTS ===")
    try:
        with open("/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py") as fh:
            lines = fh.readlines()[:40]
        for i, line in enumerate(lines):
            P(f"  {i+1:4d}| {line.rstrip()}")
    except Exception as e:
        P(f"Error: {e}")

    # 6. Check if _triton_kernels dir exists
    P("\n=== _triton_kernels DIR ===")
    tkdir = "/home/runner/aiter/aiter/ops/triton/_triton_kernels"
    if os.path.exists(tkdir):
        for root, dirs, files in os.walk(tkdir):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), tkdir)
                P(f"  {rel}")
    else:
        P("_triton_kernels NOT FOUND")
        # Check what IS in ops/triton/
        for f in sorted(os.listdir("/home/runner/aiter/aiter/ops/triton/")):
            P(f"  ops/triton/{f}")


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    _probe()

    # Use proven approach
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16)
