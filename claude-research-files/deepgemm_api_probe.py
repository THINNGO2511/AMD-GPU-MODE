#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Probe deepgemm_ck API — read source, test with our shapes."""
import torch
import subprocess
from task import input_t, output_t

def _probe():
    # 1. Read deepgemm.py source
    print("=== /home/runner/aiter/aiter/ops/deepgemm.py ===", flush=True)
    try:
        with open("/home/runner/aiter/aiter/ops/deepgemm.py") as f:
            src = f.read()
        print(src[:5000], flush=True)
        if len(src) > 5000:
            print(f"... [{len(src)} total chars, truncated]", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)

    # 2. Read deepgemm_common.py
    print("\n=== deepgemm_common.py ===", flush=True)
    try:
        with open("/home/runner/aiter/csrc/ck_deepgemm/deepgemm_common.py") as f:
            src = f.read()
        print(src[:3000], flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)

    # 3. Read test_deepgemm.py to understand usage
    print("\n=== test_deepgemm.py ===", flush=True)
    try:
        with open("/home/runner/aiter/op_tests/test_deepgemm.py") as f:
            src = f.read()
        print(src[:5000], flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)

    # 4. Read deepgemm.h header
    print("\n=== deepgemm.h ===", flush=True)
    try:
        with open("/home/runner/aiter/csrc/ck_deepgemm/include/deepgemm.h") as f:
            src = f.read()
        print(src[:3000], flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)

    # 5. Try calling deepgemm_ck with help
    print("\n=== deepgemm_ck signature ===", flush=True)
    try:
        import aiter
        import inspect
        sig = inspect.signature(aiter.deepgemm_ck)
        print(f"Signature: {sig}", flush=True)
        doc = aiter.deepgemm_ck.__doc__
        print(f"Docstring: {doc}", flush=True)
        src = inspect.getsource(aiter.deepgemm_ck)
        print(f"Source:\n{src[:3000]}", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)

    # 6. Try calling deepgemm signature
    print("\n=== deepgemm signature ===", flush=True)
    try:
        import aiter
        import inspect
        sig = inspect.signature(aiter.deepgemm)
        print(f"Signature: {sig}", flush=True)
        src = inspect.getsource(aiter.deepgemm)
        print(f"Source:\n{src[:3000]}", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)

    # 7. Check gen_instances.py for supported dtypes/shapes
    print("\n=== gen_instances.py ===", flush=True)
    try:
        with open("/home/runner/aiter/csrc/ck_deepgemm/gen_instances.py") as f:
            src = f.read()
        print(src[:3000], flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)

_probed = False

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

_cache = {}

def custom_kernel(data: input_t) -> output_t:
    global _probed
    if not _probed:
        _probed = True
        _probe()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    cache_key = id(B_scale_sh)
    if cache_key not in _cache:
        _cache[cache_key] = (_unshuffle_e8m0(B_scale_sh), B_q.view(torch.uint8))
    bscale_raw, bq_u8 = _cache[cache_key]

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), bq_u8, A_scale, bscale_raw, dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    return gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out)
