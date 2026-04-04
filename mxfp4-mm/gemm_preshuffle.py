#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — gemm_a16wfp4_preshuffle with B_shuffle input.

Uses CK's preshuffle pipeline — B_shuffle is already in the right format.
Skips unshuffle overhead entirely. Last attempt failed with Triton e8m0
dtype KeyError. This version probes the API and reports what it needs.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

from task import input_t, output_t
import torch

_call = 0
_yc = {}
_fallback = False

def custom_kernel(data: input_t) -> output_t:
    global _call, _fallback
    _call += 1
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    if not _fallback:
        try:
            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
            key = (m, n)
            if key not in _yc:
                _yc[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
            out = _yc[key]

            if _call <= 2:
                print(f"[PRESHUFFLE] Trying: A={A.shape} B_shuffle={B_shuffle.shape} "
                      f"B_scale_sh dtype={B_scale_sh.dtype} shape={B_scale_sh.shape}", flush=True)

            # Try 1: B_shuffle + B_scale_sh as-is
            try:
                gemm_a16wfp4_preshuffle(A, B_shuffle.view(torch.uint8), B_scale_sh.view(torch.uint8),
                                        dtype=torch.bfloat16, y=out)
                if _call <= 2:
                    print(f"[PRESHUFFLE] Try1 SUCCESS", flush=True)
                return out
            except Exception as e:
                if _call <= 2:
                    print(f"[PRESHUFFLE] Try1 fail: {e}", flush=True)

            # Try 2: B_shuffle as fp4x2 + B_scale_sh as e8m0
            try:
                gemm_a16wfp4_preshuffle(A, B_shuffle, B_scale_sh,
                                        dtype=torch.bfloat16, y=out)
                if _call <= 2:
                    print(f"[PRESHUFFLE] Try2 SUCCESS", flush=True)
                return out
            except Exception as e:
                if _call <= 2:
                    print(f"[PRESHUFFLE] Try2 fail: {e}", flush=True)

            # Try 3: with config=None
            try:
                gemm_a16wfp4_preshuffle(A, B_shuffle, B_scale_sh,
                                        dtype=torch.bfloat16, y=out, config=None)
                if _call <= 2:
                    print(f"[PRESHUFFLE] Try3 SUCCESS", flush=True)
                return out
            except Exception as e:
                if _call <= 2:
                    print(f"[PRESHUFFLE] Try3 fail: {e}", flush=True)

            # Try 4: Inspect function signature
            if _call <= 2:
                import inspect
                sig = inspect.signature(gemm_a16wfp4_preshuffle)
                print(f"[PRESHUFFLE] Signature: {sig}", flush=True)

            _fallback = True
        except ImportError as e:
            if _call <= 2:
                print(f"[PRESHUFFLE] Import fail: {e}", flush=True)
            _fallback = True

    # Fallback: standard gemm_a16wfp4
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    su = B_scale_sh.view(torch.uint8); sm, sn = su.shape
    d0, d1 = sm // 32, sn // 8; total = sm * sn
    idx = torch.arange(total, dtype=torch.int64, device=su.device)
    idx = idx.view(d0, d1, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(-1)
    bscale_raw = torch.take(su.reshape(-1), idx).view(sm, sn)
    bq_u8 = B_q.view(torch.uint8)

    key = (m, n)
    if key not in _yc:
        _yc[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _yc[key]

    if k == 1536:
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), bq_u8, asc, bscale_raw, dtype=torch.bfloat16)
    gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out, config=None)
    return out
