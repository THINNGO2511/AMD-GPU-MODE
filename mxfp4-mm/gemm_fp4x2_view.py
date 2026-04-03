#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — gemm_a4w4 via fp4x2 dtype view.

FINDING: gemm_a4w4 rejects uint8 A. It needs torch.float4_e2m1fn_x2.
dynamic_mxfp4_quant returns uint8 but the BITS are fp4x2 — same data.

Fix: A_fp4.view(aiter_dtypes.fp4x2) — zero-cost reinterpret.

If gemm_a4w4 accepts this, it dispatches to ASM .co kernels.
The task also provides B_q as fp4x2 dtype and B_scale_sh as e8m0.

Try multiple API signatures since it changed:
1. gemm_a4w4(A_fp4x2, B_q, A_scale, B_scale_sh, bias, dtype, alpha, beta, bpreshuffle)
2. gemm_a4w4(A_fp4x2, B_shuffle, A_scale_shuffled, B_scale_sh, ...)

Print all errors for debugging.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

from task import input_t, output_t
import torch
import sys

_y_cache = {}
_call_count = 0

def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _call_count += 1
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)

    # Quantize A to fp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4_u8, A_scale = dynamic_mxfp4_quant(A)

    # Try fp4x2 dtype view
    try:
        from aiter import dtypes as aiter_dtypes
        fp4x2_dtype = aiter_dtypes.fp4x2
        A_fp4 = A_fp4_u8.view(fp4x2_dtype)

        if _call_count <= 2:
            print(f"[GEMM] A_fp4 dtype: {A_fp4.dtype}, shape: {A_fp4.shape}", flush=True)
            print(f"[GEMM] B_q dtype: {B_q.dtype}, shape: {B_q.shape}", flush=True)
            print(f"[GEMM] B_shuffle dtype: {B_shuffle.dtype}, shape: {B_shuffle.shape}", flush=True)
            print(f"[GEMM] A_scale dtype: {A_scale.dtype}, shape: {A_scale.shape}", flush=True)
            print(f"[GEMM] B_scale_sh dtype: {B_scale_sh.dtype}, shape: {B_scale_sh.shape}", flush=True)
    except Exception as e:
        if _call_count <= 2:
            print(f"[GEMM] fp4x2 view failed: {e}", flush=True)
        A_fp4 = A_fp4_u8

    # Attempt 1: gemm_a4w4 with B_shuffle + shuffled scales (bpreshuffle=1)
    try:
        from aiter import gemm_a4w4
        result = gemm_a4w4(
            A_fp4, B_shuffle, A_scale, B_scale_sh,
            None, torch.bfloat16, 1.0, 0.0, 1
        )
        if _call_count <= 2:
            print(f"[GEMM] Attempt 1 SUCCESS! result shape: {result.shape}", flush=True)
        return result
    except Exception as e:
        if _call_count <= 2:
            print(f"[GEMM] Attempt 1 failed: {e}", flush=True)

    # Attempt 2: gemm_a4w4 with B_q (raw, not shuffled) + unshuffled scales
    try:
        from aiter import gemm_a4w4
        # Unshuffle B scales
        su = B_scale_sh.view(torch.uint8)
        sm, sn = su.shape
        d0, d1 = sm // 32, sn // 8
        total = sm * sn
        idx = torch.arange(total, dtype=torch.int64, device=su.device)
        idx = idx.view(d0, d1, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(-1)
        bscale_raw = torch.take(su.reshape(-1), idx).view(sm, sn)
        # Try with raw B_q + unshuffled scales, bpreshuffle=0
        B_q_fp4 = B_q  # Already fp4x2 dtype from eval
        result = gemm_a4w4(
            A_fp4, B_q_fp4, A_scale, bscale_raw.view(B_scale_sh.dtype),
            None, torch.bfloat16, 1.0, 0.0, 0
        )
        if _call_count <= 2:
            print(f"[GEMM] Attempt 2 SUCCESS! result shape: {result.shape}", flush=True)
        return result
    except Exception as e:
        if _call_count <= 2:
            print(f"[GEMM] Attempt 2 failed: {e}", flush=True)

    # Attempt 3: gemm_a4w4 with shuffled A_scale
    try:
        from aiter import gemm_a4w4
        from aiter.utility.fp4_utils import e8m0_shuffle
        A_scale_sh = e8m0_shuffle(A_scale)
        result = gemm_a4w4(
            A_fp4, B_shuffle, A_scale_sh, B_scale_sh,
            None, torch.bfloat16, 1.0, 0.0, 1
        )
        if _call_count <= 2:
            print(f"[GEMM] Attempt 3 SUCCESS! result shape: {result.shape}", flush=True)
        return result
    except Exception as e:
        if _call_count <= 2:
            print(f"[GEMM] Attempt 3 failed: {e}", flush=True)

    # Attempt 4: Try torch.ops.aiter directly
    try:
        if _call_count <= 2:
            ops = [x for x in dir(torch.ops.aiter) if 'a4w4' in x.lower() or 'asm' in x.lower()]
            print(f"[GEMM] torch.ops.aiter a4w4/asm ops: {ops}", flush=True)
            for op_name in ops[:3]:
                try:
                    op = getattr(torch.ops.aiter, op_name)
                    print(f"[GEMM]   {op_name}: {op}", flush=True)
                except Exception:
                    pass
    except Exception:
        pass

    # Fallback to proven path
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    su = B_scale_sh.view(torch.uint8)
    sm, sn = su.shape
    d0, d1 = sm // 32, sn // 8
    total = sm * sn
    idx = torch.arange(total, dtype=torch.int64, device=su.device)
    idx = idx.view(d0, d1, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(-1)
    bscale_raw = torch.take(su.reshape(-1), idx).view(sm, sn)
    bq_u8 = B_q.view(torch.uint8)

    out = _y_cache[key]
    if k == 1536:
        return gemm_afp4wfp4(A_fp4_u8.view(torch.uint8), bq_u8, A_scale, bscale_raw, dtype=torch.bfloat16)
    else:
        gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out, config=None)
        return out
