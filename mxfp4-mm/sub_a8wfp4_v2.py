import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
import torch
import time
from task import input_t, output_t

_y_cache = {}
_probed = False

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def _probe(A, B_q, B_scale_sh, m, n, k):
    global _probed
    if _probed:
        return
    _probed = True

    from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    bq_u8 = B_q.view(torch.uint8)

    # gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config)
    # x = fp8 A with block scales, w = fp4 B, y = output
    # x_scales = A block scales (E8M0?), w_scales = B scales (E8M0)

    # First: quantize A to fp8 with block scales using aiter
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        # Get A_fp4 and A_scale — we want fp8 equivalent
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        print(f"[A8v2] A_fp4 shape={A_fp4.shape} dtype={A_fp4.dtype}")
        print(f"[A8v2] A_scale shape={A_scale.shape} dtype={A_scale.dtype}")
    except Exception as e:
        print(f"[A8v2] quant error: {e}")

    # Try: use MXFP4-quantized A as fp8 input? (wrong dtype but test API)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)

    # Attempt 1: fp4 A (from mxfp4 quant) + fp4 B — wrong A dtype but test call pattern
    try:
        gemm_a8wfp4(A_fp4.view(torch.uint8), bq_u8, out, A_scale, bscale_raw)
        print(f"[A8v2] fp4 A attempt: out={out[0,:3]}")
    except Exception as e:
        print(f"[A8v2] fp4 A: {str(e)[:300]}")

    # Attempt 2: Check for dynamic_fp8_quant or similar
    try:
        from aiter.ops.triton.quant import dynamic_per_token_fp8_quant
        A_fp8, A_fp8_scale = dynamic_per_token_fp8_quant(A)
        print(f"[A8v2] per_token_fp8: A_fp8={A_fp8.shape} {A_fp8.dtype}, scale={A_fp8_scale.shape}")
        gemm_a8wfp4(A_fp8, bq_u8, out, A_fp8_scale, bscale_raw)
        print(f"[A8v2] per_token_fp8 call SUCCESS! out={out[0,:3]}")
    except Exception as e:
        print(f"[A8v2] per_token_fp8: {str(e)[:300]}")

    # Attempt 3: Check what quant functions exist
    try:
        import aiter.ops.triton.quant as qmod
        funcs = [x for x in dir(qmod) if 'quant' in x.lower() or 'fp8' in x.lower()]
        print(f"[A8v2] quant module functions: {funcs}")
    except Exception as e:
        print(f"[A8v2] quant module: {e}")

    # Attempt 4: bf16 A directly (maybe kernel auto-quantizes?)
    try:
        gemm_a8wfp4(A, bq_u8, out, A_scale, bscale_raw)
        print(f"[A8v2] bf16 A direct: out={out[0,:3]}")
    except Exception as e:
        print(f"[A8v2] bf16 A: {str(e)[:300]}")

    # Attempt 5: Simple global fp8 quant
    try:
        from aiter import dtypes as aiter_dtypes
        FP8 = aiter_dtypes.fp8
        amax = A.abs().max()
        fp8_max = torch.finfo(FP8).max
        scale = (amax / fp8_max).view(1)
        A_fp8 = (A.float() / scale).clamp(-fp8_max, fp8_max).to(FP8)
        # scale needs to be per-block E8M0? Or per-tensor?
        # Try per-tensor scale repeated
        a_scale_tensor = torch.full((m, k // 32), 127, dtype=torch.uint8, device=A.device)
        gemm_a8wfp4(A_fp8, bq_u8, out, a_scale_tensor, bscale_raw)
        print(f"[A8v2] global fp8: out={out[0,:3]}")
    except Exception as e:
        print(f"[A8v2] global fp8: {str(e)[:300]}")


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    _probe(A, B_q, B_scale_sh, m, n, k)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    bq_u8 = B_q.view(torch.uint8)

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), bq_u8, asc, bscale_raw, dtype=torch.bfloat16)

    gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out)
    return out
