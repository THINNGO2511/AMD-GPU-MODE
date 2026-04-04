import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
import torch
import inspect
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

def _probe(A, B_q, B_shuffle, B_scale_sh, m, n, k):
    global _probed
    if _probed:
        return
    _probed = True

    # 1. Check gemm_a8wfp4
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
        src = inspect.getsource(gemm_a8wfp4)
        print(f"[A8] gemm_a8wfp4 FOUND ({len(src)} chars)")
        print(src[:1500])
        sig = inspect.signature(gemm_a8wfp4)
        print(f"[A8] Signature: {sig}")
    except Exception as e:
        print(f"[A8] gemm_a8wfp4 error: {e}")

    # 2. Check what other functions exist
    try:
        import aiter.ops.triton.gemm.basic.gemm_a8wfp4 as mod
        funcs = [x for x in dir(mod) if not x.startswith('_')]
        print(f"[A8] Module exports: {funcs}")
    except Exception as e:
        print(f"[A8] Module error: {e}")

    # 3. Try calling it — fp8 A + fp4 B
    try:
        bscale_raw = _unshuffle_e8m0(B_scale_sh)
        bq_u8 = B_q.view(torch.uint8)

        # Quantize A to fp8
        from aiter import dtypes as aiter_dtypes
        FP8_DTYPE = aiter_dtypes.fp8
        amax = A.abs().max()
        FP8_MAX = torch.finfo(FP8_DTYPE).max
        scale = amax / FP8_MAX
        A_fp8 = (A.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

        out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        result = gemm_a8wfp4(A_fp8, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out,
                              a_scale=scale)
        print(f"[A8] Call SUCCESS! out sample: {out[0,:5]}")

        # Compare with reference (a16wfp4)
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        ref = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=ref)
        diff = (out.float() - ref.float()).abs()
        print(f"[A8] vs a16wfp4: max_diff={diff.max():.4f} mean_diff={diff.mean():.4f}")
        mismatch = ((diff > 0.01 + 0.01 * ref.float().abs()).sum().item()) / out.numel()
        print(f"[A8] mismatch_ratio={mismatch:.4f}")
    except Exception as e:
        print(f"[A8] Call failed: {str(e)[:500]}")

    # 4. Try without a_scale
    try:
        result = gemm_a8wfp4(A_fp8, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out)
        print(f"[A8] No a_scale SUCCESS!")
    except Exception as e:
        print(f"[A8] No a_scale: {str(e)[:200]}")

    # 5. Try with bf16 A (maybe it auto-quantizes like a16wfp4?)
    try:
        result = gemm_a8wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out)
        print(f"[A8] bf16 A SUCCESS!")
    except Exception as e:
        print(f"[A8] bf16 A: {str(e)[:200]}")

    # 6. Also probe gemm_afp4wfp4_pre_quant_atomic
    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic import gemm_afp4wfp4_pre_quant_atomic
        sig = inspect.signature(gemm_afp4wfp4_pre_quant_atomic)
        print(f"[A8] pre_quant_atomic sig: {sig}")
    except Exception as e:
        print(f"[A8] pre_quant_atomic: {e}")


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    _probe(A, B_q, B_shuffle, B_scale_sh, m, n, k)

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
