import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
import torch
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes

FP8 = aiter_dtypes.fp8
FP8_MAX = torch.finfo(FP8).max

_y_cache = {}
_probed = False
_bscale_ref = None
_bscale_raw = None
_bscale_trimmed = None
_bq_u8 = None

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def _probe(A, m, n, k):
    global _probed
    if _probed:
        return
    _probed = True

    from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4

    # Manual per-token fp8 quant: amax per row, scale, clamp, cast
    amax = A.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # (M, 1)
    scale = amax / FP8_MAX  # (M, 1)
    A_fp8 = (A.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)  # (M, K) fp8
    print(f"[A8v4] A_fp8: {A_fp8.shape} {A_fp8.dtype}, scale: {scale.shape}")

    out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)

    # Call: gemm_a8wfp4(x, w, y, x_scales, w_scales)
    # x_scales should be (M, 1), w_scales should be (N, K/32)
    try:
        gemm_a8wfp4(A_fp8, _bq_u8, out, scale.to(torch.float32), _bscale_trimmed)
        print(f"[A8v4] float32 scale SUCCESS! out={out[0,:5]}")

        # Compare with reference
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        ref = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=ref)
        diff = (out.float() - ref.float()).abs()
        mismatch = ((diff > 0.01 + 0.01 * ref.float().abs()).sum().item()) / out.numel()
        print(f"[A8v4] vs ref: max={diff.max():.4f} mean={diff.mean():.4f} mismatch={mismatch:.6f}")

        # Time the full path
        import time
        torch.cuda.synchronize()
        for t in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            am = A.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            sc = am / FP8_MAX
            afp8 = (A.float() / sc).clamp(-FP8_MAX, FP8_MAX).to(FP8)
            gemm_a8wfp4(afp8, _bq_u8, out, sc.to(torch.float32), _bscale_trimmed)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"[A8v4] quant+gemm({m},{n},{k}): {(t1-t0)*1e6:.1f}us")

    except Exception as e:
        print(f"[A8v4] float32 scale: {str(e)[:500]}")

    # Try scale as fp32 scalar per token
    try:
        scale_f32 = scale.squeeze(1).to(torch.float32)  # (M,)
        gemm_a8wfp4(A_fp8, _bq_u8, out, scale_f32.unsqueeze(1), _bscale_trimmed)
        print(f"[A8v4] squeezed scale SUCCESS!")
    except Exception as e:
        print(f"[A8v4] squeezed: {str(e)[:200]}")


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bscale_trimmed, _bq_u8
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        _bscale_trimmed = _bscale_raw[:n, :]

    _probe(A, m, n, k)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq_u8, asc, _bscale_raw, dtype=torch.bfloat16)

    gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out)
    return out
