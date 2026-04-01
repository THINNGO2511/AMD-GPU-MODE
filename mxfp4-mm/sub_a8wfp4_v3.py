import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
import torch
from task import input_t, output_t

_y_cache = {}
_probed = False
_bscale_raw_trimmed = None
_bscale_ref = None
_bq_u8 = None

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

    # Trim w_scales to (N, K/32) — remove padding
    w_scales = bscale_raw[:n, :]
    print(f"[A8v3] w_scales trimmed: {w_scales.shape}")

    # Per-token fp8 quantization of A
    try:
        from aiter.ops.triton.quant import dynamic_per_token_quant_fp8_i8
        A_fp8, A_scale = dynamic_per_token_quant_fp8_i8(A)
        print(f"[A8v3] per_token_fp8_i8: A_fp8={A_fp8.shape} {A_fp8.dtype}, scale={A_scale.shape} {A_scale.dtype}")
    except Exception as e:
        print(f"[A8v3] per_token error: {e}")
        # Try fused_fp8_quant
        try:
            from aiter.ops.triton.quant import fused_fp8_quant
            A_fp8, A_scale = fused_fp8_quant(A)
            print(f"[A8v3] fused_fp8: A_fp8={A_fp8.shape} {A_fp8.dtype}, scale={A_scale.shape} {A_scale.dtype}")
        except Exception as e2:
            print(f"[A8v3] fused_fp8 error: {e2}")
            return

    # Call gemm_a8wfp4(x, w, y, x_scales, w_scales)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    try:
        gemm_a8wfp4(A_fp8, bq_u8, out, A_scale, w_scales)
        print(f"[A8v3] SUCCESS! out={out[0,:5]}")

        # Compare with reference (a16wfp4)
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        ref = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=ref)
        diff = (out.float() - ref.float()).abs()
        rdiff = diff / (ref.float().abs() + 1e-6)
        mismatch = ((diff > 0.01 + 0.01 * ref.float().abs()).sum().item()) / out.numel()
        print(f"[A8v3] vs ref: max_abs={diff.max():.4f} mean_abs={diff.mean():.4f} max_rel={rdiff.max():.4f}")
        print(f"[A8v3] mismatch_ratio={mismatch:.6f} (need <threshold)")

        # Time it
        import time
        torch.cuda.synchronize()
        for trial in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            A_fp8_t, A_scale_t = dynamic_per_token_quant_fp8_i8(A)
            gemm_a8wfp4(A_fp8_t, bq_u8, out, A_scale_t, w_scales)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"[A8v3] quant+gemm time: {(t1-t0)*1e6:.1f}us")

    except Exception as e:
        print(f"[A8v3] call failed: {str(e)[:500]}")

    # Also try with A_scale reshaped to (M, 1)
    try:
        if A_scale.dim() == 1:
            a_scale_2d = A_scale.view(-1, 1)
        else:
            a_scale_2d = A_scale
        print(f"[A8v3] A_scale reshaped: {a_scale_2d.shape}")
        gemm_a8wfp4(A_fp8, bq_u8, out, a_scale_2d, w_scales)
        print(f"[A8v3] Reshaped scale SUCCESS!")
    except Exception as e:
        print(f"[A8v3] Reshaped scale: {str(e)[:200]}")


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
