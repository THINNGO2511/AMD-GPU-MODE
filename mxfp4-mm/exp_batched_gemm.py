"""
GEMM: Probe batched_gemm_a16wfp4 — could batch multiple shapes in one call.
Signature: batched_gemm_a16wfp4(x, w, w_scales, dtype, y, config, transpose_bm, prequant, y_scale)
Takes same inputs as gemm_a16wfp4 but with batch dimension.
Question: what does "batched" mean here? Batch dimension on x? Or multiple N/K combos?
"""
import torch
import sys
import inspect
from task import input_t, output_t

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_warmed = False
_y_cache = {}
_batched_ok = False

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _probe_batched():
    """Probe batched_gemm_a16wfp4 API."""
    global _batched_ok
    try:
        from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import batched_gemm_a16wfp4
        sig = inspect.signature(batched_gemm_a16wfp4)
        print(f"[BATCH] batched_gemm_a16wfp4 sig: {sig}", file=sys.stderr)

        # Read source to understand batch dimension
        src = inspect.getsource(batched_gemm_a16wfp4)
        print(f"[BATCH] Source ({len(src)} chars):", file=sys.stderr)
        for line in src.split('\n')[:60]:
            print(f"  {line}", file=sys.stderr)

        # Test with simple inputs
        # Maybe x is (batch, M, K) and w is (batch, N, K/2)?
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        M, N, K = 16, 256, 512
        B_dummy = 2
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        B_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
        B_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device='cuda')

        # Try without batch dim first (should work like gemm_a16wfp4)
        try:
            Y = batched_gemm_a16wfp4(A, B_fp4, B_scale, dtype=torch.bfloat16)
            print(f"[BATCH] 2D input works! Y.shape={Y.shape}", file=sys.stderr)
        except Exception as e:
            print(f"[BATCH] 2D input error: {str(e)[:150]}", file=sys.stderr)

        # Try with batch dim
        A_batch = A.unsqueeze(0).expand(B_dummy, -1, -1).contiguous()
        B_fp4_batch = B_fp4.unsqueeze(0).expand(B_dummy, -1, -1).contiguous()
        B_scale_batch = B_scale.unsqueeze(0).expand(B_dummy, -1, -1).contiguous()
        try:
            Y = batched_gemm_a16wfp4(A_batch, B_fp4_batch, B_scale_batch, dtype=torch.bfloat16)
            print(f"[BATCH] 3D input works! Y.shape={Y.shape}", file=sys.stderr)
            _batched_ok = True
        except Exception as e:
            print(f"[BATCH] 3D input error: {str(e)[:150]}", file=sys.stderr)

    except Exception as e:
        print(f"[BATCH] import error: {e}", file=sys.stderr)

    # Also probe fused_gemm_afp4wfp4_a16w16
    try:
        from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_a16w16 import fused_gemm_afp4wfp4_a16w16
        sig = inspect.signature(fused_gemm_afp4wfp4_a16w16)
        print(f"\n[FUSED] fused_gemm_afp4wfp4_a16w16 sig: {sig}", file=sys.stderr)
        src = inspect.getsource(fused_gemm_afp4wfp4_a16w16)
        print(f"[FUSED] Source ({len(src)} chars):", file=sys.stderr)
        for line in src.split('\n')[:40]:
            print(f"  {line}", file=sys.stderr)
    except Exception as e:
        print(f"[FUSED] error: {e}", file=sys.stderr)

    # Also probe gemm_a8wfp4 source in detail
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
        src = inspect.getsource(gemm_a8wfp4)
        print(f"\n[A8WFP4] Source ({len(src)} chars):", file=sys.stderr)
        for line in src.split('\n')[:50]:
            print(f"  {line}", file=sys.stderr)
    except Exception as e:
        print(f"[A8WFP4] error: {e}", file=sys.stderr)


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    _probe_batched()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try:
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
        except:
            pass
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        _prewarm()

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    cfg = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                        y=_y_cache[key], config=cfg)
