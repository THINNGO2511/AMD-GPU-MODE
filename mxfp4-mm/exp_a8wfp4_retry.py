"""
GEMM: Retry gemm_a8wfp4 — quantize A to fp8 (not fp4), then GEMM with fp4 B.
Previous blocker was eval framework scale shape assertion.
Let's probe the actual error and see if we can work around it.
gemm_a8wfp4 could be faster than a16wfp4 (half the A bandwidth).
"""
import torch
import sys
from task import input_t, output_t

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_warmed = False
_y_cache = {}
_a8wfp4_ok = None

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


def _probe_a8wfp4():
    """Probe gemm_a8wfp4 API to understand scale requirements."""
    global _a8wfp4_ok
    try:
        import inspect
        from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4

        sig = inspect.signature(gemm_a8wfp4)
        print(f"[A8WFP4] Signature: {sig}", file=sys.stderr)

        try:
            src = inspect.getsource(gemm_a8wfp4)
            # Print first 1500 chars and look for scale validation
            print(f"[A8WFP4] Source ({len(src)} chars):", file=sys.stderr)
            for line in src.split('\n')[:60]:
                print(f"  {line}", file=sys.stderr)
        except:
            print("[A8WFP4] Cannot get source", file=sys.stderr)

        # Test with small matrices
        M, N, K = 16, 256, 512
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

        # Quantize A to fp8
        from aiter import dtypes as aiter_dtypes
        FP8_DTYPE = aiter_dtypes.fp8
        amax = A.abs().amax()
        fp8_max = torch.finfo(FP8_DTYPE).max
        scale = amax / fp8_max
        A_fp8 = (A / scale).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
        A_scale_row = scale.expand(M, 1).to(torch.float32)  # per-row scale

        # Create dummy B
        B_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
        B_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device='cuda')

        Y = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

        # Try different scale formats
        for desc, xs in [
            ("per-row (M,1) f32", A_scale_row),
            ("scalar f32", scale.unsqueeze(0).to(torch.float32)),
            ("per-row (M,) f32", A_scale_row.squeeze(1)),
        ]:
            try:
                gemm_a8wfp4(A_fp8, B_fp4, Y, xs, B_scale, dtype=torch.bfloat16)
                print(f"[A8WFP4] {desc}: WORKED! Y[:2,:4]={Y[:2,:4]}", file=sys.stderr)
                _a8wfp4_ok = desc
                return True
            except Exception as e:
                print(f"[A8WFP4] {desc}: {str(e)[:150]}", file=sys.stderr)

    except ImportError:
        print("[A8WFP4] gemm_a8wfp4 not available", file=sys.stderr)
    except Exception as e:
        print(f"[A8WFP4] Probe error: {e}", file=sys.stderr)

    _a8wfp4_ok = None
    return False


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True

    ok = _probe_a8wfp4()
    print(f"[A8WFP4] Available: {ok}, format: {_a8wfp4_ok}", file=sys.stderr)

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

    # Standard Triton path (proven fastest)
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
