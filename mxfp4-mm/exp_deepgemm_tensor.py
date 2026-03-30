"""
GEMM: Probe deepgemm with group_layout as Tensor + test fused/batched GEMM APIs.
Key findings from last probe:
- deepgemm(XQ, WQ, Y, group_layout: Tensor, x_scale=None, w_scale=None)
- group_layout must be a Tensor, NOT an int
- fused_gemm_afp4wfp4_a16w16.py exists (fused quant+GEMM?)
- batched_gemm_a16wfp4.py exists (multi-shape batching?)
"""
import torch
import sys
from task import input_t, output_t

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_warmed = False
_y_cache = {}

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


def _probe_apis():
    """Probe deepgemm with tensor group_layout + fused/batched GEMM."""
    import inspect

    # 1. Probe deepgemm_ck source
    try:
        from aiter.ops.deepgemm import deepgemm_ck, deepgemm
        try:
            src = inspect.getsource(deepgemm_ck)
            print(f"[DG] deepgemm_ck source ({len(src)} chars):", file=sys.stderr)
            for line in src.split('\n')[:40]:
                print(f"  {line}", file=sys.stderr)
        except:
            print("[DG] deepgemm_ck is C++ (no Python source)", file=sys.stderr)
            # It's a C++ function, probe via torch.ops
            try:
                op = torch.ops.aiter.deepgemm_ck
                print(f"[DG] torch.ops.aiter.deepgemm_ck = {op}", file=sys.stderr)
                schema = op._schemas
                print(f"[DG] schemas: {schema}", file=sys.stderr)
            except Exception as e:
                print(f"[DG] torch.ops probe: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[DG] import error: {e}", file=sys.stderr)

    # 2. Test deepgemm with tensor group_layout
    try:
        import aiter
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter import dtypes

        M, N, K = 16, 256, 512
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        B_fp4, B_scale = dynamic_mxfp4_quant(B)
        Y = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

        # Try various group_layout tensors
        for desc, gl in [
            ("zeros(1) int32", torch.zeros(1, dtype=torch.int32, device='cuda')),
            ("zeros(1) int64", torch.zeros(1, dtype=torch.int64, device='cuda')),
            ("empty(0) int32", torch.empty(0, dtype=torch.int32, device='cuda')),
            ("ones(M) int32", torch.ones(M, dtype=torch.int32, device='cuda')),
            ("arange(M) int32", torch.arange(M, dtype=torch.int32, device='cuda')),
            ("zeros(M,N//K) int32", torch.zeros(M, N//K, dtype=torch.int32, device='cuda')),
        ]:
            for xq, wq, xs, ws, fdesc in [
                (A_fp4.view(dtypes.fp4x2), B_fp4.view(dtypes.fp4x2), A_scale, B_scale, "fp4+scale"),
                (A_fp4.view(torch.uint8), B_fp4.view(torch.uint8), A_scale, B_scale, "u8+scale"),
                (A, B, None, None, "bf16+none"),
            ]:
                try:
                    aiter.deepgemm(xq, wq, Y, gl, xs, ws)
                    print(f"[DG] WORKED: {fdesc} gl={desc} Y[0,:4]={Y[0,:4]}", file=sys.stderr)
                    return
                except Exception as e:
                    err = str(e)[:100]
                    if 'group_layou' not in err:
                        print(f"[DG] {fdesc} gl={desc}: {err}", file=sys.stderr)
    except Exception as e:
        print(f"[DG] deepgemm test error: {e}", file=sys.stderr)

    # 3. Probe fused GEMM APIs
    try:
        from aiter.ops.triton.gemm.fused import fused_gemm_afp4wfp4_a16w16
        print(f"[FUSED] fused_gemm_afp4wfp4_a16w16 found", file=sys.stderr)
        try:
            src = inspect.getsource(fused_gemm_afp4wfp4_a16w16)
            # Show the function signatures
            for line in src.split('\n'):
                if 'def ' in line:
                    print(f"[FUSED] {line.strip()}", file=sys.stderr)
            print(f"[FUSED] Full source ({len(src)} chars):", file=sys.stderr)
            for line in src.split('\n')[:30]:
                print(f"  {line}", file=sys.stderr)
        except:
            pass
    except ImportError:
        try:
            import aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_a16w16 as fmod
            fns = [a for a in dir(fmod) if not a.startswith('_')]
            print(f"[FUSED] Module attrs: {fns}", file=sys.stderr)
        except Exception as e:
            print(f"[FUSED] import error: {e}", file=sys.stderr)

    # 4. Probe batched GEMM
    try:
        import aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 as bmod
        fns = [a for a in dir(bmod) if not a.startswith('_')]
        print(f"[BATCH] batched_gemm_a16wfp4 attrs: {fns}", file=sys.stderr)
        for fn_name in fns:
            fn = getattr(bmod, fn_name)
            if callable(fn):
                try:
                    sig = inspect.signature(fn)
                    print(f"[BATCH] {fn_name}{sig}", file=sys.stderr)
                except:
                    pass
        try:
            src = inspect.getsource(bmod)
            print(f"[BATCH] Source ({len(src)} chars):", file=sys.stderr)
            for line in src.split('\n')[:40]:
                print(f"  {line}", file=sys.stderr)
        except:
            pass
    except Exception as e:
        print(f"[BATCH] import error: {e}", file=sys.stderr)

    # 5. Probe gemm_a8wfp4
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
        sig = inspect.signature(gemm_a8wfp4)
        print(f"[A8WFP4] sig: {sig}", file=sys.stderr)
        try:
            src = inspect.getsource(gemm_a8wfp4)
            # Show first 40 lines of the actual function
            in_func = False
            count = 0
            for line in src.split('\n'):
                if 'def gemm_a8wfp4' in line:
                    in_func = True
                if in_func:
                    print(f"  {line}", file=sys.stderr)
                    count += 1
                    if count > 40:
                        break
        except:
            pass
    except Exception as e:
        print(f"[A8WFP4] error: {e}", file=sys.stderr)

    # 6. Probe afp4wfp4_pre_quant_atomic
    try:
        import aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic as pqmod
        fns = [a for a in dir(pqmod) if not a.startswith('_') and callable(getattr(pqmod, a))]
        print(f"[PREQUANT_ATOMIC] attrs: {fns}", file=sys.stderr)
        for fn_name in fns[:3]:
            fn = getattr(pqmod, fn_name)
            try:
                sig = inspect.signature(fn)
                print(f"[PREQUANT_ATOMIC] {fn_name}{sig}", file=sys.stderr)
            except:
                pass
    except Exception as e:
        print(f"[PREQUANT_ATOMIC] error: {e}", file=sys.stderr)


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True

    _probe_apis()

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
