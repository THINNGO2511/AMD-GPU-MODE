#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Debug: Test if _mxfp4_quant_op can be called from inside a tl.dot_scaled kernel.
Print exact error if it fails.
"""
from task import input_t, output_t
import torch
import sys
import triton
import triton.language as tl

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_probed = False


def P(msg):
    print(f"D: {msg}", file=sys.stderr)


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _debug():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. Try importing _mxfp4_quant_op
    P("=== IMPORT TEST ===")
    try:
        from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
        P(f"IMPORT SUCCESS: {_mxfp4_quant_op}")
        P(f"Type: {type(_mxfp4_quant_op)}")
        P(f"Is JIT: {hasattr(_mxfp4_quant_op, 'fn')}")
        if hasattr(_mxfp4_quant_op, 'fn'):
            P(f"JIT fn: {_mxfp4_quant_op.fn}")

        # Check signature
        import inspect
        try:
            sig = inspect.signature(_mxfp4_quant_op)
            P(f"Signature: {sig}")
        except Exception as e:
            P(f"Signature error: {e}")

        # Check if it's a Triton constexpr function
        P(f"Dir: {[a for a in dir(_mxfp4_quant_op) if not a.startswith('__')][:20]}")

    except ImportError as e:
        P(f"IMPORT FAILED: {e}")
    except Exception as e:
        P(f"IMPORT ERROR: {type(e).__name__}: {e}")

    # 2. Also try the __init__.py path
    P("\n=== ALT IMPORT ===")
    try:
        from aiter.ops.triton.quant import _mxfp4_quant_op as qop2
        P(f"ALT IMPORT SUCCESS: {qop2}")
        P(f"Same object: {qop2 is _mxfp4_quant_op}" if '_mxfp4_quant_op' in dir() else "original failed")
    except Exception as e:
        P(f"ALT IMPORT: {e}")

    # 3. Test tl.dot_scaled in a minimal kernel
    P("\n=== tl.dot_scaled TEST ===")
    try:
        @triton.jit
        def _test_dot_scaled(
            a_ptr, b_ptr, c_ptr,
            a_scale_ptr, b_scale_ptr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ):
            offs_m = tl.arange(0, BLOCK_M)
            offs_n = tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K // 2)
            offs_ks = tl.arange(0, BLOCK_K // 32)

            a = tl.load(a_ptr + offs_m[:, None] * (BLOCK_K // 2) + offs_k[None, :])
            b = tl.load(b_ptr + offs_n[:, None] * (BLOCK_K // 2) + offs_k[None, :])
            a_s = tl.load(a_scale_ptr + offs_m[:, None] * (BLOCK_K // 32) + offs_ks[None, :])
            b_s = tl.load(b_scale_ptr + offs_n[:, None] * (BLOCK_K // 32) + offs_ks[None, :])

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc = tl.dot_scaled(a, a_s, "e2m1", b, b_s, "e2m1", acc)

            tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc.to(tl.bfloat16))

        M, N, K = 32, 32, 128
        A_fp4 = torch.zeros((M, K // 2), dtype=torch.uint8, device='cuda')
        B_fp4 = torch.zeros((N, K // 2), dtype=torch.uint8, device='cuda')
        A_scale = torch.full((M, K // 32), 127, dtype=torch.uint8, device='cuda')
        B_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device='cuda')
        C = torch.zeros((M, N), dtype=torch.bfloat16, device='cuda')

        _test_dot_scaled[(1,)](A_fp4, B_fp4, C, A_scale, B_scale,
                               BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,
                               num_warps=4, num_stages=1)
        torch.cuda.synchronize()
        P(f"tl.dot_scaled SUCCESS! C[0,0]={C[0,0].item()}")
    except Exception as e:
        P(f"tl.dot_scaled FAILED: {type(e).__name__}: {str(e)[:300]}")

    # 4. Read first 20 lines of quant.py for _mxfp4_quant_op signature
    P("\n=== QUANT OP DEFINITION ===")
    try:
        with open("/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant/quant.py") as f:
            lines = f.readlines()
        # Find the function def
        for i, line in enumerate(lines):
            if '_mxfp4_quant_op' in line and ('def ' in line or '@' in line):
                start = max(0, i - 3)
                end = min(len(lines), i + 20)
                for j in range(start, end):
                    P(f"  {j+1:4d}| {lines[j].rstrip()}")
                P("  ---")
    except Exception as e:
        P(f"Read error: {e}")


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    _debug()

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16)
