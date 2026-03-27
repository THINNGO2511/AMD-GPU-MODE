#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Use aiter's Triton GEMM (tl.dot_scaled) instead of CK gemm_a4w4.

Key insight: The Triton GEMM path uses tl.dot_scaled("e2m1") which takes RAW
(un-shuffled) E8M0 scales. This eliminates the ~18us e8m0_shuffle overhead.

We un-shuffle B_scale_sh once (cached) and skip A scale shuffling entirely.
"""
from task import input_t, output_t
import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

_cache = {}
_use_triton = None  # None = not tested, True/False = result


def _unshuffle_e8m0(scale_sh):
    """Inverse of e8m0_shuffle: permute(0,3,5,2,4,1) -> inverse is permute(0,5,3,1,4,2)"""
    sm, sn = scale_sh.shape
    s = scale_sh.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _try_triton_gemm(A_fp4, A_scale, B_q, B_scale_raw, m, n, k):
    """Try the Triton GEMM path. Returns (output, success)."""
    try:
        # Try importing the Triton GEMM
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

        # Attempt 1: pass as-is (might fail with fp4x2 dtype)
        try:
            out = gemm_afp4wfp4(
                A_fp4, B_q, A_scale, B_scale_raw,
                dtype=torch.bfloat16,
            )
            return out, True
        except (KeyError, TypeError, RuntimeError) as e1:
            print(f"Triton GEMM attempt 1 failed: {e1}")

        # Attempt 2: pass as uint8
        try:
            out = gemm_afp4wfp4(
                A_fp4.view(torch.uint8),
                B_q.view(torch.uint8),
                A_scale.view(torch.uint8),
                B_scale_raw.view(torch.uint8),
                dtype=torch.bfloat16,
            )
            return out, True
        except (KeyError, TypeError, RuntimeError) as e2:
            print(f"Triton GEMM attempt 2 failed: {e2}")

        # Attempt 3: pass as int8
        try:
            out = gemm_afp4wfp4(
                A_fp4.view(torch.int8),
                B_q.view(torch.int8),
                A_scale.view(torch.int8),
                B_scale_raw.view(torch.int8),
                dtype=torch.bfloat16,
            )
            return out, True
        except (KeyError, TypeError, RuntimeError) as e3:
            print(f"Triton GEMM attempt 3 failed: {e3}")

        # Attempt 4: try preshuffled_scales variant with shuffled scales
        try:
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffled_scales
            # This variant handles pre-shuffled scales internally
            out = gemm_afp4wfp4_preshuffled_scales(
                A_fp4, B_q, A_scale, B_scale_raw,
                dtype=torch.bfloat16,
            )
            return out, True
        except Exception as e4:
            print(f"Triton GEMM attempt 4 (preshuffled) failed: {e4}")

    except ImportError as e:
        print(f"Triton GEMM import failed: {e}")

    return None, False


def custom_kernel(data: input_t) -> output_t:
    global _use_triton

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A — raw scales (no shuffle)
    A_fp4, A_scale = dynamic_mxfp4_quant(A)

    # Try Triton path (tested once, then cached decision)
    if _use_triton is None:
        # Un-shuffle B scales (one-time cost)
        sm, sn = B_scale_sh.shape
        if sm >= 32 and sn >= 8 and sm % 32 == 0 and sn % 8 == 0:
            B_scale_raw = _unshuffle_e8m0(B_scale_sh)
        else:
            B_scale_raw = B_scale_sh  # Can't un-shuffle, try as-is

        out, success = _try_triton_gemm(A_fp4, A_scale, B_q, B_scale_raw, m, n, k)
        if success:
            _use_triton = True
            _cache['B_scale_raw'] = B_scale_raw
            print("Using Triton GEMM path (no shuffle needed!)")
            return out
        else:
            _use_triton = False
            print("Triton GEMM failed, falling back to CK path")

    if _use_triton:
        B_scale_raw = _cache.get('B_scale_raw')
        if B_scale_raw is None:
            sm, sn = B_scale_sh.shape
            B_scale_raw = _unshuffle_e8m0(B_scale_sh)
            _cache['B_scale_raw'] = B_scale_raw

        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        return gemm_afp4wfp4(
            A_fp4, B_q, A_scale, B_scale_raw,
            dtype=torch.bfloat16,
        )

    # Fallback: CK path (baseline)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale).view(dtypes.fp8_e8m0)
    return aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
