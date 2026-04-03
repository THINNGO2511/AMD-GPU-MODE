#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Fix gemm_a4w4 data format for correct accuracy.

ROUND 9 BREAKTHROUGH: gemm_a4w4 RUNS with fp4x2 dtype tensors!
But accuracy is wrong (16848/16896 elements, values like 20 vs 163).

Root cause candidates:
1. A_scale is uint8, kernel expects float8_e8m0fnu
2. A_fp4 data needs shuffling (like B_shuffle vs B_q)
3. A_scale needs e8m0_shuffle (like B_scale_sh)

This tries 6 combinations systematically and reports accuracy for each.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

from task import input_t, output_t
import torch
import sys

_call_count = 0
_best_attempt = None  # Will be set after first call determines best combo
_y_cache = {}

def custom_kernel(data: input_t) -> output_t:
    global _call_count, _best_attempt
    _call_count += 1
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter import dtypes as aiter_dtypes
    from aiter import gemm_a4w4

    fp4x2 = aiter_dtypes.fp4x2
    e8m0 = aiter_dtypes.fp8_e8m0

    # Quantize A
    A_fp4_u8, A_scale_u8 = dynamic_mxfp4_quant(A)
    A_fp4 = A_fp4_u8.view(fp4x2)
    A_scale_e8m0 = A_scale_u8.view(e8m0)  # Fix 1: proper dtype

    # If we already know the best attempt, use it
    if _best_attempt is not None:
        return _run_attempt(_best_attempt, A, A_fp4, A_fp4_u8, A_scale_u8,
                           A_scale_e8m0, B_q, B_shuffle, B_scale_sh,
                           m, n, k, fp4x2, e8m0)

    # First call: try all combos, compute reference, find best
    # Reference result
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    su = B_scale_sh.view(torch.uint8)
    sm, sn = su.shape
    d0, d1 = sm // 32, sn // 8
    total = sm * sn
    idx = torch.arange(total, dtype=torch.int64, device=su.device)
    idx = idx.view(d0, d1, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(-1)
    bscale_raw = torch.take(su.reshape(-1), idx).view(sm, sn)
    bq_u8 = B_q.view(torch.uint8)
    ref = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=ref, config=None)

    # Try shuffling A
    try:
        from aiter.ops.shuffle import shuffle_weight
        A_fp4_shuffled = shuffle_weight(A_fp4, (16, 16))
        has_shuffle = True
    except Exception as e:
        print(f"[GEMM] shuffle_weight(A) failed: {e}", flush=True)
        A_fp4_shuffled = A_fp4
        has_shuffle = False

    # Try shuffling A_scale
    try:
        from aiter.utility.fp4_utils import e8m0_shuffle
        A_scale_shuffled = e8m0_shuffle(A_scale_u8).view(e8m0)
        has_scale_shuffle = True
    except Exception as e:
        print(f"[GEMM] e8m0_shuffle(A_scale) failed: {e}", flush=True)
        A_scale_shuffled = A_scale_e8m0
        has_scale_shuffle = False

    attempts = [
        # (name, A_data, A_scale, B_data, B_scale, bpreshuffle)
        ("raw_A+e8m0_scale+B_shuffle+bpre1",
         A_fp4, A_scale_e8m0, B_shuffle, B_scale_sh, 1),
        ("shuffled_A+e8m0_scale+B_shuffle+bpre1",
         A_fp4_shuffled if has_shuffle else A_fp4, A_scale_e8m0, B_shuffle, B_scale_sh, 1),
        ("shuffled_A+shuffled_Ascale+B_shuffle+bpre1",
         A_fp4_shuffled if has_shuffle else A_fp4, A_scale_shuffled if has_scale_shuffle else A_scale_e8m0, B_shuffle, B_scale_sh, 1),
        ("raw_A+shuffled_Ascale+B_shuffle+bpre1",
         A_fp4, A_scale_shuffled if has_scale_shuffle else A_scale_e8m0, B_shuffle, B_scale_sh, 1),
        # Also try bpreshuffle=0 with raw B
        ("raw_A+e8m0_scale+B_raw+bpre0",
         A_fp4, A_scale_e8m0, B_q, B_scale_sh, 0),
        ("shuffled_A+shuffled_Ascale+B_raw+bpre0",
         A_fp4_shuffled if has_shuffle else A_fp4, A_scale_shuffled if has_scale_shuffle else A_scale_e8m0, B_q, B_scale_sh, 0),
    ]

    best_err = float('inf')
    best_idx = -1
    for i, (name, a_data, a_scale, b_data, b_scale, bpre) in enumerate(attempts):
        try:
            result = gemm_a4w4(a_data, b_data, a_scale, b_scale,
                              None, torch.bfloat16, 1.0, 0.0, bpre)
            # Compare to reference
            diff = (result.float() - ref.float()).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            n_wrong = (diff > 0.1 * ref.float().abs().clamp(min=1e-6) + 0.1).sum().item()
            total_els = result.numel()
            print(f"[GEMM] {name}: max_err={max_err:.3f} mean_err={mean_err:.3f} "
                  f"wrong={n_wrong}/{total_els}", flush=True)
            if n_wrong < best_err:
                best_err = n_wrong
                best_idx = i
        except Exception as e:
            print(f"[GEMM] {name}: FAILED {e}", flush=True)

    if best_idx >= 0 and best_err < attempts[best_idx][0].count('x'):  # If any attempt has decent accuracy
        _best_attempt = best_idx
        print(f"[GEMM] BEST: attempt {best_idx} ({attempts[best_idx][0]}) with {best_err} wrong", flush=True)
    else:
        print(f"[GEMM] ALL attempts have bad accuracy. Best: {best_idx} with {best_err} wrong", flush=True)
        _best_attempt = -1  # Signal to use fallback

    sys.stdout.flush()

    # Return reference result for this call
    return ref


def _run_attempt(attempt_idx, A, A_fp4, A_fp4_u8, A_scale_u8,
                A_scale_e8m0, B_q, B_shuffle, B_scale_sh,
                m, n, k, fp4x2, e8m0):
    """Run the winning attempt or fallback."""
    if attempt_idx < 0:
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
        key = (m, n)
        if key not in _y_cache:
            _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        out = _y_cache[key]
        if k == 1536:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            af, asc = dynamic_mxfp4_quant(A)
            return gemm_afp4wfp4(af.view(torch.uint8), bq_u8, asc, bscale_raw, dtype=torch.bfloat16)
        gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out, config=None)
        return out

    from aiter import gemm_a4w4
    try:
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle
        A_fp4_shuffled = shuffle_weight(A_fp4, (16, 16))
        A_scale_shuffled = e8m0_shuffle(A_scale_u8).view(e8m0)
    except Exception:
        A_fp4_shuffled = A_fp4
        A_scale_shuffled = A_scale_e8m0

    # Dispatch based on attempt_idx
    combos = [
        (A_fp4, A_scale_e8m0, B_shuffle, B_scale_sh, 1),
        (A_fp4_shuffled, A_scale_e8m0, B_shuffle, B_scale_sh, 1),
        (A_fp4_shuffled, A_scale_shuffled, B_shuffle, B_scale_sh, 1),
        (A_fp4, A_scale_shuffled, B_shuffle, B_scale_sh, 1),
        (A_fp4, A_scale_e8m0, B_q, B_scale_sh, 0),
        (A_fp4_shuffled, A_scale_shuffled, B_q, B_scale_sh, 0),
    ]
    a_data, a_scale, b_data, b_scale, bpre = combos[attempt_idx]
    return gemm_a4w4(a_data, b_data, a_scale, b_scale,
                    None, torch.bfloat16, 1.0, 0.0, bpre)
