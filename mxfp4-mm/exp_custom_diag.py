#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Diagnostic: Test custom tl.dot_scaled kernel on ONE shape (M=32, N=4096, K=512).
Prints whether custom or fallback was used + timing.
"""
from task import input_t, output_t
import torch
import sys
import time
import triton
import triton.language as tl

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_done = False

def P(m):
    print(f"CK: {m}", file=sys.stderr)


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _shuffle_scales_cdna4(scales, mfma_nonkdim=16):
    s = scales.clone().view(torch.uint8)
    sm, sn = s.shape
    if mfma_nonkdim == 16:
        s = s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        s = s.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return s.view(sm // 32, sn * 32)


@triton.jit
def _test_gemm(
    a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_k_iter = tl.cdiv(K, BLOCK_K)

    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)) % (M // 32)
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % (N // 32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    a_scale_ptrs = a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    b_scale_ptrs = b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        a_scales = tl.load(a_scale_ptrs).reshape(
            BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
        ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)

        b_scales = tl.load(b_scale_ptrs).reshape(
            BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
        ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        a_scale_ptrs += BLOCK_K * stride_ask
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             c, mask=c_mask)


def _try_custom(A_fp4, A_scale, B_q, B_scale_raw, m, n, k):
    """Try custom kernel on one shape. Returns result or None."""
    BM, BN, BK = 32, 128, 256

    # Shuffle scales for CDNA4
    A_sc_sh = _shuffle_scales_cdna4(A_scale)
    B_sc_sh = _shuffle_scales_cdna4(B_scale_raw)

    C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    grid = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
    _test_gemm[grid](
        A_fp4, B_q, C, A_sc_sh, B_sc_sh,
        m, n, k,
        A_fp4.stride(0), A_fp4.stride(1),
        B_q.stride(1), B_q.stride(0),
        C.stride(0), C.stride(1),
        A_sc_sh.stride(0), A_sc_sh.stride(1),
        B_sc_sh.stride(0), B_sc_sh.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        mfma_nonkdim=16,
        num_warps=4, num_stages=2,
        matrix_instr_nonkdim=16,
    )
    return C


def _diag(A, B_q_u8, B_scale_raw, m, n, k):
    """Run diagnostic on one call."""
    global _done
    if _done:
        return
    if not (m == 32 and k == 512):
        return
    _done = True

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4, A_scale = dynamic_mxfp4_quant(A)

    # Try custom kernel
    try:
        t0 = time.time()
        C_custom = _try_custom(A_fp4.view(torch.uint8), A_scale, B_q_u8, B_scale_raw, m, n, k)
        torch.cuda.synchronize()
        t_custom = (time.time() - t0) * 1e6
        P(f"CUSTOM: compiled+ran in {t_custom:.0f}us")

        # Compare with reference (aiter)
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        C_ref = gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16)

        diff = (C_custom.float() - C_ref.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        mismatch = (diff > 0.01).sum().item()
        total = C_custom.numel()
        P(f"ACCURACY: max={max_err:.6f} mean={mean_err:.6f} mismatch={mismatch}/{total}")

        # Timing (warmup then measure)
        for _ in range(3):
            _try_custom(A_fp4.view(torch.uint8), A_scale, B_q_u8, B_scale_raw, m, n, k)
        torch.cuda.synchronize()

        N_ITER = 20
        t0 = time.time()
        for _ in range(N_ITER):
            _try_custom(A_fp4.view(torch.uint8), A_scale, B_q_u8, B_scale_raw, m, n, k)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / N_ITER * 1e6
        P(f"TIMING: custom={dt:.1f}us (M={m},N={n},K={k})")

        # Compare with aiter timing
        for _ in range(3):
            gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N_ITER):
            gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        dt_ref = (time.time() - t0) / N_ITER * 1e6
        P(f"TIMING: aiter={dt_ref:.1f}us")
        P(f"SPEEDUP: {dt_ref/dt:.2f}x")

    except Exception as e:
        P(f"CUSTOM FAILED: {type(e).__name__}: {str(e)[:200]}")


_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    # Run diagnostic on first K=512 shape
    _diag(A, _bq_u8, _bscale_raw, m, n, k)

    # Always return correct result via proven fallback
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg)
