#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Diagnostic: Run custom tl.dot_scaled kernel on ALL 6 shapes.
Print timing + accuracy vs aiter for each shape.
Custom kernel handles K=512 only; fallback for K=7168/2048/1536.
Output is <30 lines to survive truncation.
"""
from task import input_t, output_t
import torch
import sys
import time
import triton
import triton.language as tl

_bscale_ref = None
_bscale_raw = None
_bscale_sh = None
_bq_u8 = None
_tested = set()
_c_cache = {}

def P(m):
    print(f"T: {m}", file=sys.stderr)

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def _shuffle_scales_cdna4(scales):
    s = scales.clone().view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    s = s.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return s.view(sm // 32, sn * 32)

@triton.jit
def _gemm(
    a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(0)
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(0, num_k_iter):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_sc = tl.load(a_scale_ptrs).reshape(
            BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
        ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
        b_sc = tl.load(b_scale_ptrs).reshape(
            BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
        ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
        acc += tl.dot_scaled(a, a_sc, "e2m1", b, b_sc, "e2m1")
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        a_scale_ptrs += BLOCK_K * stride_ask
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c, mask=c_mask)


def _run_custom(A_fp4, A_sc_sh, B_q, B_sc_sh, m, n, k, bm, bn, bk):
    key = (m, n)
    if key not in _c_cache:
        _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[key]
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _gemm[grid](
        A_fp4, B_q, C, A_sc_sh, B_sc_sh,
        m, n, k,
        A_fp4.stride(0), A_fp4.stride(1),
        B_q.stride(1), B_q.stride(0),
        C.stride(0), C.stride(1),
        A_sc_sh.stride(0), A_sc_sh.stride(1),
        B_sc_sh.stride(0), B_sc_sh.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        mfma_nonkdim=16,
        num_warps=4, num_stages=2,
        matrix_instr_nonkdim=16,
    )
    return C


def _test_shape(A, B_q_u8, B_scale_raw, B_scale_sh, m, n, k):
    """Test custom kernel on one shape, print timing."""
    shape_key = (m, n, k)
    if shape_key in _tested:
        return
    _tested.add(shape_key)

    # Only test K=512 shapes (custom kernel handles these)
    if k != 512:
        return

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    # Pad A_scale to multiple of 32 rows for shuffle
    pad_m = (32 - A_scale.shape[0] % 32) % 32
    if pad_m > 0:
        A_scale_padded = torch.nn.functional.pad(A_scale.view(torch.uint8), (0, 0, 0, pad_m), value=127)
    else:
        A_scale_padded = A_scale
    A_sc_sh = _shuffle_scales_cdna4(A_scale_padded)
    # Also pad A_fp4 to match
    if pad_m > 0:
        A_fp4_padded = torch.nn.functional.pad(A_fp4.view(torch.uint8), (0, 0, 0, pad_m), value=0)
    else:
        A_fp4_padded = A_fp4.view(torch.uint8)

    # Try multiple configs
    configs = [
        (32, 64, 256, "BM32_BN64"),
        (32, 128, 256, "BM32_BN128"),
        (32, 32, 256, "BM32_BN32"),
    ]
    if m <= 16:
        configs = [
            (32, 32, 256, "BM32_BN32"),
            (32, 64, 256, "BM32_BN64"),
        ]

    # Reference
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    C_ref = gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16)

    N_ITER = 20
    # aiter timing
    for _ in range(5):
        gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    dt_aiter = (time.time() - t0) / N_ITER * 1e6

    # Custom configs
    best_name = "aiter"
    best_dt = dt_aiter
    for bm, bn, bk, name in configs:
        try:
            # Warmup
            m_padded = A_fp4_padded.shape[0]
            for _ in range(3):
                _run_custom(A_fp4_padded, A_sc_sh, B_q_u8, B_scale_sh, m_padded, n, k, bm, bn, bk)
            torch.cuda.synchronize()

            # Check accuracy (use padded M for kernel, slice output to actual M)
            m_padded = A_fp4_padded.shape[0]
            C_custom_full = _run_custom(A_fp4_padded, A_sc_sh, B_q_u8, B_scale_sh, m_padded, n, k, bm, bn, bk)
            C_custom = C_custom_full[:m, :n]
            torch.cuda.synchronize()
            diff = (C_custom.float() - C_ref.float()).abs()
            mismatch = (diff > 0.01).sum().item()

            # Time
            t0 = time.time()
            for _ in range(N_ITER):
                _run_custom(A_fp4_padded, A_sc_sh, B_q_u8, B_scale_sh, m_padded, n, k, bm, bn, bk)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / N_ITER * 1e6

            if dt < best_dt:
                best_dt = dt
                best_name = name
            P(f"M{m}N{n}K{k} {name}: {dt:.1f}us err={mismatch}")
        except Exception as e:
            P(f"M{m}N{n}K{k} {name}: FAIL {type(e).__name__}")

    P(f"M{m}N{n}K{k} aiter:{dt_aiter:.1f}us best:{best_name}={best_dt:.1f}us x{dt_aiter/best_dt:.2f}")


_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bscale_sh, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        _bscale_sh = _shuffle_scales_cdna4(_bscale_raw)

    # Run diagnostic (once per shape)
    _test_shape(A, _bq_u8, _bscale_raw, _bscale_sh, m, n, k)

    # Always return correct via proven fallback
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg)
