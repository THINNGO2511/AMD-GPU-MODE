#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
CRITICAL TEST: Fused A-quant + GEMM in SINGLE kernel via _mxfp4_quant_op.
Tests whether fused path beats aiter's gemm_a16wfp4.
Prints timing to STDOUT for visibility.
"""
from task import input_t, output_t
import torch
import sys
import time
import triton
import triton.language as tl

try:
    from aiter.ops.triton.quant import _mxfp4_quant_op
    _HAS_QOP = True
except ImportError:
    _HAS_QOP = False

_ref = None
_raw = None
_sh = None
_bq = None
_tested = set()

def _unshuffle_e8m0(s):
    s = s.view(torch.uint8)
    sm, sn = s.shape
    return s.view(sm // 32, sn // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(sm, sn)

def _shuffle(s):
    s = s.clone().view(torch.uint8)
    sm, sn = s.shape
    return s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm // 32, sn * 32)


@triton.jit
def _fused_kernel(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    """Single kernel: load bf16 A, quantize inline, GEMM with FP4 B."""
    SCALE_GROUP_SIZE: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_k_iter = tl.cdiv(K, BLOCK_K)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k_packed = tl.arange(0, BLOCK_K // 2)

    # A bf16 pointers (full K elements, not packed)
    a_bf16_ptrs = a_ptr + offs_am[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak

    # B fp4 pointers (K x N via stride swap)
    b_ptrs = b_ptr + offs_k_packed[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # B scale pointers (shuffled: N//32 x K layout)
    offs_bsn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % (N // 32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)
    b_scale_ptrs = b_scales_ptr + offs_bsn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        # Load A as bf16 and quantize inline
        a_bf16 = tl.load(a_bf16_ptrs).to(tl.float32)
        a_fp4, a_scales = _mxfp4_quant_op(a_bf16, BLOCK_K, BLOCK_M, SCALE_GROUP_SIZE)

        # Load B fp4
        b = tl.load(b_ptrs)

        # Load + unshuffle B scales
        b_scales = tl.load(b_scale_ptrs).reshape(
            BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
        ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)

        # A_scales from _mxfp4_quant_op should be in correct format already
        accumulator += tl.dot_scaled(a_fp4, a_scales, "e2m1", b, b_scales, "e2m1")

        a_bf16_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             c, mask=c_mask)


def _run_fused(A_bf16, B_q, B_sc_sh, m, n, k):
    C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    BM, BN, BK = 32, 64, 256
    grid = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
    _fused_kernel[grid](
        A_bf16, B_q, C, B_sc_sh,
        m, n, k,
        A_bf16.stride(0), A_bf16.stride(1),
        B_q.stride(1), B_q.stride(0),
        C.stride(0), C.stride(1),
        B_sc_sh.stride(0), B_sc_sh.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        mfma_nonkdim=16,
        num_warps=4, num_stages=2,
        matrix_instr_nonkdim=16,
    )
    return C


def _test(A, bq, braw, bsh, m, n, k):
    sk = (m, n, k)
    if sk in _tested or k != 512:
        return
    _tested.add(sk)

    if not _HAS_QOP:
        print(f"FUSED: _mxfp4_quant_op NOT AVAILABLE", flush=True)
        return

    # Need M padded to 32 for scale layout
    pad = (32 - m % 32) % 32
    if pad > 0:
        A_padded = torch.nn.functional.pad(A, (0, 0, 0, pad), value=0.0)
    else:
        A_padded = A
    mp = A_padded.shape[0]

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    NI = 20

    # Test fused kernel accuracy
    try:
        # Warmup
        for _ in range(3):
            _run_fused(A_padded, bq, bsh, mp, n, k)
        torch.cuda.synchronize()

        C_fused = _run_fused(A_padded, bq, bsh, mp, n, k)
        torch.cuda.synchronize()
        C_ref = gemm_a16wfp4(A, bq, braw, dtype=torch.bfloat16)
        err = (C_fused[:m, :n].float() - C_ref.float()).abs().max().item()
        mismatch = ((C_fused[:m, :n].float() - C_ref.float()).abs() > 0.01).sum().item()

        # Timing: fused (SINGLE kernel launch)
        t0 = time.time()
        for _ in range(NI):
            _run_fused(A_padded, bq, bsh, mp, n, k)
        torch.cuda.synchronize()
        dt_fused = (time.time() - t0) / NI * 1e6

        # Timing: aiter (also single fused kernel)
        for _ in range(5):
            gemm_a16wfp4(A, bq, braw, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(NI):
            gemm_a16wfp4(A, bq, braw, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        dt_aiter = (time.time() - t0) / NI * 1e6

        print(f"FUSED M{m}N{n}K{k}: fused={dt_fused:.1f}us aiter={dt_aiter:.1f}us x{dt_aiter/dt_fused:.2f} err={err:.6f} mismatch={mismatch}", flush=True)
    except Exception as e:
        print(f"FUSED M{m}N{n}K{k}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)


_K7 = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
       "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
       "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
       "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024}


def custom_kernel(data: input_t) -> output_t:
    global _ref, _raw, _sh, _bq
    A, B, Bq, Bs, Bss = data
    m, k = A.shape
    n = B.shape[0]
    if _ref is not Bss:
        _ref = Bss
        _raw = _unshuffle_e8m0(Bss)
        _bq = Bq.view(torch.uint8)
        _sh = _shuffle(_raw)
    _test(A, _bq, _raw, _sh, m, n, k)
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq, asc, _raw, dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, _bq, _raw, dtype=torch.bfloat16, config=_K7 if k == 7168 else None)
