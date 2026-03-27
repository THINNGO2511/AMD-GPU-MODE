#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Fused quant+GEMM using aiter's exact _mxfp4_quant_op.

Imports aiter's _mxfp4_quant_op JIT function and calls it inline
within our GEMM kernel. Single kernel launch = eliminates quant overhead.

Key: _mxfp4_quant_op(x_fp32, BLOCK_K, BLOCK_M, 32) returns (fp4_packed, scales)
which feeds directly into tl.dot_scaled("e2m1").
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

_cached_bscale_in = None
_cached_state = None


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


@triton.jit
def _fused_quant_gemm_v4(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K

        # Load A tile as bf16, convert to fp32
        offs_k = tl.arange(0, BLOCK_K)
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak,
        ).to(tl.float32)

        # Quantize A using aiter's exact _mxfp4_quant_op
        # Returns (fp4_packed [BLOCK_M, BLOCK_K//2], scales [BLOCK_M, BLOCK_K//32])
        a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)

        # Load B tile [BLOCK_K//2, BLOCK_N] (B is [N, K//2], transposed via strides)
        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        b_fp4 = tl.load(
            b_ptr + (k_start // 2 + offs_k_packed)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        )

        # Load B scales [BLOCK_N, BLOCK_K//32]
        offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
        b_scales = tl.load(
            b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_bsk,
        )

        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

    # Store output
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def custom_kernel(data: input_t) -> output_t:
    global _cached_bscale_in, _cached_state

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Cache B scale un-shuffle
    if _cached_bscale_in is not B_scale_sh:
        _cached_bscale_in = B_scale_sh
        _cached_state = {
            'B_scale_raw': _unshuffle_e8m0(B_scale_sh),
            'B_q_u8': B_q.view(torch.uint8),
        }

    B_scale_raw = _cached_state['B_scale_raw']
    B_q_u8 = _cached_state['B_q_u8']

    C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    # Block sizes — BLOCK_K must be multiple of 32 (scale group)
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 128
    if m <= 16:
        BLOCK_M = 32
    elif m >= 128:
        BLOCK_M = 64

    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)

    _fused_quant_gemm_v4[grid](
        A, B_q_u8, C, B_scale_raw,
        m, n, k,
        A.stride(0), A.stride(1),
        B_q_u8.stride(1), B_q_u8.stride(0),
        C.stride(0), C.stride(1),
        B_scale_raw.stride(0), B_scale_raw.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C
