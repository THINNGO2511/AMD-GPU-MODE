#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Fused GEMM v3 — Use aiter's dynamic_mxfp4_quant for A, then our own Triton
GEMM kernel with tl.dot_scaled. This isolates whether the issue is with
our quantization or with how we call tl.dot_scaled.

If this works, we know dot_scaled calling works and the issue was quant.
If it fails, the issue is with how we call dot_scaled.
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
from aiter.ops.triton.quant import dynamic_mxfp4_quant

_cached_bscale_in = None
_cached_state = None


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


@triton.jit
def _simple_fp4_gemm_kernel(
    a_fp4_ptr, b_fp4_ptr, c_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Simple GEMM kernel using tl.dot_scaled with pre-quantized A and B."""
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K // 2)

    # Pointers
    a_ptrs = a_fp4_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_fp4_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)
    a_scale_ptrs = a_scales_ptr + offs_m[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    b_scale_ptrs = b_scales_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for _ in range(num_k_iters):
        a_scales = tl.load(a_scale_ptrs)
        b_scales = tl.load(b_scale_ptrs)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        acc = tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1", acc)

        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        a_scale_ptrs += (BLOCK_K // SCALE_GROUP) * stride_ask
        b_scale_ptrs += (BLOCK_K // SCALE_GROUP) * stride_bsk

    c = acc.to(c_ptr.type.element_ty)
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

    # Cache B un-shuffle
    if _cached_bscale_in is not B_scale_sh:
        _cached_bscale_in = B_scale_sh
        _cached_state = {
            'B_scale_raw': _unshuffle_e8m0(B_scale_sh),
            'B_q_u8': B_q.view(torch.uint8),
        }

    B_scale_raw = _cached_state['B_scale_raw']
    B_q_u8 = _cached_state['B_q_u8']

    # Quantize A using aiter (known correct)
    A_fp4, A_scale = dynamic_mxfp4_quant(A)

    # Output
    C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    # Config — match the aiter kernel's approach
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128
    if m <= 16:
        BLOCK_M = 32
    if k < 128:
        BLOCK_K = max(64, triton.next_power_of_2(k))

    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)

    _simple_fp4_gemm_kernel[grid](
        A_fp4, B_q_u8, C, A_scale, B_scale_raw,
        m, n, k,
        A_fp4.stride(0), A_fp4.stride(1),
        B_q_u8.stride(1), B_q_u8.stride(0),
        C.stride(0), C.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale_raw.stride(0), B_scale_raw.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C
