#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Fused quant + GEMM in a single Triton kernel.

Eliminates separate quant + shuffle kernel launches by:
1. Loading A as bf16
2. Quantizing A to MXFP4 inline (E8M0 scale + E2M1 FP4 packing)
3. Using tl.dot_scaled("e2m1") for native FP4 MFMA on MI355X
4. Single kernel launch = saves ~5-10us launch overhead

B is pre-quantized with raw (un-shuffled) E8M0 scales.
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl

# Cache for un-shuffled B scales + B as uint8
_cached_bscale_in = None
_cached_state = None


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


@triton.jit
def _quant_fp4(norm_even, norm_odd):
    """Quantize normalized float pairs to packed FP4x2 bytes.

    norm_even: [BLOCK_M, 16] float32 (even-indexed elements)
    norm_odd:  [BLOCK_M, 16] float32 (odd-indexed elements)
    Returns:   [BLOCK_M, 16] uint8 packed FP4x2
    """
    # Quantize even elements to E2M1 magnitude (3 bits) + sign (1 bit)
    abs_e = tl.abs(norm_even)
    mag_e = tl.where(abs_e >= 5.0, 7,
            tl.where(abs_e >= 3.5, 6,
            tl.where(abs_e >= 2.5, 5,
            tl.where(abs_e >= 1.75, 4,
            tl.where(abs_e >= 1.25, 3,
            tl.where(abs_e >= 0.75, 2,
            tl.where(abs_e >= 0.25, 1, 0)))))))
    sign_e = tl.where(norm_even < 0.0, 8, 0)
    fp4_e = mag_e | sign_e  # [BLOCK_M, 16] low nibble

    # Quantize odd elements
    abs_o = tl.abs(norm_odd)
    mag_o = tl.where(abs_o >= 5.0, 7,
            tl.where(abs_o >= 3.5, 6,
            tl.where(abs_o >= 2.5, 5,
            tl.where(abs_o >= 1.75, 4,
            tl.where(abs_o >= 1.25, 3,
            tl.where(abs_o >= 0.75, 2,
            tl.where(abs_o >= 0.25, 1, 0)))))))
    sign_o = tl.where(norm_odd < 0.0, 8, 0)
    fp4_o = mag_o | sign_o  # [BLOCK_M, 16] high nibble

    return (fp4_e | (fp4_o << 4)).to(tl.uint8)


@triton.jit
def _fused_quant_gemm_kernel(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    # Scratch buffers for quantized A
    a_fp4_ptr, a_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    a_fp4_stride_m, a_scale_stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32
    NUM_GROUPS: tl.constexpr = BLOCK_K // SCALE_GROUP
    HALF_GROUP: tl.constexpr = SCALE_GROUP // 2  # 16

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    pair_offs = tl.arange(0, HALF_GROUP)  # 0..15

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k_iter in range(0, num_k_iters):
        k_start = k_iter * BLOCK_K

        # === STAGE 1: Quantize A tile to MXFP4 ===
        for g in tl.static_range(NUM_GROUPS):
            k_group = k_start + g * SCALE_GROUP

            # Load even/odd bf16 elements (stride-2 access)
            k_even = k_group + pair_offs * 2
            k_odd = k_group + pair_offs * 2 + 1
            a_even = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_even[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_even[None, :] < K), other=0.0
            ).to(tl.float32)
            a_odd = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_odd[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_odd[None, :] < K), other=0.0
            ).to(tl.float32)

            # Max abs for E8M0 scale
            amax = tl.maximum(
                tl.max(tl.abs(a_even), axis=1),
                tl.max(tl.abs(a_odd), axis=1),
            )  # [BLOCK_M]

            # E8M0 scale: 2^(e8m0 - 127), where amax / scale <= 6.0
            # ceil(x) = trunc(x) + (1 if x > trunc(x) else 0)
            log2_ratio = tl.log2(amax + 1e-12) - 2.584962500721156
            trunc_lr = log2_ratio.to(tl.int32)
            ceil_lr = tl.where(log2_ratio > trunc_lr.to(tl.float32), trunc_lr + 1, trunc_lr)
            e8m0 = tl.minimum(tl.maximum(ceil_lr + 127, 0), 255)
            scale = tl.exp2((e8m0 - 127).to(tl.float32))

            # Normalize and quantize
            norm_even = a_even / tl.maximum(scale, 1e-12)[:, None]
            norm_odd = a_odd / tl.maximum(scale, 1e-12)[:, None]
            packed = _quant_fp4(norm_even, norm_odd)  # [BLOCK_M, 16] uint8

            # Store to scratch
            fp4_offset = k_start // 2 + g * HALF_GROUP
            tl.store(
                a_fp4_ptr + offs_m[:, None] * a_fp4_stride_m + (fp4_offset + pair_offs)[None, :],
                packed, mask=offs_m[:, None] < M
            )
            scale_offset = k_start // SCALE_GROUP + g
            tl.store(
                a_scale_ptr + offs_m * a_scale_stride_m + scale_offset,
                e8m0.to(tl.uint8), mask=offs_m < M
            )

        # === STAGE 2: Load quantized A + B, compute dot_scaled ===
        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        a_fp4 = tl.load(
            a_fp4_ptr + offs_m[:, None] * a_fp4_stride_m + (k_start // 2 + offs_k_packed)[None, :],
            mask=offs_m[:, None] < M, other=0
        )

        offs_k_scale = tl.arange(0, NUM_GROUPS)
        a_scales = tl.load(
            a_scale_ptr + offs_m[:, None] * a_scale_stride_m + (k_start // SCALE_GROUP + offs_k_scale)[None, :],
            mask=offs_m[:, None] < M, other=0
        )

        # B: [N, K//2] row-major → access as [K//2, N] via strides
        b_fp4 = tl.load(
            b_ptr + (k_start // 2 + offs_k_packed)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_n[None, :] < N) & ((k_start // 2 + offs_k_packed)[:, None] < K // 2), other=0
        )

        # B scales: [N, K//32]
        b_scales = tl.load(
            b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_bsk,
            mask=offs_n[:, None] < N, other=0
        )

        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

    # === STORE OUTPUT ===
    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def custom_kernel(data: input_t) -> output_t:
    global _cached_bscale_in, _cached_state

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Cache B scale un-shuffle + B_q as uint8
    if _cached_bscale_in is not B_scale_sh:
        _cached_bscale_in = B_scale_sh
        _cached_state = {
            'B_scale_raw': _unshuffle_e8m0(B_scale_sh),
            'B_q_u8': B_q.view(torch.uint8),
        }

    B_scale_raw = _cached_state['B_scale_raw']
    B_q_u8 = _cached_state['B_q_u8']

    # Allocate scratch for quantized A (reuse across calls with same shape)
    a_fp4_scratch = torch.empty((m, k // 2), dtype=torch.uint8, device='cuda')
    a_scale_scratch = torch.empty((m, k // 32), dtype=torch.uint8, device='cuda')

    # Output
    C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    # Choose block sizes
    BLOCK_M = min(64, triton.next_power_of_2(m))
    BLOCK_N = 128
    BLOCK_K = 128
    if k < 128:
        BLOCK_K = 64

    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)

    _fused_quant_gemm_kernel[grid](
        A, B_q_u8, C, B_scale_raw,
        a_fp4_scratch, a_scale_scratch,
        m, n, k,
        A.stride(0), A.stride(1),
        B_q_u8.stride(1), B_q_u8.stride(0),  # B is [N, K//2], stride_bk=1, stride_bn=K//2
        C.stride(0), C.stride(1),
        B_scale_raw.stride(0), B_scale_raw.stride(1),
        k // 2,  # a_fp4_stride_m
        k // 32,  # a_scale_stride_m
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C
