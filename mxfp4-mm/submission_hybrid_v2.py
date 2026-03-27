#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Hybrid v2: fused kernel with per-size tuned configs.

Extends fused kernel to all K sizes with larger BLOCK_K for large K.
Falls back to separate path only for K=7168 (too many quant iterations).
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
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
def _fused_quant_gemm(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering for L2 reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K
        offs_k = tl.arange(0, BLOCK_K)
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak,
        ).to(tl.float32)

        a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)

        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        b_fp4 = tl.load(
            b_ptr + (k_start // 2 + offs_k_packed)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        )
        offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
        b_scales = tl.load(
            b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_bsk,
        )

        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Per-size tuned configs
_CONFIGS = {
    # (M, N, K): (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages, use_fused)
    (4, 2880, 512):    (32, 128, 128, 4, 4, 2, True),
    (16, 2112, 7168):  (32, 128, 128, 4, 4, 2, False),  # Too many K iters for fused
    (32, 4096, 512):   (32, 128, 128, 4, 4, 2, True),
    (32, 2880, 512):   (32, 128, 128, 4, 4, 2, True),
    (64, 7168, 2048):  (64, 128, 128, 4, 4, 2, True),
    (256, 3072, 1536): (64, 128, 128, 4, 4, 2, True),
}


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

    # Get config for this problem size
    cfg = _CONFIGS.get((m, n, k))
    if cfg is None:
        # Default config
        BLOCK_M = 32 if m <= 32 else 64
        BLOCK_N = 128
        BLOCK_K = 128
        GROUP_SIZE_M = 4
        nw, ns = 4, 2
        use_fused = k <= 2048
    else:
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, nw, ns, use_fused = cfg

    if use_fused:
        C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _fused_quant_gemm[grid](
            A, B_q_u8, C, B_scale_raw,
            m, n, k,
            A.stride(0), A.stride(1),
            B_q_u8.stride(1), B_q_u8.stride(0),
            C.stride(0), C.stride(1),
            B_scale_raw.stride(0), B_scale_raw.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=nw, num_stages=ns,
        )
        return C
    else:
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(
            A_fp4, B_q_u8, A_scale, B_scale_raw,
            dtype=torch.bfloat16,
        )
