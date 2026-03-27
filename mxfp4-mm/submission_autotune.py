#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Autotuned fused quant+GEMM kernel.

Uses @triton.autotune to search across BLOCK_M/N/K, GROUP_SIZE_M, num_warps
for each (M, N, K) problem size. Fused for K≤2048, separate for K=7168.
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


@triton.autotune(
    configs=[
        # BK=128, small tiles — max parallelism for small M
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=8, num_stages=2),
        # BK=128, larger M tiles
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=0),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_quant_gemm_autotuned(
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
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak,
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
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c, mask=c_mask)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    if k <= 1024:
        # FUSED AUTOTUNED PATH (best for small K)
        C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        grid = lambda META: (triton.cdiv(m, META['BLOCK_M']) * triton.cdiv(n, META['BLOCK_N']),)
        _fused_quant_gemm_autotuned[grid](
            A, _bq_u8, C, _bscale_raw,
            m, n, k,
            A.stride(0), A.stride(1),
            _bq_u8.stride(1), _bq_u8.stride(0),
            C.stride(0), C.stride(1),
            _bscale_raw.stride(0), _bscale_raw.stride(1),
        )
        return C
    else:
        # SEPARATE PATH for K>1024 (tuned gemm_afp4wfp4 faster for large K)
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4, _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)
