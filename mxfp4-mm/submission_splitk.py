#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Split-K for large K cases + fused for small K.

For M=16,K=7168: only 17 output blocks without split-K (304 CUs underutilized).
Split-K=4 gives 68 blocks, much better GPU utilization.

Calls _gemm_afp4wfp4_kernel directly with custom configs per problem size.
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_kernel,
    _gemm_afp4wfp4_reduce_kernel,
    _get_config,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import get_splitk

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# Per-problem tuned configs: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_KSPLIT, num_warps, num_stages)
_CONFIGS = {
    # Large K — use split-K for more parallelism
    (16, 2112, 7168):  (32, 128, 256, 4, 4, 4, 2),   # split-K=4 → 17*4=68 blocks
    (64, 7168, 2048):  (64, 128, 256, 4, 2, 4, 2),   # split-K=2 → 56*2=112 blocks
    (256, 3072, 1536): (64, 128, 256, 4, 1, 4, 2),   # enough blocks already
}


def _direct_triton_gemm(A_fp4, B_q_u8, A_scale, B_scale_raw, m, n, k, config_tuple):
    """Call _gemm_afp4wfp4_kernel directly with custom config."""
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_KSPLIT, nw, ns = config_tuple

    # Handle split-K adjustments
    SPLITK_BLOCK_SIZE, BLOCK_K_adj, NUM_KSPLIT_adj = get_splitk(k, BLOCK_K, NUM_KSPLIT)
    BLOCK_K_adj = max(BLOCK_K_adj, 128)

    # Transpose B for kernel (B is [N, K//2], kernel wants [K//2, N])
    B_t = B_q_u8.T.contiguous() if not B_q_u8.T.is_contiguous() else B_q_u8.T

    if NUM_KSPLIT_adj > 1:
        # Split-K: write partial results, then reduce
        y_pp = torch.empty((NUM_KSPLIT_adj, m, n), dtype=torch.float32, device='cuda')

        grid = (NUM_KSPLIT_adj * triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _gemm_afp4wfp4_kernel[grid](
            A_fp4, B_t, y_pp, A_scale, B_scale_raw,
            m, n, k,
            A_fp4.stride(0), A_fp4.stride(1),
            B_t.stride(0), B_t.stride(1),
            y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
            A_scale.stride(0), A_scale.stride(1),
            B_scale_raw.stride(0), B_scale_raw.stride(1),
            BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K_adj,
            GROUP_SIZE_M=GROUP_SIZE_M, NUM_KSPLIT=NUM_KSPLIT_adj,
            SPLITK_BLOCK_SIZE=SPLITK_BLOCK_SIZE,
            num_warps=nw, num_stages=ns, waves_per_eu=0,
            matrix_instr_nonkdim=16, cache_modifier=".cg",
        )

        # Reduce partial results (sum across K splits)
        return y_pp.sum(dim=0).to(torch.bfloat16)
    else:
        # No split-K
        SPLITK_BLOCK_SIZE = 2 * k
        y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _gemm_afp4wfp4_kernel[grid](
            A_fp4, B_t, y, A_scale, B_scale_raw,
            m, n, k,
            A_fp4.stride(0), A_fp4.stride(1),
            B_t.stride(0), B_t.stride(1),
            0, y.stride(0), y.stride(1),  # stride_ck=0 for no split-K
            A_scale.stride(0), A_scale.stride(1),
            B_scale_raw.stride(0), B_scale_raw.stride(1),
            BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K_adj,
            GROUP_SIZE_M=GROUP_SIZE_M, NUM_KSPLIT=1,
            SPLITK_BLOCK_SIZE=SPLITK_BLOCK_SIZE,
            num_warps=nw, num_stages=ns, waves_per_eu=0,
            matrix_instr_nonkdim=16, cache_modifier=".cg",
        )
        return y


@triton.jit
def _fused_quant_gemm(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
        a_tile = tl.load(a_ptr + offs_m[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak).to(tl.float32)
        a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)
        b_fp4 = tl.load(b_ptr + (k_start // 2 + tl.arange(0, BLOCK_K // 2))[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_scales = tl.load(b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + tl.arange(0, BLOCK_K // SCALE_GROUP))[None, :] * stride_bsk)
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
        # FUSED PATH — single kernel
        C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        grid = (triton.cdiv(m, 32) * triton.cdiv(n, 128),)
        _fused_quant_gemm[grid](
            A, _bq_u8, C, _bscale_raw, m, n, k,
            A.stride(0), A.stride(1), _bq_u8.stride(1), _bq_u8.stride(0),
            C.stride(0), C.stride(1), _bscale_raw.stride(0), _bscale_raw.stride(1),
            BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE_M=4,
            num_warps=4, num_stages=2,
        )
        return C
    else:
        # SEPARATE PATH with split-K
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        cfg = _CONFIGS.get((m, n, k))
        if cfg:
            return _direct_triton_gemm(A_fp4, _bq_u8, A_scale, _bscale_raw, m, n, k, cfg)
        else:
            # Fallback: default wrapper
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
            return gemm_afp4wfp4(A_fp4, _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)
