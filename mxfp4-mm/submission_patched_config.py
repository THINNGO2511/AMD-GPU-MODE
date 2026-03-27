#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Monkey-patch _get_config for better Triton GEMM configs.

For large-K cases, the default _get_config might not return optimal configs.
We override it to try split-K and different tile sizes for our specific problem sizes.
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
import aiter.ops.triton.gemm.basic.gemm_afp4wfp4 as _gemm_wrapper

# Save original
_original_get_config = None
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# Custom configs for our problem sizes
_CUSTOM_CONFIGS = {
    # (M, N, K): config_dict
    # Try split-K=2 for M=16,K=7168 (only 17 blocks without split)
    (16, 2112, 7168): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 4, "NUM_KSPLIT": 2,
        "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    },
    # M=64,K=2048 — try larger BLOCK_K
    (64, 7168, 2048): {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1,
        "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    },
    # M=256,K=1536 — try different tiles
    (256, 3072, 1536): {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1,
        "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    },
}


def _patched_get_config(M, N, K):
    """Return custom config if available, else fall back to original."""
    # K in _get_config is K//2 (packed size), need to convert
    # Actually, looking at wrapper: M, K = x.shape; N, K = w.shape
    # x is [M, K//2], w is [N, K//2], so K here is K//2
    actual_K = K * 2
    key = (M, N, actual_K)
    if key in _CUSTOM_CONFIGS:
        return _CUSTOM_CONFIGS[key], "custom"
    return _original_get_config(M, N, K)


def _install_patch():
    global _original_get_config
    if _original_get_config is None:
        # Get the original _get_config from the wrapper module
        _original_get_config = _gemm_wrapper._get_config
        # Monkey-patch
        _gemm_wrapper._get_config = _patched_get_config


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
    _install_patch()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    if k <= 1024:
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
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return _gemm_wrapper.gemm_afp4wfp4(
            A_fp4, _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16
        )
