#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Custom fused quant+GEMM for ALL K sizes.
Key insight: quantize each BLOCK_K chunk of A independently (MXFP4 is per-32-element blocks).
The issue with previous fused kernel for large K was NOT redundant quant — it's that
_mxfp4_quant_op is called per K-iteration which is correct (each 32-element block is independent).

The real overhead for separate path is:
1. dynamic_mxfp4_quant kernel launch (~2μs)
2. gemm_afp4wfp4 kernel launch (~2μs)
3. Intermediate buffer allocation

Fused kernel eliminates all 3. Let's make it work well for ALL K.
The previous issue: fused was slower for K>1024 because of poor config selection.
Solution: aggressive autotuning with AMD opts for large K.
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_c_cache = {}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


@triton.autotune(
    configs=[
        # K=512 configs (proven)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2),
        # Large K configs - try larger BLOCK_N for better compute/memory ratio
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2),
        # num_stages=1 for less register pressure on large K
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=1),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_quant_gemm(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
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
    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_iter * BLOCK_K
        # Load A tile as bf16, convert to fp32 for quant
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak,
        ).to(tl.float32)
        # Quantize A block to MXFP4 inline
        a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)
        # Load B block (already MXFP4)
        b_fp4 = tl.load(
            b_ptr + (k_start // 2 + tl.arange(0, BLOCK_K // 2))[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        )
        b_scales = tl.load(
            b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + tl.arange(0, BLOCK_K // SCALE_GROUP))[None, :] * stride_bsk,
        )
        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c,
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    # Use fused kernel for ALL K sizes
    c_key = (m, n)
    if c_key not in _c_cache:
        _c_cache[c_key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[c_key]
    grid = lambda META: (triton.cdiv(m, META['BLOCK_M']) * triton.cdiv(n, META['BLOCK_N']),)
    _fused_quant_gemm[grid](
        A, _bq_u8, C, _bscale_raw,
        m, n, k,
        A.stride(0), A.stride(1),
        _bq_u8.stride(1), _bq_u8.stride(0),
        C.stride(0), C.stride(1),
        _bscale_raw.stride(0), _bscale_raw.stride(1),
    )
    return C
