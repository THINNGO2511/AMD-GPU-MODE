#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — XCD remap patch for gemm_a16wfp4.

The installed gemm_a16wfp4 kernel does NOT have remap_xcd, but the
gemm_afp4wfp4 kernel DOES. On MI355X with 8 XCDs, this remap is critical
for L2 cache locality.

Strategy:
  1. At import time, write a patched _gemm_a16wfp4_kernel to /tmp/ that
     adds remap_xcd (identical to how gemm_afp4wfp4 does it)
  2. Monkey-patch the installed module so gemm_a16wfp4() uses our patched kernel
  3. Keep all proven optimizations (prewarm, K7168 config, K1536 afp4 path, output cache)

The patch adds exactly 2 lines to the kernel:
  - Import remap_xcd
  - pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
"""
from task import input_t, output_t
import torch
import sys
import os
import importlib
import importlib.util

# ── Global caches ──
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_warmed = False
_patched = False

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# ── Patched kernel source ──
# This is the EXACT _gemm_a16wfp4_kernel from the latest aiter GitHub,
# with remap_xcd added (same pattern as gemm_afp4wfp4).
# Changes marked with "# PATCH:" comments.

_PATCHED_KERNEL_SOURCE = r'''
# SPDX-License-Identifier: MIT
# Patched _gemm_a16wfp4_kernel with remap_xcd for MI355X (8 XCDs)

import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

import triton

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Self-contained remap_xcd (from aiter pid_preprocessing.py on GitHub main)
# Included inline so it works even if the runner has an older aiter version
@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid

_gemm_a16wfp4_repr = make_kernel_repr(
    "_gemm_a16wfp4_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "cache_modifier",
        "NUM_KSPLIT",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a16wfp4_repr)
def _gemm_a16wfp4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    GRID_MN: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    # PATCH: remap for XCD locality on MI355X (8 XCDs, same as gemm_afp4wfp4)
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_k_bf16 = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split_bf16 = pid_k * SPLITK_BLOCK_SIZE + offs_k_bf16
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split_bf16[None, :] * stride_ak
        )

        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )
        # Create pointers for the first block of A and B scales
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
        )
        # B scales are N x K even though B operand is K x N.
        b_scale_ptrs = (
            b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            b_scales = tl.load(b_scale_ptrs)
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a_bf16 = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a_bf16 = tl.load(
                    a_ptrs,
                    mask=offs_k_bf16[None, :] < 2 * K - k * BLOCK_SIZE_K,
                    other=0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2),
                    other=0,
                    cache_modifier=cache_modifier,
                )

            a, a_scales = _mxfp4_quant_op(a_bf16, BLOCK_SIZE_K, BLOCK_SIZE_M, 32)

            accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if ATOMIC_ADD:
            tl.atomic_add(c_ptrs, c, mask=c_mask, sem="relaxed")
        else:
            tl.store(c_ptrs, c, mask=c_mask)


def _get_config(
    M,
    N,
    K,
    shuffle=False,
):
    shuffle_suffix = "_PRESHUFFLED" if shuffle else ""
    config_name = f"GEMM-A16WFP4{shuffle_suffix}"
    return get_gemm_config(config_name, M, N, 2 * K)
'''


def _apply_patch():
    """Write patched kernel to /tmp and monkey-patch the installed module."""
    global _patched
    if _patched:
        return
    _patched = True

    try:
        patch_dir = "/tmp/aiter_xcd_patch"
        os.makedirs(patch_dir, exist_ok=True)
        patch_file = os.path.join(patch_dir, "gemm_a16wfp4_xcd.py")

        with open(patch_file, "w") as f:
            f.write(_PATCHED_KERNEL_SOURCE)

        # Load the patched module
        spec = importlib.util.spec_from_file_location(
            "gemm_a16wfp4_xcd", patch_file
        )
        patched_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patched_mod)

        patched_kernel = patched_mod._gemm_a16wfp4_kernel

        # Monkey-patch the inner kernel module
        import aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 as kernel_mod
        kernel_mod._gemm_a16wfp4_kernel = patched_kernel

        # CRITICAL: Also patch the wrapper module's already-imported reference.
        # The wrapper does `from ... import _gemm_a16wfp4_kernel` at module load,
        # so we must overwrite its local binding too.
        import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as wrapper_mod
        wrapper_mod._gemm_a16wfp4_kernel = patched_kernel

        print("[XCD-PATCH] SUCCESS: Patched _gemm_a16wfp4_kernel with remap_xcd", file=sys.stderr)
    except Exception as e:
        print(f"[XCD-PATCH] FAILED: {e} — falling back to unpatched kernel", file=sys.stderr)


def _prewarm(B_q_u8, B_scale_raw):
    """Pre-warm Triton JIT for all benchmark shapes (including patched kernel)."""
    global _warmed
    if _warmed:
        return
    _warmed = True

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    n_actual = B_q_u8.shape[0]
    k_actual = B_q_u8.shape[1] * 2

    # All 6 benchmark shapes: (M, N, K)
    shapes = [
        (4, 2880, 512), (16, 2112, 7168), (32, 4096, 512),
        (32, 2880, 512), (64, 7168, 2048), (256, 3072, 1536),
    ]

    for m, n, k in shapes:
        if n != n_actual or k != k_actual:
            continue
        try:
            dummy_a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
            if k == 1536:
                A_fp4, A_scale = dynamic_mxfp4_quant(dummy_a)
                _ = gemm_afp4wfp4(A_fp4.view(torch.uint8), B_q_u8, A_scale, B_scale_raw,
                                  dtype=torch.bfloat16)
            else:
                y = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
                cfg = _K7168_CONFIG if k == 7168 else None
                _ = gemm_a16wfp4(dummy_a, B_q_u8, B_scale_raw, dtype=torch.bfloat16,
                                y=y, config=cfg)
            print(f"[PREWARM] M={m},N={n},K={k} OK", file=sys.stderr)
        except Exception as e:
            print(f"[PREWARM] M={m},N={n},K={k} FAIL: {e}", file=sys.stderr)
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        # Apply XCD remap patch BEFORE any gemm_a16wfp4 import/call
        _apply_patch()
        _prewarm(_bq_u8, _bscale_raw)

    if k == 1536:
        # afp4wfp4 path — already HAS remap_xcd in its kernel
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    # a16wfp4 path — NOW uses our patched kernel with remap_xcd
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    cfg = _K7168_CONFIG if k == 7168 else None

    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)
