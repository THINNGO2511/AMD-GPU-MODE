#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Config sweep: find optimal BN/warps/stages for each of the 6 benchmark shapes.
Keeps kernel code identical to exp_fused_diag.py, only varies launch params.
Prints all results to STDOUT for popcorn visibility.
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

# ---- helpers ----

def _unshuffle_e8m0(s):
    s = s.view(torch.uint8)
    sm, sn = s.shape
    return s.view(sm // 32, sn // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(sm, sn)

def _shuffle(s):
    s = s.clone().view(torch.uint8)
    sm, sn = s.shape
    return s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm // 32, sn * 32)


# ---- kernel (identical to exp_fused_diag.py) ----

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


# ---- split-K kernel (accumulates partial sums per K-split) ----

@triton.jit
def _fused_splitk_kernel(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    NUM_KSPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    """Split-K variant: each program handles a K-slice, writes partial f32 sum."""
    SCALE_GROUP_SIZE: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_mn = tl.cdiv(M, BLOCK_M) * num_pid_n
    pid_ks = pid // num_mn
    pid_mn = pid % num_mn
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # K range for this split
    k_per_split = tl.cdiv(K, NUM_KSPLIT)
    # Align to BLOCK_K
    k_per_split = tl.cdiv(k_per_split, BLOCK_K) * BLOCK_K
    k_start = pid_ks * k_per_split
    k_end = min(k_start + k_per_split, K)
    num_k_iter = tl.cdiv(k_end - k_start, BLOCK_K)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k_packed = tl.arange(0, BLOCK_K // 2)

    a_bf16_ptrs = a_ptr + offs_am[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak
    b_ptrs = b_ptr + (k_start // 2 + offs_k_packed)[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    offs_bsn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % (N // 32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)
    b_scale_ptrs = b_scales_ptr + offs_bsn[:, None] * stride_bsn + (k_start + offs_ks)[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ki in range(0, num_k_iter):
        a_bf16 = tl.load(a_bf16_ptrs).to(tl.float32)
        a_fp4, a_scales = _mxfp4_quant_op(a_bf16, BLOCK_K, BLOCK_M, SCALE_GROUP_SIZE)
        b = tl.load(b_ptrs)
        b_scales = tl.load(b_scale_ptrs).reshape(
            BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
        ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
        accumulator += tl.dot_scaled(a_fp4, a_scales, "e2m1", b, b_scales, "e2m1")
        a_bf16_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        b_scale_ptrs += BLOCK_K * stride_bsk

    # Write partial f32 sum to split plane
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # c_ptr layout: (NUM_KSPLIT, M, N) in float32
    offset = pid_ks * M * N
    tl.store(c_ptr + offset + offs_cm[:, None] * stride_cn * 0 + offs_cm[:, None] * N + offs_cn[None, :],
             accumulator, mask=c_mask)


def _run_fused(A_bf16, B_q, B_sc_sh, m, n, k, BM, BN, BK, nw, ns):
    C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
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
        num_warps=nw, num_stages=ns,
        matrix_instr_nonkdim=16,
    )
    return C


def _run_splitk(A_bf16, B_q, B_sc_sh, m, n, k, BM, BN, BK, nw, ns, ksplit):
    # Partial sums in f32
    C_partial = torch.zeros((ksplit, m, n), dtype=torch.float32, device='cuda')
    grid = (ksplit * triton.cdiv(m, BM) * triton.cdiv(n, BN),)
    _fused_splitk_kernel[grid](
        A_bf16, B_q, C_partial, B_sc_sh,
        m, n, k,
        A_bf16.stride(0), A_bf16.stride(1),
        B_q.stride(1), B_q.stride(0),
        C_partial.stride(1), C_partial.stride(2),
        B_sc_sh.stride(0), B_sc_sh.stride(1),
        NUM_KSPLIT=ksplit,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        mfma_nonkdim=16,
        num_warps=nw, num_stages=ns,
        matrix_instr_nonkdim=16,
    )
    return C_partial.sum(dim=0).to(torch.bfloat16)


# ---- sweep logic ----

def _time_fn(fn, niters=10):
    """Time a callable over niters, return mean microseconds."""
    # Warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(niters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / niters * 1e6


def _sweep_k512(A, bq, braw, bsh, m, n, k):
    """Sweep BN/warps/stages for K=512 shapes (single-pass, no split-K)."""
    pad = (32 - m % 32) % 32
    A_pad = torch.nn.functional.pad(A, (0, 0, 0, pad), value=0.0) if pad > 0 else A
    mp = A_pad.shape[0]

    # Aiter baseline
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    dt_aiter = _time_fn(lambda: gemm_a16wfp4(A, bq, braw, dtype=torch.bfloat16))
    print(f"SWEEP M{m}N{n}K{k} AITER_BASELINE: {dt_aiter:.1f}us", flush=True)

    configs = [
        # (BN, num_warps, num_stages)
        (32, 4, 2),
        (64, 4, 2),   # current default
        (64, 4, 3),
        (64, 8, 2),
        (128, 4, 2),
        (128, 8, 2),
        (128, 4, 3),
        (256, 4, 2),
        (256, 8, 2),
    ]

    best_t = 1e9
    best_cfg = None
    for bn, nw, ns in configs:
        if bn > n:
            continue
        BM, BK = 32, 256
        tag = f"BN{bn}_NW{nw}_NS{ns}"
        try:
            dt = _time_fn(lambda bn=bn, nw=nw, ns=ns: _run_fused(A_pad, bq, bsh, mp, n, k, BM, bn, BK, nw, ns))
            speedup = dt_aiter / dt if dt > 0 else 0
            marker = " <<<BEST" if dt < best_t else ""
            print(f"SWEEP M{m}N{n}K{k} {tag}: {dt:.1f}us (x{speedup:.2f} vs aiter){marker}", flush=True)
            if dt < best_t:
                best_t = dt
                best_cfg = tag
        except Exception as e:
            print(f"SWEEP M{m}N{n}K{k} {tag}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)

    if best_cfg:
        print(f"SWEEP M{m}N{n}K{k} WINNER: {best_cfg} at {best_t:.1f}us (aiter={dt_aiter:.1f}us, x{dt_aiter/best_t:.2f})", flush=True)


def _sweep_splitk(A, bq, braw, bsh, m, n, k):
    """Sweep split-K configs for K=7168/2048/1536 shapes."""
    pad = (32 - m % 32) % 32
    A_pad = torch.nn.functional.pad(A, (0, 0, 0, pad), value=0.0) if pad > 0 else A
    mp = A_pad.shape[0]

    # Aiter baseline (with K7168 config if applicable)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    K7_cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
              "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
              "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
              "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024}
    cfg_aiter = K7_cfg if k == 7168 else None
    dt_aiter = _time_fn(lambda: gemm_a16wfp4(A, bq, braw, dtype=torch.bfloat16, config=cfg_aiter))
    print(f"SWEEP M{m}N{n}K{k} AITER_BASELINE: {dt_aiter:.1f}us", flush=True)

    # Also test afp4wfp4 path for K=1536 (known to be better)
    if k == 1536:
        try:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
            def _afp4_path():
                af, asc = dynamic_mxfp4_quant(A)
                return gemm_afp4wfp4(af.view(torch.uint8), bq, asc, braw, dtype=torch.bfloat16)
            dt_afp4 = _time_fn(_afp4_path)
            print(f"SWEEP M{m}N{n}K{k} AFP4WFP4_BASELINE: {dt_afp4:.1f}us", flush=True)
        except Exception as e:
            print(f"SWEEP M{m}N{n}K{k} AFP4WFP4_BASELINE: FAILED {str(e)[:100]}", flush=True)

    # Single-pass fused (no split-K) as reference
    BM, BK = 32, 256
    for bn, nw, ns in [(64, 4, 2), (128, 4, 2)]:
        if bn > n:
            continue
        tag = f"NOSPLIT_BN{bn}_NW{nw}_NS{ns}"
        try:
            dt = _time_fn(lambda bn=bn, nw=nw, ns=ns: _run_fused(A_pad, bq, bsh, mp, n, k, BM, bn, BK, nw, ns))
            print(f"SWEEP M{m}N{n}K{k} {tag}: {dt:.1f}us", flush=True)
        except Exception as e:
            print(f"SWEEP M{m}N{n}K{k} {tag}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)

    # Split-K configs
    if k == 7168:
        ksplits = [2, 4, 7, 8, 14]
    elif k == 2048:
        ksplits = [2, 4]
    elif k == 1536:
        ksplits = [2, 3]
    else:
        ksplits = [2, 4]

    best_t = 1e9
    best_cfg = None
    for ksplit in ksplits:
        for bn, nw, ns in [(64, 4, 2), (64, 8, 2), (128, 4, 2), (128, 8, 2)]:
            if bn > n:
                continue
            # Check K divisibility: each split must be divisible by BK=256
            k_per_split = ((k + ksplit - 1) // ksplit + BK - 1) // BK * BK
            if k_per_split < BK:
                continue
            tag = f"KS{ksplit}_BN{bn}_NW{nw}_NS{ns}"
            try:
                dt = _time_fn(
                    lambda ksplit=ksplit, bn=bn, nw=nw, ns=ns:
                        _run_splitk(A_pad, bq, bsh, mp, n, k, BM, bn, BK, nw, ns, ksplit)
                )
                marker = " <<<BEST" if dt < best_t else ""
                print(f"SWEEP M{m}N{n}K{k} {tag}: {dt:.1f}us{marker}", flush=True)
                if dt < best_t:
                    best_t = dt
                    best_cfg = tag
            except Exception as e:
                print(f"SWEEP M{m}N{n}K{k} {tag}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)

    if best_cfg:
        print(f"SWEEP M{m}N{n}K{k} WINNER: {best_cfg} at {best_t:.1f}us (aiter={dt_aiter:.1f}us, x{dt_aiter/best_t:.2f})", flush=True)


def _do_sweep(A, bq, braw, bsh, m, n, k):
    sk = (m, n, k)
    if sk in _tested:
        return
    _tested.add(sk)

    if not _HAS_QOP:
        print(f"SWEEP: _mxfp4_quant_op NOT AVAILABLE", flush=True)
        return

    print(f"\n===== SWEEP M={m} N={n} K={k} =====", flush=True)
    try:
        if k == 512:
            _sweep_k512(A, bq, braw, bsh, m, n, k)
        else:
            _sweep_splitk(A, bq, braw, bsh, m, n, k)
    except Exception as e:
        print(f"SWEEP M{m}N{n}K{k} OUTER_FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)


# ---- production fallback (always returns correct result) ----

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

    # Run sweep diagnostics (once per shape)
    _do_sweep(A, _bq, _raw, _sh, m, n, k)

    # Always return correct result via aiter
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq, asc, _raw, dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, _bq, _raw, dtype=torch.bfloat16, config=_K7 if k == 7168 else None)
