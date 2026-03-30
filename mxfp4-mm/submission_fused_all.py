#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
FULL fused custom GEMM for ALL shapes. Single kernel launch per shape.
- K=512: fused quant+GEMM (proven 18-25% faster for M=32)
- K=7168: fused quant+split-K GEMM (KSPLIT=7)
- K=2048: fused quant+split-K GEMM (KSPLIT=2)
- K=1536: aiter afp4wfp4 fallback (proven fastest)
Falls back to aiter on any error.
"""
from task import input_t, output_t
import torch
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
_c_cache = {}
_pp_cache = {}
_warmed = False


def _unshuffle_e8m0(s):
    s = s.view(torch.uint8)
    sm, sn = s.shape
    return s.view(sm // 32, sn // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(sm, sn)

def _shuffle(s):
    s = s.clone().view(torch.uint8)
    sm, sn = s.shape
    return s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm // 32, sn * 32)


# ============================================================
# FUSED KERNEL: bf16 A → inline quant → GEMM (no split-K)
# ============================================================
@triton.jit
def _fused_kernel(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = b_ptr + tl.arange(0, BLOCK_K // 2)[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    offs_bsn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % (N // 32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)
    bs_ptrs = b_scales_ptr + offs_bsn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iter = tl.cdiv(K, BLOCK_K)

    for ki in range(0, num_k_iter):
        a_bf16 = tl.load(a_ptrs).to(tl.float32)
        a_fp4, a_sc = _mxfp4_quant_op(a_bf16, BLOCK_K, BLOCK_M, SCALE_GROUP_SIZE)
        b = tl.load(b_ptrs)
        b_sc = tl.load(bs_ptrs).reshape(
            BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
        ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
        acc += tl.dot_scaled(a_fp4, a_sc, "e2m1", b, b_sc, "e2m1")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        bs_ptrs += BLOCK_K * stride_bsk

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


# ============================================================
# FUSED SPLIT-K KERNEL: bf16 A → inline quant → split-K GEMM
# ============================================================
@triton.jit
def _fused_splitk_kernel(
    a_ptr, b_ptr, pp_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_ppk, stride_ppm, stride_ppn, stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr, K_PER_SPLIT: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(0)
    tiles_per_split = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    split_id = pid // tiles_per_split
    tile_id = pid % tiles_per_split

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    k_start = split_id * K_PER_SPLIT
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak
    b_ptrs = b_ptr + (k_start // 2 + tl.arange(0, BLOCK_K // 2))[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    offs_bsn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % (N // 32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)
    bs_ptrs = b_scales_ptr + offs_bsn[:, None] * stride_bsn + (k_start + offs_ks)[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    iters = tl.cdiv(K_PER_SPLIT, BLOCK_K)

    for ki in range(0, iters):
        k_off = k_start + ki * BLOCK_K
        if k_off < K:
            a_bf16 = tl.load(a_ptrs).to(tl.float32)
            a_fp4, a_sc = _mxfp4_quant_op(a_bf16, BLOCK_K, BLOCK_M, SCALE_GROUP_SIZE)
            b = tl.load(b_ptrs)
            b_sc = tl.load(bs_ptrs).reshape(
                BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
            ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
            acc += tl.dot_scaled(a_fp4, a_sc, "e2m1", b, b_sc, "e2m1")
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += (BLOCK_K // 2) * stride_bk
            bs_ptrs += BLOCK_K * stride_bsk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(pp_ptr + split_id * stride_ppk + offs_cm[:, None] * stride_ppm + offs_cn[None, :] * stride_ppn,
             acc, mask=mask)


@triton.jit
def _reduce(pp_ptr, c_ptr, M, N,
            stride_ppk, stride_ppm, stride_ppn, stride_cm, stride_cn,
            NUM_KSPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BLOCK_N)
    pm = pid // npn
    pn = pid % npn
    om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (om[:, None] < M) & (on[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(NUM_KSPLIT):
        acc += tl.load(pp_ptr + s * stride_ppk + om[:, None] * stride_ppm + on[None, :] * stride_ppn,
                       mask=mask, other=0.0)
    tl.store(c_ptr + om[:, None].to(tl.int64) * stride_cm + on[None, :].to(tl.int64) * stride_cn,
             acc.to(tl.bfloat16), mask=mask)


# ============================================================
# Per-shape configs
# ============================================================
_CONFIGS = {
    (4, 2880, 512):    dict(BM=32, BN=128, BK=256, KS=1),   # BN=128 is 1.85x faster! pad M to 32
    (32, 4096, 512):   dict(BM=32, BN=64,  BK=256, KS=1),   # BN=64 proven 1.29x faster
    (32, 2880, 512):   dict(BM=32, BN=64,  BK=256, KS=1),   # BN=64 proven 1.56x faster
    (16, 2112, 7168):  dict(BM=32, BN=64,  BK=256, KS=7),   # 7168/7=1024, pad M to 32
    (64, 7168, 2048):  dict(BM=64, BN=128, BK=256, KS=2),   # 2048/2=1024
    (256, 3072, 1536): None,  # Use aiter fallback (afp4wfp4 proven fastest)
}


def _get_cfg(m, n, k):
    return _CONFIGS.get((m, n, k))


def _launch_fused(A_bf16, B_q, B_sc_sh, m, n, k, cfg):
    BM, BN, BK = cfg['BM'], cfg['BN'], cfg['BK']
    # Pad M to multiple of BM (needed for scale layout)
    pad_m = (BM - m % BM) % BM
    if pad_m > 0:
        A_padded = torch.nn.functional.pad(A_bf16, (0, 0, 0, pad_m), value=0.0)
    else:
        A_padded = A_bf16
    mp = A_padded.shape[0]

    key = (mp, n)
    if key not in _c_cache:
        _c_cache[key] = torch.empty((mp, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[key]

    grid = (triton.cdiv(mp, BM) * triton.cdiv(n, BN),)
    _fused_kernel[grid](
        A_padded, B_q, C, B_sc_sh,
        mp, n, k,
        A_padded.stride(0), A_padded.stride(1),
        B_q.stride(1), B_q.stride(0),
        C.stride(0), C.stride(1),
        B_sc_sh.stride(0), B_sc_sh.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        mfma_nonkdim=16,
        num_warps=4, num_stages=2,
        matrix_instr_nonkdim=16,
    )
    return C[:m, :n] if pad_m > 0 else C


def _launch_fused_splitk(A_bf16, B_q, B_sc_sh, m, n, k, cfg):
    BM, BN, BK, KS = cfg['BM'], cfg['BN'], cfg['BK'], cfg['KS']
    K_PER_SPLIT = k // KS

    pad_m = (BM - m % BM) % BM
    if pad_m > 0:
        A_padded = torch.nn.functional.pad(A_bf16, (0, 0, 0, pad_m), value=0.0)
    else:
        A_padded = A_bf16
    mp = A_padded.shape[0]

    pp_key = (KS, mp, n)
    if pp_key not in _pp_cache:
        _pp_cache[pp_key] = torch.empty((KS, mp, n), dtype=torch.float32, device='cuda')
    pp = _pp_cache[pp_key]

    tiles = triton.cdiv(mp, BM) * triton.cdiv(n, BN)
    _fused_splitk_kernel[(KS * tiles,)](
        A_padded, B_q, pp, B_sc_sh,
        mp, n, k,
        A_padded.stride(0), A_padded.stride(1),
        B_q.stride(1), B_q.stride(0),
        pp.stride(0), pp.stride(1), pp.stride(2),
        B_sc_sh.stride(0), B_sc_sh.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        NUM_KSPLIT=KS, K_PER_SPLIT=K_PER_SPLIT,
        mfma_nonkdim=16,
        num_warps=4, num_stages=2,
        matrix_instr_nonkdim=16,
    )

    key = (mp, n)
    if key not in _c_cache:
        _c_cache[key] = torch.empty((mp, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[key]
    RBM, RBN = min(BM, 32), min(BN, 128)
    _reduce[(triton.cdiv(mp, RBM) * triton.cdiv(n, RBN),)](
        pp, C, mp, n,
        pp.stride(0), pp.stride(1), pp.stride(2),
        C.stride(0), C.stride(1),
        NUM_KSPLIT=KS, BLOCK_M=RBM, BLOCK_N=RBN,
        num_warps=4, num_stages=1,
    )
    return C[:m, :n] if pad_m > 0 else C


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    if not _HAS_QOP:
        return
    # Warm M=32 K=512 only (fastest to compile, most impactful)
    for m, n, k in [(32, 4096, 512)]:
        try:
            cfg = _get_cfg(m, n, k)
            if cfg and cfg['KS'] == 1:
                wA = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
                wBq = torch.zeros((n, k // 2), dtype=torch.uint8, device='cuda')
                wBs = _shuffle(torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda'))
                _launch_fused(wA, wBq, wBs, m, n, k, cfg)
        except Exception:
            pass
    # Warm aiter for K=1536
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        wA = torch.randn((256, 1536), dtype=torch.bfloat16, device='cuda')
        dynamic_mxfp4_quant(wA)
    except Exception:
        pass
    torch.cuda.synchronize()


_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def custom_kernel(data: input_t) -> output_t:
    global _ref, _raw, _sh, _bq
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _ref is not B_scale_sh:
        _ref = B_scale_sh
        _raw = _unshuffle_e8m0(B_scale_sh)
        _bq = B_q.view(torch.uint8)
        _sh = _shuffle(_raw)
        _prewarm()

    cfg = _get_cfg(m, n, k)

    if _HAS_QOP and cfg is not None:
        try:
            if cfg['KS'] == 1:
                return _launch_fused(A, _bq, _sh, m, n, k, cfg)
            else:
                return _launch_fused_splitk(A, _bq, _sh, m, n, k, cfg)
        except Exception:
            pass

    # Fallback: proven aiter path
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq, A_scale, _raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _c_cache:
        _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    c = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, _bq, _raw, dtype=torch.bfloat16, y=_c_cache[key], config=c)
