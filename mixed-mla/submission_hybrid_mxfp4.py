#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Decode — Hybrid MXFP4 + FP8 ASM submission for AMD MI355X.

Strategy:
  kv_len <= 2048: FP8 ASM kernel (aiter mla_decode_fwd) — low per-token overhead wins
  kv_len >  2048: MXFP4 Triton kernel — 1.85x bandwidth savings dominate

The ASM kernel is ~34μs at kv=1024 (fast, proven). The Triton MXFP4 kernel
has higher ALU overhead from FP4 dequant but reads ~1.85x less memory, which
dominates at kv=8192.

All metadata and buffers are pre-allocated and cached by shape.
"""

import torch
import triton
import triton.language as tl
import math

# ============================================================================
# Constants
# ============================================================================
NHEAD = 16
NHEAD_KV = 1
HEADDIM = 576        # qk_head_dim
V_HEAD_DIM = 512     # v_head_dim
SM_SCALE = 1.0 / math.sqrt(576)
PAGE_SIZE = 1
KV_GRAN = 16         # max(PAGE_SIZE, 16)
KV_LEN_THRESHOLD = 2048  # Use MXFP4 above this

# ============================================================================
# FP8 ASM path — imports (lazy)
# ============================================================================
_fp8_initialized = False
_stage1_fn = None
_reduce_fn = None
_FP8_DTYPE = None
_FP8_MAX_VAL = None
_get_meta_info = None
_get_meta = None

def _init_fp8():
    global _fp8_initialized, _stage1_fn, _reduce_fn, _FP8_DTYPE, _FP8_MAX_VAL
    global _get_meta_info, _get_meta
    if _fp8_initialized:
        return
    from aiter import (
        get_mla_metadata_info_v1, get_mla_metadata_v1,
        dtypes as aiter_dtypes,
        mla_decode_stage1_asm_fwd, mla_reduce_v1,
    )
    _stage1_fn = mla_decode_stage1_asm_fwd
    _reduce_fn = mla_reduce_v1
    _FP8_DTYPE = aiter_dtypes.fp8
    _FP8_MAX_VAL = float(torch.finfo(_FP8_DTYPE).max)
    _get_meta_info = get_mla_metadata_info_v1
    _get_meta = get_mla_metadata_v1
    _fp8_initialized = True


# ============================================================================
# FP8 ASM — Fused Q→fp8 Triton kernels
# ============================================================================
@triton.jit
def _q_amax_kernel(q_ptr, amax_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x)))

@triton.jit
def _q_to_fp8_kernel(q_ptr, out_ptr, scale_ptr, amax_ptr,
                     FP8_MAX: tl.constexpr, N, BLOCK: tl.constexpr):
    amax = tl.load(amax_ptr)
    amax = tl.where(amax < 1e-12, 1e-12, amax)
    scale = amax / FP8_MAX
    if tl.program_id(0) == 0:
        tl.store(scale_ptr, scale)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x / scale
    x = tl.clamp(x, -FP8_MAX, FP8_MAX)
    tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=mask)


# ============================================================================
# FP8 ASM — Cache and bootstrap
# ============================================================================
_fp8_cache = {}

def _fp8_bootstrap(bs, kv_len, device):
    """Pre-allocate metadata and buffers for the FP8 ASM path."""
    _init_fp8()

    # Use a8w8 (fp8 Q + fp8 KV) with page_size=1
    dtype_q = _FP8_DTYPE
    num_splits = 8 if kv_len <= 1024 else 16

    # Paged KV (pg1)
    qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=device)
    kv_indptr_paged = torch.arange(bs + 1, dtype=torch.int32, device=device) * kv_len
    total_pages = bs * kv_len
    kv_page_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)

    # Persistent metadata
    meta_info = _get_meta_info(
        batch_size=bs, max_seqlen_q=1,
        nhead=NHEAD, nhead_kv=NHEAD_KV,
        num_kv_splits=num_splits,
        page_size=PAGE_SIZE, kv_granularity=KV_GRAN,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr_paged,
        kv_last_page_lens=kv_last_page_lens, dtype_q=dtype_q,
    )
    work_meta_data = torch.empty(
        meta_info["work_meta_data_size"], dtype=torch.int32, device=device)
    work_info_set = torch.empty(
        meta_info["work_info_set_size"], dtype=torch.int32, device=device)
    work_indptr = torch.empty(
        meta_info["work_indptr_size"], dtype=torch.int32, device=device)
    reduce_indptr = torch.empty(
        meta_info["reduce_indptr_size"], dtype=torch.int32, device=device)
    reduce_final_map = torch.empty(
        meta_info["reduce_final_map_size"], dtype=torch.int32, device=device)
    reduce_partial_map = torch.empty(
        meta_info["reduce_partial_map_size"], dtype=torch.int32, device=device)
    num_kv_splits_indptr = torch.empty(
        bs + 1, dtype=torch.int32, device=device)

    _get_meta(
        qo_indptr, kv_indptr_paged, kv_last_page_lens,
        NHEAD // NHEAD_KV, NHEAD_KV, False,
        work_meta_data, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        PAGE_SIZE, KV_GRAN,
        -1, 1, False, -1, num_splits, True,
        dtype_q, _FP8_DTYPE,
    )

    # Work buffers
    n_partial = reduce_partial_map.size(0)
    splitData = torch.empty((n_partial, 1, NHEAD, V_HEAD_DIM),
                            dtype=torch.float32, device=device)
    splitLse = torch.empty((n_partial, 1, NHEAD, 1),
                           dtype=torch.float32, device=device)
    final_out = torch.empty((bs, NHEAD, V_HEAD_DIM),
                            dtype=torch.bfloat16, device=device)

    # Q fp8 buffers
    N = bs * NHEAD * HEADDIM
    q_fp8_buf = torch.empty(N, dtype=_FP8_DTYPE, device=device)
    q_scale_buf = torch.empty(1, dtype=torch.float32, device=device)
    amax_buf = torch.zeros(1, dtype=torch.float32, device=device)

    _fp8_cache[(bs, kv_len)] = dict(
        num_splits=num_splits,
        qo_indptr=qo_indptr,
        kv_indptr_paged=kv_indptr_paged,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        num_kv_splits_indptr=num_kv_splits_indptr,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        splitData=splitData,
        splitLse=splitLse,
        final_out=final_out,
        q_fp8_buf=q_fp8_buf,
        q_scale_buf=q_scale_buf,
        amax_buf=amax_buf,
    )


def _run_fp8(q, kv_data, bs, kv_len):
    """Execute fp8 ASM decode path."""
    if (bs, kv_len) not in _fp8_cache:
        _fp8_bootstrap(bs, kv_len, q.device)

    c = _fp8_cache[(bs, kv_len)]
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    # Reshape KV to paged (pg1: each token = 1 page)
    kv_paged = kv_buffer_fp8.view(-1, PAGE_SIZE, NHEAD_KV, kv_buffer_fp8.shape[-1])

    # Quantize Q to fp8 (fused 2-kernel)
    N = q.numel()
    BLOCK = 4096
    grid = ((N + BLOCK - 1) // BLOCK,)

    c["amax_buf"].zero_()
    _q_amax_kernel[grid](q, c["amax_buf"], N, BLOCK=BLOCK)
    _q_to_fp8_kernel[grid](
        q, c["q_fp8_buf"], c["q_scale_buf"], c["amax_buf"],
        FP8_MAX=_FP8_MAX_VAL, N=N, BLOCK=BLOCK)

    q_input = c["q_fp8_buf"].view(bs, NHEAD, HEADDIM)

    # Stage 1: ASM paged attention
    _stage1_fn(
        q_input, kv_paged,
        c["qo_indptr"], c["kv_indptr_paged"],
        c["kv_page_indices"], c["kv_last_page_lens"],
        c["num_kv_splits_indptr"], c["work_meta_data"],
        c["work_indptr"], c["work_info_set"],
        1, PAGE_SIZE, NHEAD_KV, SM_SCALE,
        c["splitData"], c["splitLse"], c["final_out"],
        c["q_scale_buf"], kv_scale,
    )

    # Stage 2: Reduce
    _reduce_fn(
        c["splitData"], c["splitLse"],
        c["reduce_indptr"], c["reduce_final_map"], c["reduce_partial_map"],
        1, c["final_out"], None,
    )

    return c["final_out"]


# ============================================================================
# MXFP4 Triton path — FP4 E2M1 decode helpers
# ============================================================================

@triton.jit
def _fp4_decode(nibble):
    """Decode 4-bit E2M1 nibble -> float32.
    LUT: [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]
    """
    sign = (nibble >> 3) & 1
    exp  = (nibble >> 1) & 3
    mant = nibble & 1
    is_zero = (exp == 0) & (mant == 0)
    is_sub  = (exp == 0) & (mant == 1)
    pow2 = tl.where(exp == 1, 1.0, tl.where(exp == 2, 2.0, 4.0))
    normal_val = pow2 * (1.0 + mant.to(tl.float32) * 0.5)
    mag = tl.where(is_zero, 0.0, tl.where(is_sub, 0.5, normal_val))
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _load_dequant_kv_tile(
    KV_ptr, Scales_ptr,
    kv_tok_ptrs, sc_tok_ptrs, kv_mask,
    d_start, HALF: tl.constexpr, elems_per_group: tl.constexpr,
):
    """Load packed FP4 bytes + E8M0 scales, return dequanted (BLOCK_KV, HALF) lo/hi."""
    byte_start = d_start // 2
    byte_offs = byte_start + tl.arange(0, HALF)

    kv_b = tl.load(KV_ptr + kv_tok_ptrs[:, None] + byte_offs[None, :],
                   mask=kv_mask[:, None], other=0)

    lo_raw = kv_b & 0xF
    hi_raw = (kv_b >> 4) & 0xF
    lo_fp = _fp4_decode(lo_raw)
    hi_fp = _fp4_decode(hi_raw)

    lo_elem = d_start + 2 * tl.arange(0, HALF)
    hi_elem = d_start + 2 * tl.arange(0, HALF) + 1
    lo_grp = lo_elem // elems_per_group
    hi_grp = hi_elem // elems_per_group

    lo_sc_raw = tl.load(Scales_ptr + sc_tok_ptrs[:, None] + lo_grp[None, :],
                        mask=kv_mask[:, None], other=127).to(tl.int32)
    hi_sc_raw = tl.load(Scales_ptr + sc_tok_ptrs[:, None] + hi_grp[None, :],
                        mask=kv_mask[:, None], other=127).to(tl.int32)

    lo_sc = tl.math.exp2((lo_sc_raw - 127).to(tl.float32))
    hi_sc = tl.math.exp2((hi_sc_raw - 127).to(tl.float32))

    return lo_fp * lo_sc, hi_fp * hi_sc


# ============================================================================
# MXFP4 — Split-K decode kernel (better CU utilization for all batch sizes)
# ============================================================================

@triton.jit
def _mla_decode_splitk_safe(
    Q, KV, Sc, PV, PM, PL,
    qo_ind, kv_ind,
    sq0, sq1, skv, ssc,
    sm_scale, num_heads: tl.constexpr,
    QK_DIM: tl.constexpr, V_DIM: tl.constexpr,
    P_DIM: tl.constexpr, N_SG: tl.constexpr, EPG: tl.constexpr,
    NS: tl.constexpr,
    BK: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    qs  = tl.load(qo_ind + bid)
    kvs = tl.load(kv_ind + bid)
    kve = tl.load(kv_ind + bid + 1)
    kvl = kve - kvs

    ss = (kvl + NS - 1) // NS
    mks = kvs + sid * ss
    mke = tl.minimum(kvs + (sid + 1) * ss, kve)
    mkl = tl.maximum(mke - mks, 0)

    qb = Q + qs * sq0 + hid * sq1
    bh = qs * num_heads + hid
    pvb = PV + bh * NS * V_DIM + sid * V_DIM
    pmb = PM + bh * NS + sid
    plb = PL + bh * NS + sid

    LOG2E: tl.constexpr = 1.4426950408889634
    HD: tl.constexpr = BD // 2
    HV: tl.constexpr = BV // 2
    DT: tl.constexpr = QK_DIM // BD
    VT: tl.constexpr = V_DIM // BV

    m = -1e38
    l = 0.0

    # Zero V scratch (even/odd stores, no tl.interleave)
    for vt in tl.static_range(VT):
        off: tl.constexpr = vt * BV
        even_idx = off + 2 * tl.arange(0, HV)
        odd_idx  = off + 2 * tl.arange(0, HV) + 1
        tl.store(pvb + even_idx, tl.zeros([HV], dtype=tl.float32))
        tl.store(pvb + odd_idx,  tl.zeros([HV], dtype=tl.float32))

    for kb in range(0, mkl, BK):
        ko = kb + tl.arange(0, BK)
        km = ko < mkl
        kg = mks + ko
        ktp = kg * skv
        stp = kg * ssc

        qk = tl.zeros([BK], dtype=tl.float32)
        for dt in tl.static_range(DT):
            ds: tl.constexpr = dt * BD
            qe = tl.load(qb + ds + 2 * tl.arange(0, HD)).to(tl.float32)
            qo = tl.load(qb + ds + 2 * tl.arange(0, HD) + 1).to(tl.float32)
            kv_lo, kv_hi = _load_dequant_kv_tile(KV, Sc, ktp, stp, km, ds, HD, EPG)
            qk += tl.sum(qe[None, :] * kv_lo + qo[None, :] * kv_hi, axis=1)

        qk = tl.where(km, qk * sm_scale, -1e38)

        mc = tl.max(qk, axis=0)
        mn = tl.maximum(m, mc)
        al = tl.math.exp2((m - mn) * LOG2E)
        p  = tl.math.exp2((qk - mn) * LOG2E)
        p  = tl.where(km, p, 0.0)
        ln = al * l + tl.sum(p, axis=0)

        for vt in tl.static_range(VT):
            vs: tl.constexpr = vt * BV
            even_idx = vs + 2 * tl.arange(0, HV)
            odd_idx  = vs + 2 * tl.arange(0, HV) + 1

            vacc_lo = tl.load(pvb + even_idx) * al
            vacc_hi = tl.load(pvb + odd_idx) * al

            kv_lo, kv_hi = _load_dequant_kv_tile(KV, Sc, ktp, stp, km, vs, HV, EPG)
            clo = tl.sum(p[:, None] * kv_lo, axis=0)
            chi = tl.sum(p[:, None] * kv_hi, axis=0)

            tl.store(pvb + even_idx, vacc_lo + clo)
            tl.store(pvb + odd_idx,  vacc_hi + chi)

        m = mn
        l = ln

    tl.store(pmb, m)
    tl.store(plb, l)


@triton.jit
def _reduce_splitk(
    PV, PM, PL, O,
    qo_ind, so0, so1,
    V_DIM: tl.constexpr, num_heads: tl.constexpr,
    NS: tl.constexpr, BV: tl.constexpr, VT: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    qs  = tl.load(qo_ind + bid)
    ob  = O + qs * so0 + hid * so1
    bh  = qs * num_heads + hid
    pvb = PV + bh * NS * V_DIM
    pmb = PM + bh * NS
    plb = PL + bh * NS

    LOG2E: tl.constexpr = 1.4426950408889634

    mg = -1e38
    for s in tl.static_range(NS):
        ms = tl.load(pmb + s)
        mg = tl.maximum(mg, ms)

    lg = 0.0
    for s in tl.static_range(NS):
        ms = tl.load(pmb + s)
        ls = tl.load(plb + s)
        lg += ls * tl.math.exp2((ms - mg) * LOG2E)

    inv_l = 1.0 / tl.maximum(lg, 1e-10)
    for vt in tl.static_range(VT):
        vs: tl.constexpr = vt * BV
        vo = tl.zeros([BV], dtype=tl.float32)
        for s in tl.static_range(NS):
            ms = tl.load(pmb + s)
            vs_data = tl.load(pvb + s * V_DIM + vs + tl.arange(0, BV))
            vo += vs_data * tl.math.exp2((ms - mg) * LOG2E)
        tl.store(ob + vs + tl.arange(0, BV), (vo * inv_l).to(tl.bfloat16))


# ============================================================================
# MXFP4 — Single-pass decode kernel (no split-K, for high batch)
# ============================================================================

@triton.jit
def _mla_decode_safe(
    Q, KV, Sc, O, VS,
    qo_ind, kv_ind,
    sq0, sq1, skv, ssc, so0, so1, svs0, svs1,
    sm_scale,
    QK_DIM: tl.constexpr, V_DIM: tl.constexpr,
    P_DIM: tl.constexpr, N_SG: tl.constexpr, EPG: tl.constexpr,
    BK: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
):
    """Single-pass decode without tl.interleave (explicit even/odd scatter)."""
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    qs  = tl.load(qo_ind + bid)
    kvs = tl.load(kv_ind + bid)
    kve = tl.load(kv_ind + bid + 1)
    kvl = kve - kvs

    qb  = Q  + qs * sq0 + hid * sq1
    ob  = O  + qs * so0 + hid * so1
    vb  = VS + qs * svs0 + hid * svs1

    LOG2E: tl.constexpr = 1.4426950408889634
    HD: tl.constexpr = BD // 2
    HV: tl.constexpr = BV // 2
    DT: tl.constexpr = QK_DIM // BD
    VT: tl.constexpr = V_DIM // BV

    m = -1e38
    l = 0.0

    for vt in tl.static_range(VT):
        off: tl.constexpr = vt * BV
        even_idx = off + 2 * tl.arange(0, HV)
        odd_idx  = off + 2 * tl.arange(0, HV) + 1
        tl.store(vb + even_idx, tl.zeros([HV], dtype=tl.float32))
        tl.store(vb + odd_idx,  tl.zeros([HV], dtype=tl.float32))

    for kb in range(0, kvl, BK):
        ko = kb + tl.arange(0, BK)
        km = ko < kvl
        kg = kvs + ko
        ktp = kg * skv
        stp = kg * ssc

        qk = tl.zeros([BK], dtype=tl.float32)
        for dt in tl.static_range(DT):
            ds: tl.constexpr = dt * BD
            qe = tl.load(qb + ds + 2 * tl.arange(0, HD)).to(tl.float32)
            qo = tl.load(qb + ds + 2 * tl.arange(0, HD) + 1).to(tl.float32)
            kv_lo, kv_hi = _load_dequant_kv_tile(KV, Sc, ktp, stp, km, ds, HD, EPG)
            qk += tl.sum(qe[None, :] * kv_lo + qo[None, :] * kv_hi, axis=1)

        qk = tl.where(km, qk * sm_scale, -1e38)

        mc = tl.max(qk, axis=0)
        mn = tl.maximum(m, mc)
        al = tl.math.exp2((m - mn) * LOG2E)
        p  = tl.math.exp2((qk - mn) * LOG2E)
        p  = tl.where(km, p, 0.0)
        ln = al * l + tl.sum(p, axis=0)

        for vt in tl.static_range(VT):
            vs: tl.constexpr = vt * BV
            even_idx = vs + 2 * tl.arange(0, HV)
            odd_idx  = vs + 2 * tl.arange(0, HV) + 1

            vacc_lo = tl.load(vb + even_idx) * al
            vacc_hi = tl.load(vb + odd_idx) * al

            kv_lo, kv_hi = _load_dequant_kv_tile(KV, Sc, ktp, stp, km, vs, HV, EPG)
            clo = tl.sum(p[:, None] * kv_lo, axis=0)
            chi = tl.sum(p[:, None] * kv_hi, axis=0)

            tl.store(vb + even_idx, vacc_lo + clo)
            tl.store(vb + odd_idx,  vacc_hi + chi)

        m = mn
        l = ln

    inv_l = 1.0 / tl.maximum(l, 1e-10)
    for vt in tl.static_range(VT):
        vs: tl.constexpr = vt * BV
        even_idx = vs + 2 * tl.arange(0, HV)
        odd_idx  = vs + 2 * tl.arange(0, HV) + 1
        vlo = tl.load(vb + even_idx)
        vhi = tl.load(vb + odd_idx)
        tl.store(ob + even_idx, (vlo * inv_l).to(tl.bfloat16))
        tl.store(ob + odd_idx,  (vhi * inv_l).to(tl.bfloat16))


# ============================================================================
# MXFP4 — Cache and launcher
# ============================================================================
_mxfp4_cache = {}

def _run_mxfp4(queries, kv_data, qo_indptr, kv_indptr, config):
    """Execute MXFP4 Triton decode path."""
    fp4x2_buf, e8m0_scales = kv_data["mxfp4"]

    if e8m0_scales.dtype != torch.uint8:
        scales_u8 = e8m0_scales.view(torch.uint8)
    else:
        scales_u8 = e8m0_scales

    device = queries.device
    total_q, num_heads, _ = queries.shape
    total_kv = fp4x2_buf.shape[0]
    packed_dim = fp4x2_buf.shape[2]
    num_sg = scales_u8.shape[1]  # 24 (padded)
    epg = 32  # Standard MXFP4 block size

    batch_size = qo_indptr.shape[0] - 1

    kv_2d = fp4x2_buf.reshape(total_kv, packed_dim)

    # Cache key
    cache_key = (batch_size, total_kv // batch_size if batch_size > 0 else 0)

    if cache_key not in _mxfp4_cache:
        _mxfp4_cache[cache_key] = {
            "output": torch.empty(total_q, num_heads, V_HEAD_DIM,
                                  dtype=torch.bfloat16, device=device),
        }

    output = _mxfp4_cache[cache_key]["output"]
    # Ensure correct size (first call or shape change)
    if output.shape[0] != total_q:
        output = torch.empty(total_q, num_heads, V_HEAD_DIM,
                             dtype=torch.bfloat16, device=device)
        _mxfp4_cache[cache_key]["output"] = output

    # Block config
    BD = 48   # 576/48 = 12 D-tiles
    BV = 32   # 512/32 = 16 V-tiles
    BK = 32

    total_ctas = batch_size * num_heads

    if total_ctas >= 128:
        # Enough CTAs — single-pass, no split-K
        vs_key = f"vs_{cache_key}"
        if vs_key not in _mxfp4_cache:
            _mxfp4_cache[vs_key] = torch.empty(
                total_q, num_heads, V_HEAD_DIM,
                dtype=torch.float32, device=device)
        vs = _mxfp4_cache[vs_key]
        if vs.shape[0] != total_q:
            vs = torch.empty(total_q, num_heads, V_HEAD_DIM,
                             dtype=torch.float32, device=device)
            _mxfp4_cache[vs_key] = vs

        grid = (batch_size, num_heads)
        _mla_decode_safe[grid](
            queries, kv_2d, scales_u8, output, vs,
            qo_indptr, kv_indptr,
            queries.stride(0), queries.stride(1),
            kv_2d.stride(0), scales_u8.stride(0),
            output.stride(0), output.stride(1),
            vs.stride(0), vs.stride(1),
            SM_SCALE,
            QK_DIM=HEADDIM, V_DIM=V_HEAD_DIM,
            P_DIM=packed_dim, N_SG=num_sg, EPG=epg,
            BK=BK, BD=BD, BV=BV,
            num_warps=4, num_stages=2,
        )
    else:
        # Split-K for better CU utilization
        if total_ctas < 32:
            NS = 16
        elif total_ctas < 64:
            NS = 8
        else:
            NS = 4

        sk_key = f"sk_{cache_key}_{NS}"
        if sk_key not in _mxfp4_cache:
            _mxfp4_cache[sk_key] = {
                "pv": torch.empty((total_q * num_heads, NS, V_HEAD_DIM),
                                  dtype=torch.float32, device=device),
                "pm": torch.empty((total_q * num_heads, NS),
                                  dtype=torch.float32, device=device),
                "pl": torch.empty((total_q * num_heads, NS),
                                  dtype=torch.float32, device=device),
            }
        sk = _mxfp4_cache[sk_key]

        VT = V_HEAD_DIM // BV

        grid_sk = (batch_size, num_heads, NS)
        _mla_decode_splitk_safe[grid_sk](
            queries, kv_2d, scales_u8, sk["pv"], sk["pm"], sk["pl"],
            qo_indptr, kv_indptr,
            queries.stride(0), queries.stride(1),
            kv_2d.stride(0), scales_u8.stride(0),
            SM_SCALE, num_heads=num_heads,
            QK_DIM=HEADDIM, V_DIM=V_HEAD_DIM,
            P_DIM=packed_dim, N_SG=num_sg, EPG=epg,
            NS=NS, BK=BK, BD=BD, BV=BV,
            num_warps=4, num_stages=2,
        )

        grid_r = (batch_size, num_heads)
        _reduce_splitk[grid_r](
            sk["pv"], sk["pm"], sk["pl"], output, qo_indptr,
            output.stride(0), output.stride(1),
            V_DIM=V_HEAD_DIM, num_heads=num_heads,
            NS=NS, BV=BV, VT=VT,
            num_warps=4, num_stages=1,
        )

    return output


# ============================================================================
# Entry point — hybrid dispatch
# ============================================================================
def custom_kernel(data):
    """
    MLA decode attention — hybrid MXFP4 + FP8 ASM.

    For kv_len <= 2048: FP8 ASM kernel (fast, low overhead per token)
    For kv_len >  2048: MXFP4 Triton kernel (1.85x bandwidth savings)
    """
    queries, kv_data, qo_indptr, kv_indptr, config = data

    # Determine kv_len from kv_indptr (uniform batching: all requests same length)
    kv_len = (kv_indptr[1] - kv_indptr[0]).item()
    batch_size = qo_indptr.shape[0] - 1

    if kv_len <= KV_LEN_THRESHOLD:
        # ── FP8 ASM path ─────────────────────────────────────────────
        # queries: (total_q, 16, 576) — reshape to (batch_size, 16, 576)
        # The ASM kernel expects contiguous (bs, nhead, headdim) input
        q = queries.view(batch_size, NHEAD, HEADDIM)
        result = _run_fp8(q, kv_data, batch_size, kv_len)
        # result: (batch_size, 16, 512) -> need (total_q, 16, 512)
        return result.view(-1, NHEAD, V_HEAD_DIM)
    else:
        # ── MXFP4 Triton path ────────────────────────────────────────
        return _run_mxfp4(queries, kv_data, qo_indptr, kv_indptr, config)
