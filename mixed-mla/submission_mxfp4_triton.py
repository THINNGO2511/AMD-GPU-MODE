#!/usr/bin/env python3
"""
MXFP4 MLA Decode Attention Kernel for AMD MI355X (gfx950)
GPU MODE Hackathon - Phase 1 Qualifier

Uses MXFP4 KV cache (E2M1 FP4 + E8M0 scales) for 1.85x bandwidth savings.
Single Triton kernel with online softmax and V accumulation via scratch buffer.
Split-K variant for small-batch CU utilization.
"""

import torch
import triton
import triton.language as tl
import math

# ============================================================================
# FP4 E2M1 decode (inline JIT helper)
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
# Main decode kernel (single-pass, scratch buffer for V)
# ============================================================================

@triton.jit
def _mla_decode(
    Q, KV, Sc, O, VS,
    qo_ind, kv_ind,
    sq0, sq1, skv, ssc, so0, so1, svs0, svs1,
    sm_scale,
    QK_DIM: tl.constexpr,   # 576
    V_DIM: tl.constexpr,    # 512
    P_DIM: tl.constexpr,    # 288
    N_SG: tl.constexpr,     # 24
    EPG: tl.constexpr,      # 24
    BK: tl.constexpr,       # BLOCK_KV
    BD: tl.constexpr,       # BLOCK_D (for QK, must divide 576, be even)
    BV: tl.constexpr,       # BLOCK_V (for V, must divide 512, be even)
):
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

    # Zero V scratch
    for vt in tl.static_range(VT):
        off = vt * BV
        tl.store(vb + off + tl.arange(0, BV), tl.zeros([BV], dtype=tl.float32))

    for kb in range(0, kvl, BK):
        ko = kb + tl.arange(0, BK)
        km = ko < kvl
        kg = kvs + ko
        ktp = kg * skv
        stp = kg * ssc

        # === QK dot product ===
        qk = tl.zeros([BK], dtype=tl.float32)
        for dt in tl.static_range(DT):
            ds = dt * BD
            qe = tl.load(qb + ds + 2 * tl.arange(0, HD)).to(tl.float32)
            qo = tl.load(qb + ds + 2 * tl.arange(0, HD) + 1).to(tl.float32)
            kv_lo, kv_hi = _load_dequant_kv_tile(KV, Sc, ktp, stp, km, ds, HD, EPG)
            qk += tl.sum(qe[None, :] * kv_lo + qo[None, :] * kv_hi, axis=1)

        qk = tl.where(km, qk * sm_scale, -1e38)

        # === Online softmax ===
        mc = tl.max(qk, axis=0)
        mn = tl.maximum(m, mc)
        al = tl.math.exp2((m - mn) * LOG2E)
        p  = tl.math.exp2((qk - mn) * LOG2E)
        p  = tl.where(km, p, 0.0)
        ln = al * l + tl.sum(p, axis=0)

        # === V accumulation ===
        for vt in tl.static_range(VT):
            vs = vt * BV
            vacc = tl.load(vb + vs + tl.arange(0, BV))
            vacc = vacc * al
            kv_lo, kv_hi = _load_dequant_kv_tile(KV, Sc, ktp, stp, km, vs, HV, EPG)
            clo = tl.sum(p[:, None] * kv_lo, axis=0)
            chi = tl.sum(p[:, None] * kv_hi, axis=0)
            # Interleave even/odd: [lo0, hi0, lo1, hi1, ...]
            contrib = tl.interleave(clo, chi)
            tl.store(vb + vs + tl.arange(0, BV), vacc + contrib)

        m = mn
        l = ln

    # === Normalize and write output ===
    inv_l = 1.0 / tl.maximum(l, 1e-10)
    for vt in tl.static_range(VT):
        vs = vt * BV
        vacc = tl.load(vb + vs + tl.arange(0, BV))
        tl.store(ob + vs + tl.arange(0, BV), (vacc * inv_l).to(tl.bfloat16))


# ============================================================================
# Split-K kernel (for small batch / long sequence)
# ============================================================================

@triton.jit
def _mla_decode_splitk(
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

    for vt in tl.static_range(VT):
        off = vt * BV
        tl.store(pvb + off + tl.arange(0, BV), tl.zeros([BV], dtype=tl.float32))

    for kb in range(0, mkl, BK):
        ko = kb + tl.arange(0, BK)
        km = ko < mkl
        kg = mks + ko
        ktp = kg * skv
        stp = kg * ssc

        qk = tl.zeros([BK], dtype=tl.float32)
        for dt in tl.static_range(DT):
            ds = dt * BD
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
            vs = vt * BV
            vacc = tl.load(pvb + vs + tl.arange(0, BV))
            vacc = vacc * al
            kv_lo, kv_hi = _load_dequant_kv_tile(KV, Sc, ktp, stp, km, vs, HV, EPG)
            clo = tl.sum(p[:, None] * kv_lo, axis=0)
            chi = tl.sum(p[:, None] * kv_hi, axis=0)
            tl.store(pvb + vs + tl.arange(0, BV), vacc + tl.interleave(clo, chi))

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
        vs = vt * BV
        vo = tl.zeros([BV], dtype=tl.float32)
        for s in tl.static_range(NS):
            ms = tl.load(pmb + s)
            vs_data = tl.load(pvb + s * V_DIM + vs + tl.arange(0, BV))
            vo += vs_data * tl.math.exp2((ms - mg) * LOG2E)
        tl.store(ob + vs + tl.arange(0, BV), (vo * inv_l).to(tl.bfloat16))


# ============================================================================
# Fallback kernel without tl.interleave (in case Triton version doesn't have it)
# Uses tl.store with explicit even/odd indexing
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
    """Same as _mla_decode but uses explicit scatter instead of tl.interleave."""
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

    # Zero V scratch (use even/odd stores)
    for vt in tl.static_range(VT):
        off = vt * BV
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
            ds = dt * BD
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
            vs = vt * BV
            even_idx = vs + 2 * tl.arange(0, HV)
            odd_idx  = vs + 2 * tl.arange(0, HV) + 1
            
            vacc_lo = tl.load(vb + even_idx)  # (HV,) even positions
            vacc_hi = tl.load(vb + odd_idx)   # (HV,) odd positions
            vacc_lo = vacc_lo * al
            vacc_hi = vacc_hi * al

            kv_lo, kv_hi = _load_dequant_kv_tile(KV, Sc, ktp, stp, km, vs, HV, EPG)
            clo = tl.sum(p[:, None] * kv_lo, axis=0)  # (HV,)
            chi = tl.sum(p[:, None] * kv_hi, axis=0)  # (HV,)

            tl.store(vb + even_idx, vacc_lo + clo)
            tl.store(vb + odd_idx,  vacc_hi + chi)

        m = mn
        l = ln

    inv_l = 1.0 / tl.maximum(l, 1e-10)
    for vt in tl.static_range(VT):
        vs = vt * BV
        even_idx = vs + 2 * tl.arange(0, HV)
        odd_idx  = vs + 2 * tl.arange(0, HV) + 1
        vlo = tl.load(vb + even_idx)
        vhi = tl.load(vb + odd_idx)
        tl.store(ob + even_idx, (vlo * inv_l).to(tl.bfloat16))
        tl.store(ob + odd_idx,  (vhi * inv_l).to(tl.bfloat16))


# ============================================================================
# Split-K safe variant (no tl.interleave)
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

    for vt in tl.static_range(VT):
        off = vt * BV
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
            ds = dt * BD
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
            vs = vt * BV
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


# ============================================================================
# Launcher
# ============================================================================

_scratch = {}

def _get_scratch(key, shape, dtype, device):
    """Get or allocate a scratch buffer."""
    global _scratch
    if key not in _scratch or _scratch[key].shape != shape:
        _scratch[key] = torch.empty(shape, dtype=dtype, device=device)
    return _scratch[key]


def custom_kernel(data):
    """
    MLA decode attention with MXFP4 KV cache.
    Entry point matching the eval harness.
    """
    queries, kv_data, qo_indptr, kv_indptr, config = data

    fp4x2_buf, e8m0_scales = kv_data["mxfp4"]

    # Must view as uint8 — Triton doesn't recognize fp4x2 dtype
    if fp4x2_buf.dtype != torch.uint8:
        fp4x2_buf = fp4x2_buf.view(torch.uint8)
    if e8m0_scales.dtype != torch.uint8:
        scales_u8 = e8m0_scales.view(torch.uint8)
    else:
        scales_u8 = e8m0_scales

    sm_scale    = config.get('sm_scale', 1.0 / math.sqrt(576))
    qk_head_dim = config.get('qk_head_dim', 576)
    v_head_dim  = config.get('v_head_dim', 512)

    device = queries.device
    total_q, num_heads, _ = queries.shape
    total_kv = fp4x2_buf.shape[0]
    packed_dim = fp4x2_buf.shape[2]
    num_sg = scales_u8.shape[1]  # 24 (padded from 18 actual groups)
    # MXFP4 standard: block_size=32 elements per group, 576/32=18 actual groups
    # Scale tensor is padded to 24 columns. Use actual group size, not padded count.
    epg = 32  # standard MXFP4 block size — NOT qk_head_dim // num_sg
    batch_size = qo_indptr.shape[0] - 1

    kv_2d = fp4x2_buf.reshape(total_kv, packed_dim)
    output = torch.empty(total_q, num_heads, v_head_dim,
                         dtype=torch.bfloat16, device=device)

    # Block config: 576/48=12 D-tiles, 512/32=16 V-tiles
    BD = 64
    BV = 32
    BK = 32

    # Determine split-K
    total_ctas = batch_size * num_heads

    if total_ctas >= 128:
        # Enough CTAs, no split-K needed
        vs = _get_scratch('vs', (total_q, num_heads, v_head_dim),
                          torch.float32, device)

        grid = (batch_size, num_heads)
        try:
            _mla_decode[grid](
                queries, kv_2d, scales_u8, output, vs,
                qo_indptr, kv_indptr,
                queries.stride(0), queries.stride(1),
                kv_2d.stride(0), scales_u8.stride(0),
                output.stride(0), output.stride(1),
                vs.stride(0), vs.stride(1),
                sm_scale,
                QK_DIM=qk_head_dim, V_DIM=v_head_dim,
                P_DIM=packed_dim, N_SG=num_sg, EPG=epg,
                BK=BK, BD=BD, BV=BV,
                num_warps=4, num_stages=2,
            )
        except Exception:
            # Fallback to safe kernel (no tl.interleave)
            _mla_decode_safe[grid](
                queries, kv_2d, scales_u8, output, vs,
                qo_indptr, kv_indptr,
                queries.stride(0), queries.stride(1),
                kv_2d.stride(0), scales_u8.stride(0),
                output.stride(0), output.stride(1),
                vs.stride(0), vs.stride(1),
                sm_scale,
                QK_DIM=qk_head_dim, V_DIM=v_head_dim,
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

        pv = _get_scratch('pv', (total_q * num_heads, NS, v_head_dim),
                          torch.float32, device)
        pm = _get_scratch('pm', (total_q * num_heads, NS),
                          torch.float32, device)
        pl = _get_scratch('pl', (total_q * num_heads, NS),
                          torch.float32, device)

        grid_sk = (batch_size, num_heads, NS)
        VT = v_head_dim // BV

        try:
            _mla_decode_splitk[grid_sk](
                queries, kv_2d, scales_u8, pv, pm, pl,
                qo_indptr, kv_indptr,
                queries.stride(0), queries.stride(1),
                kv_2d.stride(0), scales_u8.stride(0),
                sm_scale, num_heads=num_heads,
                QK_DIM=qk_head_dim, V_DIM=v_head_dim,
                P_DIM=packed_dim, N_SG=num_sg, EPG=epg,
                NS=NS, BK=BK, BD=BD, BV=BV,
                num_warps=4, num_stages=2,
            )

            grid_r = (batch_size, num_heads)
            _reduce_splitk[grid_r](
                pv, pm, pl, output, qo_indptr,
                output.stride(0), output.stride(1),
                V_DIM=v_head_dim, num_heads=num_heads,
                NS=NS, BV=BV, VT=VT,
                num_warps=4, num_stages=1,
            )
        except Exception:
            _mla_decode_splitk_safe[grid_sk](
                queries, kv_2d, scales_u8, pv, pm, pl,
                qo_indptr, kv_indptr,
                queries.stride(0), queries.stride(1),
                kv_2d.stride(0), scales_u8.stride(0),
                sm_scale, num_heads=num_heads,
                QK_DIM=qk_head_dim, V_DIM=v_head_dim,
                P_DIM=packed_dim, N_SG=num_sg, EPG=epg,
                NS=NS, BK=BK, BD=BD, BV=BV,
                num_warps=4, num_stages=2,
            )

            grid_r = (batch_size, num_heads)
            _reduce_splitk[grid_r](
                pv, pm, pl, output, qo_indptr,
                output.stride(0), output.stride(1),
                V_DIM=v_head_dim, num_heads=num_heads,
                NS=NS, BV=BV, VT=VT,
                num_warps=4, num_stages=1,
            )

    return output
