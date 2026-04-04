#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — Shape-specific optimal num_kv_splits with aiter auto-selection logic.

Research-derived per-shape splits table:
- Implements aiter's get_meta_param efficiency formula: CU utilization vs reduce overhead
- MI355X: 304 CUs, overhead constant = 84.1 (from aiter source)
- For fp8 KV: clamps to max splits = ceil(avg_kv / min_block_n)
- Persistent mode throughout (proven faster than non-persistent)

Shape routing:
- kv<=1024: pg1 + bf16 Q (a16w8 kernel, no fp8 quant overhead)
- kv>=8192: pg8 + fp8 Q (a8w8 kernel, halved Q bandwidth)

Splits derivation per shape (MI355X, 304 CUs):
  bs=4, kv=1024:   auto=8, fp8_clamp=8    -> 8   (Session 9 confirmed)
  bs=4, kv=8192:   auto=16, fp8_clamp=16  -> 16  (Session 9 confirmed)
  bs=32, kv=1024:  auto=8, fp8_clamp=8    -> 8   (Session 9 confirmed)
  bs=32, kv=8192:  auto=16, fp8_clamp=16  -> 16  (Session 9 confirmed)
  bs=64, kv=1024:  auto=4, clamp=8->4     -> 16  (BUT sweep says 16 wins!)
  bs=64, kv=8192:  auto=4, clamp=64->4    -> 16  (BUT sweep says 16 wins!)
  bs=256, kv=1024: auto=1, clamp=8->1     -> 16  (sweep says 16 wins!)
  bs=256, kv=8192: auto=1, clamp=64->1    -> 16  (sweep says 16 wins!)

Note: aiter auto-formula optimizes for NON-PERSISTENT mode (CU occupancy).
In PERSISTENT mode, the metadata scheduler handles distribution differently,
so the sweep-based values (mostly 16) are better than formula outputs.

Session 9 sweep data:
  (4, 1024):   splits=8  -> 32.0us
  (4, 8192):   splits=16 -> 27.1us
  (32, 1024):  splits=8  -> 34.0us
  (32, 8192):  splits=16 -> 34.3us
  (64, 1024):  splits=16 -> 34.7us
  (64, 8192):  splits=16 -> 42.7us
  (256, 1024): splits=16 -> 52.5us
  (256, 8192): splits=16 -> 87.3us
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
_meta_cache = {}
_alloc_cache = {}

# -------------------------------------------------------------------------
# Per-shape optimal splits table (from Session 9 MI355X sweep data)
# -------------------------------------------------------------------------
# Key: (batch_size, kv_seq_len) -> num_kv_splits
# Session 9 sweep tested [2, 4, 8, 12, 16, 24, 32] and found:
#   - 16 optimal or tied for ALL shapes except bs<=32 kv=1024 where 8 ties/wins
#   - 32 always worse (reduction overhead outweighs parallelism)
#   - 2/4 significantly worse for kv=8192 (too few splits, underutilize CUs)
# For shapes not in table, fall back to aiter auto-formula
_OPTIMAL_SPLITS = {
    # kv=1024 shapes: bf16 Q, pg1
    (4, 1024): 8,       # 32.0us -- 8 tied with 16, less reduce overhead
    (32, 1024): 8,      # 34.0us -- 8 slightly faster than 16
    (64, 1024): 16,     # 34.7us -- 16 clearly wins at this batch size
    (256, 1024): 16,    # 52.5us -- 16 clearly wins

    # kv=8192 shapes: fp8 Q, pg8
    (4, 8192): 16,      # 27.1us -- 16 is clear winner
    (32, 8192): 16,     # 34.3us -- 16 is clear winner
    (64, 8192): 16,     # 42.7us -- 16 is clear winner
    (256, 8192): 16,    # 87.3us -- 16 is clear winner
}


def _auto_splits(batch_size, kv_seq_len, nq, max_seqlen_q=1):
    """
    Replicate aiter's get_meta_param auto-split formula.
    Used as fallback for shapes not in the sweep table.

    Formula (from aiter/mla.py):
      score(i) = bs*i / ceil(bs*i/cu_num)*cu_num * avg_kv / (avg_kv + overhead*i)
    This balances CU utilization (first term) vs reduce overhead (second term).

    Then for fp8: clamp to max splits = ceil(avg_kv / min_block_n)
    where min_block_n depends on nhead * max_seqlen_q.
    """
    cu_num = 304  # MI355X
    avg_kv = kv_seq_len  # uniform lengths
    overhead = 84.1

    # Score each candidate 1..16
    best_score, best_i = 0.0, 1
    for i in range(1, 17):
        total_work = batch_size * i
        rounded = ((total_work + cu_num - 1) // cu_num) * cu_num
        utilization = total_work / rounded
        kv_efficiency = avg_kv / (avg_kv + overhead * i)
        score = utilization * kv_efficiency
        if score > best_score:
            best_score = score
            best_i = i

    # fp8 min_block_n clamp
    nq_seq = int(nq * max_seqlen_q)
    block_n_map = {16: 128, 32: 128, 48: 64, 64: 64,
                   128: 32, 256: 32, 384: 32, 512: 32}
    min_block_n = block_n_map.get(nq_seq, 32)
    max_splits = (kv_seq_len + min_block_n - 1) // min_block_n
    best_i = min(best_i, max_splits)

    if best_i > 1:
        # Additional clamp for decode: kv_seq_len minus q overlap
        best_i = min(best_i, int(abs(kv_seq_len - max_seqlen_q) // min_block_n + 1))

    return max(1, best_i)


def _get_optimal_splits(batch_size, kv_seq_len, nq):
    """Get optimal num_kv_splits for this shape."""
    key = (batch_size, kv_seq_len)
    if key in _OPTIMAL_SPLITS:
        return _OPTIMAL_SPLITS[key]
    # Fallback: use auto formula but prefer 16 for persistent mode
    auto = _auto_splits(batch_size, kv_seq_len, nq)
    # In persistent mode, 16 generally works well when auto >= 4
    if auto >= 4:
        return 16
    return max(auto, 8)


# -------------------------------------------------------------------------
# Fused Q fp8 quantization (2 Triton kernels instead of 6 PyTorch ops)
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# Persistent metadata builder
# -------------------------------------------------------------------------
def _build_meta(batch_size, kv_seq_len, q_seq_len, nq, nkv,
                num_kv_splits, page_size, dtype_q, qo_indptr, kv_indptr):
    total_kv = batch_size * kv_seq_len

    if page_size == 1:
        num_pages = total_kv
        kv_indptr_pages = kv_indptr
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = seq_lens.to(torch.int32)
    else:
        num_pages = total_kv // page_size
        kv_indptr_pages = kv_indptr // page_size
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = (seq_lens % page_size).to(torch.int32)
        kv_last_page_len = torch.where(kv_last_page_len == 0, page_size, kv_last_page_len)

    # kv_granularity: CRITICAL formula from PR #1950
    kv_gran = max(1, 16 // page_size)

    info = get_mla_metadata_info_v1(
        batch_size, q_seq_len, nq, dtype_q, FP8_DTYPE,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr_pages, kv_last_page_len,
        nq // nkv, nkv, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=page_size,
        kv_granularity=kv_gran,
        max_seqlen_qo=q_seq_len,
        uni_seqlen_qo=q_seq_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=dtype_q,
        dtype_kv=FP8_DTYPE,
    )

    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages, page_size)


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    # ---- Shape routing ----
    # kv<=1024: pg1 + bf16 Q (a16w8 — no fp8 quant overhead, safe accuracy)
    # kv>=8192: pg8 + fp8 Q (a8w8 — halved Q bandwidth for large KV)
    if kv_seq_len <= 1024:
        page_size = 1
        dtype_q = BF16
        use_fp8_q = False
    else:
        page_size = 8
        dtype_q = FP8_DTYPE
        use_fp8_q = True

    # ---- Per-shape optimal splits ----
    num_kv_splits = _get_optimal_splits(batch_size, kv_seq_len, nq)

    # ---- Build/fetch cached metadata ----
    cache_key = (batch_size, kv_seq_len, num_kv_splits, page_size, use_fp8_q)
    if cache_key not in _meta_cache:
        _meta_cache[cache_key] = _build_meta(
            batch_size, kv_seq_len, q_seq_len, nq, nkv,
            num_kv_splits, page_size, dtype_q, qo_indptr, kv_indptr)

    (wm, wi, wis, ri, rfm, rpm,
     kv_indices, kv_last_page_len, kv_indptr_pages, ps) = _meta_cache[cache_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, ps, nkv, kv_buffer_fp8.shape[-1])

    if use_fp8_q:
        # ---- fp8 Q path (kv>=8192) ----
        alloc_key = ("fp8", q.shape[0], nq, dv, dq)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = (
                torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda"),
                torch.zeros(1, dtype=torch.float32, device="cuda"),
                torch.empty(1, dtype=torch.float32, device="cuda"),
                torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda"),
            )
        o, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

        N = q.numel()
        BLOCK = 4096
        grid = ((N + BLOCK - 1) // BLOCK,)
        amax_buf.zero_()
        _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
        _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                               FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)

        mla_decode_fwd(
            q_fp8_flat.view(q.shape[0], nq, dq), kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=ps, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            q_scale=scale_buf, kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
        return o
    else:
        # ---- bf16 Q path (kv<=1024) ----
        alloc_key = ("bf16", q.shape[0], nq, dv)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = torch.empty(
                (q.shape[0], nq, dv), dtype=BF16, device="cuda")
        o = _alloc_cache[alloc_key]

        mla_decode_fwd(
            q, kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=ps, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
        return o
