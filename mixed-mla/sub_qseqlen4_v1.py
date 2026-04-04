#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — qseqlen=4 batch folding.
Group 4 batch entries into qseqlen=4, dispatching to
mla_a8w8_qh64_qseqlen4_gqaratio16_ps.co kernel.
This is 4x more work per wavefront → much better CU utilization.
Likely how Borui Xu achieves 12.7μs.

For bs not divisible by 4, fall back to standard qseqlen=1.
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


def _build_meta_qseqlen4(batch_size_grouped, kv_seq_len, nq_effective,
                         nkv, num_kv_splits, page_size, dtype_q,
                         qo_indptr, kv_indptr):
    """Build metadata for qseqlen=4 grouped batches."""
    total_kv = int(kv_indptr[-1].item())

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

    kv_gran = max(1, 16 // page_size)
    q_seq_len = 4  # Grouped qseqlen

    info = get_mla_metadata_info_v1(
        batch_size_grouped, q_seq_len, nq_effective, dtype_q, FP8_DTYPE,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr_pages, kv_last_page_len,
        nq_effective // nkv, nkv, True,
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


def _build_meta_standard(batch_size, kv_seq_len, q_seq_len, nq, nkv,
                         num_kv_splits, page_size, dtype_q, qo_indptr, kv_indptr):
    """Standard metadata builder (same as sub_pg2_bf16q)."""
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


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    # Try qseqlen=4 folding for batch sizes divisible by 4
    use_qseqlen4 = (batch_size % 4 == 0 and batch_size >= 4)

    if use_qseqlen4:
        # === QSEQLEN=4 PATH ===
        # Group 4 consecutive batch entries into qseqlen=4
        # This dispatches to mla_a8w8_qh64_qseqlen4 kernel
        bs_grouped = batch_size // 4
        nq_effective = nq  # Keep 16 heads, set q_seq_len=4

        # Reshape Q: (bs*1, 16, 576) → (bs/4, 4*16, 576) = (bs/4, 64, 576)
        # The fold mechanism in mla.py will convert nhead=64 → fold_factor=4 → nhead=16, total_s*=4
        q_fp8_shape = (bs_grouped, nq * 4, dq)

        # Build grouped qo_indptr: each group has qseqlen=4
        qo_indptr_grouped = torch.arange(
            0, bs_grouped + 1, dtype=torch.int32, device="cuda") * 4

        # Build grouped kv_indptr: each group has 4 KV sequences concatenated
        kv_indptr_grouped = torch.arange(
            0, bs_grouped + 1, dtype=torch.int32, device="cuda") * (4 * kv_seq_len)

        page_size = 1
        num_kv_splits = 8 if total_kv <= 8192 else 16

        cache_key = ("qseqlen4", batch_size, kv_seq_len, num_kv_splits, page_size)
        if cache_key not in _meta_cache:
            _meta_cache[cache_key] = _build_meta_qseqlen4(
                bs_grouped, kv_seq_len, nq * 4, nkv,
                num_kv_splits, page_size, FP8_DTYPE,
                qo_indptr_grouped, kv_indptr_grouped)

        (wm, wi, wis, ri, rfm, rpm,
         kv_indices, kv_last_page_len, kv_indptr_pages, ps) = _meta_cache[cache_key]

        kv_buffer_4d = kv_buffer_fp8.view(-1, ps, nkv, kv_buffer_fp8.shape[-1])

        # Quantize Q to fp8
        alloc_key = ("qseqlen4_fp8", bs_grouped, nq * 4, dv, dq)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = (
                torch.empty((bs_grouped, nq * 4, dv), dtype=BF16, device="cuda"),
                torch.zeros(1, dtype=torch.float32, device="cuda"),
                torch.empty(1, dtype=torch.float32, device="cuda"),
                torch.empty(batch_size * nq * dq, dtype=FP8_DTYPE, device="cuda"),
            )
        o_grouped, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

        N = q.numel()
        BLOCK = 4096
        grid = ((N + BLOCK - 1) // BLOCK,)
        amax_buf.zero_()
        _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
        _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                               FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)

        # Reshape fp8 Q: (bs*16*576) → (bs/4, 64, 576)
        q_fp8_reshaped = q_fp8_flat.view(batch_size, nq, dq).view(bs_grouped, nq * 4, dq)

        mla_decode_fwd(
            q_fp8_reshaped, kv_buffer_4d, o_grouped,
            qo_indptr_grouped, kv_indptr_pages, kv_indices, kv_last_page_len,
            4,  # q_seq_len = 4 for grouped
            page_size=ps, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            q_scale=scale_buf, kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )

        # Reshape output: (bs/4, 64, 512) → (bs, 16, 512)
        return o_grouped.view(batch_size, nq, dv)

    else:
        # === STANDARD PATH (pg2+bf16Q for kv<=1024, pg8+fp8Q for kv>=8192) ===
        if kv_seq_len <= 1024:
            page_size = 2
            dtype_q = BF16
            use_fp8_q = False
        else:
            page_size = 8
            dtype_q = FP8_DTYPE
            use_fp8_q = True

        num_kv_splits = 8 if total_kv <= 8192 else 16

        cache_key = ("std", batch_size, kv_seq_len, num_kv_splits, page_size, use_fp8_q)
        if cache_key not in _meta_cache:
            _meta_cache[cache_key] = _build_meta_standard(
                batch_size, kv_seq_len, q_seq_len, nq, nkv,
                num_kv_splits, page_size, dtype_q, qo_indptr, kv_indptr)

        (wm, wi, wis, ri, rfm, rpm,
         kv_indices, kv_last_page_len, kv_indptr_pages, ps) = _meta_cache[cache_key]

        kv_buffer_4d = kv_buffer_fp8.view(-1, ps, nkv, kv_buffer_fp8.shape[-1])

        if use_fp8_q:
            alloc_key = ("std_fp8", q.shape[0], nq, dv, dq)
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
            alloc_key = ("std_bf16", q.shape[0], nq, dv)
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
