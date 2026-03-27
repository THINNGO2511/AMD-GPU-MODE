#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Hybrid pg2 — Best of all worlds:
- kv=1024: bf16 Q (a16w8) + pg1 → no quant overhead, safe accuracy
- kv=8192: fp8 Q (a8w8) + pg2 → fp8 saves bandwidth, pg2 halves pages (1.4% mismatch = safe)

Expected improvements vs current 59.8us:
- kv=1024 cases: save 5-10% from skipping Q quant
- kv=8192 cases: save ~28% from pg2 page halving
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


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    # === Decision logic ===
    # kv=1024: a16w8 (bf16 Q, skip quant) + pg1 (safe accuracy)
    # kv=8192: a8w8 (fp8 Q, less bandwidth) + pg2 (halves pages, 1.4% mismatch = safe)
    use_large_path = (kv_seq_len >= 8192)

    if use_large_path:
        # --- LARGE KV: fp8 Q + pg2 ---
        page_size = 2
        q_dtype = FP8_DTYPE
        num_pages = total_kv // page_size

        if total_kv <= 8192:
            num_kv_splits = 8
        else:
            num_kv_splits = 16

        cache_key = ("L", batch_size, kv_seq_len, num_kv_splits)
        if cache_key not in _meta_cache:
            kv_indptr_pages = kv_indptr // page_size
            seq_lens = kv_indptr[1:] - kv_indptr[:-1]
            kv_last_page_len = (seq_lens % page_size).to(torch.int32)
            kv_last_page_len = torch.where(kv_last_page_len == 0, page_size, kv_last_page_len)

            info = get_mla_metadata_info_v1(
                batch_size, q_seq_len, nq, FP8_DTYPE, FP8_DTYPE,
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
                kv_granularity=max(page_size, 16),
                max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
                fast_mode=False, max_split_per_batch=num_kv_splits,
                intra_batch_mode=True, dtype_q=FP8_DTYPE, dtype_kv=FP8_DTYPE,
            )

            kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
            _meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm,
                                       kv_indices, kv_last_page_len, kv_indptr_pages)

        (wm, wi, wis, ri, rfm, rpm,
         kv_indices, kv_last_page_len, kv_indptr_pages) = _meta_cache[cache_key]

        alloc_key = ("L", q.shape[0], nq, dv, dq)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = (
                torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda"),
                torch.zeros(1, dtype=torch.float32, device="cuda"),
                torch.empty(1, dtype=torch.float32, device="cuda"),
                torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda"),
            )
        o, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

        # Fused fp8 quant
        N = q.numel()
        BLOCK = 4096
        grid = ((N + BLOCK - 1) // BLOCK,)
        amax_buf.zero_()
        _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
        _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                               FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)

        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        kv_buffer_4d = kv_buffer_fp8.view(-1, page_size, nkv, kv_buffer_fp8.shape[-1])

        mla_decode_fwd(
            q_fp8_flat.view(q.shape[0], nq, dq), kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=page_size, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            q_scale=scale_buf, kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
        return o

    else:
        # --- SMALL KV: bf16 Q + pg1 (no quant, safe accuracy) ---
        page_size = 1

        if total_kv <= 8192:
            num_kv_splits = 8
        else:
            num_kv_splits = 16

        cache_key = ("S", batch_size, kv_seq_len, num_kv_splits)
        if cache_key not in _meta_cache:
            kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

            info = get_mla_metadata_info_v1(
                batch_size, q_seq_len, nq, BF16, FP8_DTYPE,
                is_sparse=False, fast_mode=False,
                num_kv_splits=num_kv_splits, intra_batch_mode=True,
            )
            work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
            (wm, wi, wis, ri, rfm, rpm) = work

            get_mla_metadata_v1(
                qo_indptr, kv_indptr, kv_last_page_len,
                nq // nkv, nkv, True,
                wm, wis, wi, ri, rfm, rpm,
                page_size=page_size,
                kv_granularity=max(page_size, 16),
                max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
                fast_mode=False, max_split_per_batch=num_kv_splits,
                intra_batch_mode=True, dtype_q=BF16, dtype_kv=FP8_DTYPE,
            )

            kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
            _meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm,
                                       kv_indices, kv_last_page_len)

        (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len) = _meta_cache[cache_key]

        alloc_key = ("S", q.shape[0], nq, dv)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda")
        o = _alloc_cache[alloc_key]

        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        kv_buffer_4d = kv_buffer_fp8.view(
            kv_buffer_fp8.shape[0], page_size, nkv, kv_buffer_fp8.shape[-1]
        )

        mla_decode_fwd(
            q, kv_buffer_4d, o,
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
            q_seq_len, page_size=page_size, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
        return o
