#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — a16w8 + pg2_fix: bf16 Q (skip quant) + page_size=2 with CORRECT metadata.
a16w8 has BETTER accuracy than a8w8 (no Q quantization error).
This means pg2 mismatch should be LOWER, making it safer to pass accuracy.
Key fix: kv_granularity = max(1, 16 // page_size) = 8
"""
import torch
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
PAGE_SIZE = 2

_meta_cache = {}
_alloc_cache = {}


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    total_kv = batch_size * kv_seq_len
    if total_kv <= 4096:
        num_kv_splits = 8
    else:
        num_kv_splits = 16

    cache_key = (batch_size, kv_seq_len, num_kv_splits)
    if cache_key not in _meta_cache:
        # pg2_fix: compute paged kv_indptr and kv_last_page_lens
        seq_lens_kv = kv_indptr[1:] - kv_indptr[:-1]
        num_pages_per_req = (seq_lens_kv + PAGE_SIZE - 1) // PAGE_SIZE
        kv_indptr_paged = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        kv_indptr_paged[1:] = torch.cumsum(num_pages_per_req, dim=0)

        kv_last_page_lens = (seq_lens_kv % PAGE_SIZE).to(torch.int32)
        kv_last_page_lens = torch.where(kv_last_page_lens == 0, PAGE_SIZE, kv_last_page_lens)

        total_pages = int(kv_indptr_paged[-1].item())

        # KEY FIX: kv_granularity = max(1, 16 // page_size) = 8
        kv_granularity = max(1, 16 // PAGE_SIZE)

        # a16w8: bf16 Q dtype
        info = get_mla_metadata_info_v1(
            batch_size, q_seq_len, nq, torch.bfloat16, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr_paged, kv_last_page_lens,
            nq // nkv, nkv, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE,
            kv_granularity=kv_granularity,
            max_seqlen_qo=q_seq_len,
            uni_seqlen_qo=q_seq_len,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=torch.bfloat16,
            dtype_kv=FP8_DTYPE,
        )

        kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
        _meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_lens, kv_indptr_paged)

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_lens, kv_indptr_paged) = _meta_cache[cache_key]

    alloc_key = (q.shape[0], nq, dv)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
    o = _alloc_cache[alloc_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    num_pages = total_kv // PAGE_SIZE
    kv_buffer_4d = kv_buffer_fp8.view(num_pages, PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    # a16w8: pass bf16 Q directly, no quantization
    mla_decode_fwd(
        q, kv_buffer_4d, o,
        qo_indptr, kv_indptr_paged, kv_indices, kv_last_page_lens,
        q_seq_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=sm_scale,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=None,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )
    return o
